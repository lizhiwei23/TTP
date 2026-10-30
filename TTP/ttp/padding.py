from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedPadResize(nn.Module):
    """Pad an image tensor spatially, then resize back to the original size."""

    def __init__(self, padding_size: int = 32, pattern: str = "zero") -> None:
        super().__init__()
        if padding_size < 0:
            raise ValueError("padding_size must be non-negative")
        if pattern not in {"zero", "white", "random"}:
            raise ValueError("pattern must be one of: zero, white, random")
        self.padding_size = padding_size
        self.pattern = pattern

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_size == 0:
            return x
        b, c, h, w = x.shape
        p = self.padding_size
        padded_h = h + 2 * p
        padded_w = w + 2 * p
        if self.pattern == "zero":
            canvas = x.new_zeros((b, c, padded_h, padded_w))
        elif self.pattern == "white":
            canvas = x.new_ones((b, c, padded_h, padded_w))
        else:
            canvas = torch.rand((b, c, padded_h, padded_w), device=x.device, dtype=x.dtype)
        canvas[:, :, p : p + h, p : p + w] = x
        return F.interpolate(canvas, size=(h, w), mode="bilinear", align_corners=False)


class TrainablePadResize(nn.Module):
    """Instance-specific trainable border used by TTP."""

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        padding_size: int = 32,
        init_max_pixel: float = 10.0,
    ) -> None:
        super().__init__()
        if padding_size <= 0:
            raise ValueError("padding_size must be positive for trainable padding")
        self.channels = channels
        self.height = height
        self.width = width
        self.padding_size = padding_size
        init_max = init_max_pixel / 255.0
        p = padding_size
        self.top = nn.Parameter(torch.empty(1, channels, p, width + 2 * p))
        self.bottom = nn.Parameter(torch.empty(1, channels, p, width + 2 * p))
        self.left = nn.Parameter(torch.empty(1, channels, height, p))
        self.right = nn.Parameter(torch.empty(1, channels, height, p))
        self.reset_parameters(init_max)

    def reset_parameters(self, init_max: float) -> None:
        for param in self.parameters():
            nn.init.uniform_(param, 0.0, init_max)

    def clamp_(self) -> None:
        with torch.no_grad():
            for param in self.parameters():
                param.clamp_(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.channels or h != self.height or w != self.width:
            raise ValueError(f"expected input shape (*, {self.channels}, {self.height}, {self.width}), got {tuple(x.shape)}")
        top = self.top.clamp(0.0, 1.0).expand(b, -1, -1, -1)
        bottom = self.bottom.clamp(0.0, 1.0).expand(b, -1, -1, -1)
        left = self.left.clamp(0.0, 1.0).expand(b, -1, -1, -1)
        right = self.right.clamp(0.0, 1.0).expand(b, -1, -1, -1)
        middle = torch.cat([left, x, right], dim=-1)
        canvas = torch.cat([top, middle, bottom], dim=-2)
        return F.interpolate(canvas, size=(h, w), mode="bilinear", align_corners=False)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    return -(probs * logits.log_softmax(dim=-1)).sum(dim=-1)


def select_low_entropy(logits: torch.Tensor, selection_p: float) -> torch.Tensor:
    if not 0 < selection_p <= 1:
        raise ValueError("selection_p must be in (0, 1]")
    count = max(1, int(round(logits.shape[0] * selection_p)))
    return torch.argsort(entropy(logits), descending=False)[:count]


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

