"""View aggregation modes. Probability mode implements manuscript Eq. 11."""
import torch


def aggregate_views(logits, weights, mode="probability"):
    if mode == "probability":
        values = logits.softmax(dim=-1)
    elif mode == "legacy_logits":
        values = logits
    else:
        raise ValueError(f"Unknown ensemble mode: {mode}")
    return torch.sum(weights[:, None] * values, dim=0, keepdim=True)
