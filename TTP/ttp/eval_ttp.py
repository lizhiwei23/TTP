from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import torchattacks  # noqa: E402
import clip.custom_clip as clip_model
from clip.custom_clip import get_coop  # noqa: E402
from data.cls_to_names import *  # noqa: F401,F403,E402
from data.datautils import AugMixAugmenter, build_dataset  # noqa: E402
from data.imagnet_prompts import imagenet_classes  # noqa: E402
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask  # noqa: E402
from ttp.ensemble import aggregate_views
from ttp.padding import FixedPadResize, TrainablePadResize, cosine_sim, entropy, select_low_entropy  # noqa: E402
from utils.tools import accuracy, set_random_seed  # noqa: E402


def resolve_classnames(dataset_name: str) -> list[str]:
    if len(dataset_name) > 1:
        return list(globals()[f"{dataset_name.lower()}_classes"])
    if dataset_name not in ["A", "R", "K", "V", "I"]:
        raise ValueError(f"Unknown dataset id: {dataset_name}")
    if dataset_name == "R":
        return [imagenet_classes[i] for i, keep in enumerate(imagenet_r_mask) if keep]
    if dataset_name == "A":
        return [imagenet_classes[i] for i in imagenet_a_mask]
    if dataset_name == "V":
        return [imagenet_classes[i] for i in imagenet_v_mask]
    return list(imagenet_classes)


def arch_slug(arch: str) -> str:
    return arch.replace("/", "-")


def atomic_torch_save(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def adv_cache_path(args: argparse.Namespace, index: int) -> Path:
    alpha = args.alpha if args.alpha is not None else args.eps / 4.0
    attack_id = f"pgd_eps{args.eps:g}_alpha{alpha:g}_steps{args.steps}_res{args.resolution}"
    return Path(args.adv_cache_dir) / args.test_sets / arch_slug(args.arch) / args.dataset_mode / f"seed_{args.seed}" / attack_id / f"{index:08d}.pt"


def build_views_transform(args: argparse.Namespace):
    base_transform = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
        ]
    )
    preprocess = transforms.Compose([transforms.ToTensor()])
    return AugMixAugmenter(
        base_transform,
        preprocess,
        n_views=args.batch_size - 1,
        augmix=len(args.test_sets) > 1,
    )


def build_base_transform(args: argparse.Namespace):
    return transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )


def as_cuda_views(images: torch.Tensor | list[torch.Tensor], gpu: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if isinstance(images, list):
        views = [view.cuda(gpu, non_blocking=True) for view in images]
    else:
        if images.dim() > 4:
            images = images.squeeze(0)
        views = [images.cuda(gpu, non_blocking=True)]
    return views[0], views


@torch.no_grad()
def image_features(model, x: torch.Tensor) -> torch.Tensor:
    features, _, _ = model.forward_features(x)
    return features


def make_or_load_adv(
    args: argparse.Namespace,
    attack,
    model,
    image: torch.Tensor,
    target: torch.Tensor,
    sample_index: int,
) -> torch.Tensor:
    cache_file = adv_cache_path(args, sample_index)
    if args.cache_adv and cache_file.exists() and not args.regen_adv:
        item = torch.load(cache_file, map_location="cpu")
        if item["sample_index"] != sample_index or int(item["target"].item()) != int(target.item()):
            raise ValueError("Attack cache sample identity mismatch; regenerate the cache")
        return item["image"].cuda(args.gpu, non_blocking=True)

    adv = attack(image, target).detach()
    if args.cache_adv:
        atomic_torch_save(
            {
                "image": adv.cpu(),
                "target": target.detach().cpu(),
                "sample_index": sample_index,
                "dataset": args.test_sets,
                "arch": args.arch,
                "eps": args.eps,
                "alpha": args.alpha,
                "steps": args.steps,
            },
            cache_file,
        )
    return adv


def tensor_to_views(image: torch.Tensor, transform) -> list[torch.Tensor]:
    pil_image = transforms.ToPILImage()(image.squeeze(0).detach().cpu().clamp(0.0, 1.0))
    return [view.unsqueeze(0) for view in transform(pil_image)]


def adapt_with_ttp(
    args: argparse.Namespace,
    model,
    image: torch.Tensor,
    views: Iterable[torch.Tensor],
    fixed_pad: FixedPadResize,
) -> torch.Tensor:
    all_views = torch.cat(list(views), dim=0)
    with torch.no_grad():
        logits = model(all_views)
        selected = select_low_entropy(logits, args.selection_p)

    selected_views = all_views[selected].detach()
    _, channels, height, width = selected_views.shape
    padder = TrainablePadResize(
        channels=channels,
        height=height,
        width=width,
        padding_size=args.padding_size,
        init_max_pixel=args.padding_init_max,
    ).cuda(args.gpu)
    optimizer = torch.optim.SGD(padder.parameters(), lr=args.padding_lr)
    optimizer.zero_grad(set_to_none=True)
    padded = padder(selected_views)
    loss = entropy(model(padded)).mean()
    loss.backward()
    optimizer.step()
    padder.clamp_()

    with torch.no_grad():
        padded_views = padder(selected_views)
        tuned_logits = model(padded_views)
        z_adv = image_features(model, image)
        z_pad_adv = image_features(model, fixed_pad(image))
        z_pad_views = image_features(model, padded_views)
        alpha = cosine_sim(z_pad_views, z_pad_adv.expand_as(z_pad_views))
        beta = cosine_sim(z_pad_views, z_adv.expand_as(z_pad_views))
        weights = torch.softmax((alpha - beta) / args.ensemble_temp, dim=0)
        return aggregate_views(tuned_logits, weights, args.ensemble_mode)


def average_prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of the mean prediction, matching the official TPT objective."""
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    mean_log_probs = log_probs.logsumexp(dim=0) - math.log(log_probs.shape[0])
    mean_log_probs = mean_log_probs.clamp_min(torch.finfo(mean_log_probs.dtype).min)
    return -(mean_log_probs * mean_log_probs.exp()).sum()


def adapt_clean_with_tpt(
    args: argparse.Namespace,
    model,
    image: torch.Tensor,
    views: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    optimizer_state: dict,
    scaler: torch.amp.GradScaler,
) -> torch.Tensor:
    """Run instance-wise TPT on a sample that passed the clean detector."""
    all_views = torch.cat(list(views), dim=0)
    selected = None

    with torch.no_grad():
        model.reset()
    optimizer.load_state_dict(optimizer_state)

    for _ in range(args.clean_tpt_steps):
        with torch.amp.autocast("cuda"):
            logits = model(all_views)
            if selected is None:
                selected = select_low_entropy(logits, args.clean_tpt_selection_p)
            loss = average_prediction_entropy(logits[selected])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    with torch.no_grad():
        tuned_logits = model(image).detach()
        model.reset()
    return tuned_logits


def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    set_random_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True

    clip_model.DOWNLOAD_ROOT = str(Path(args.model_cache_dir).expanduser().resolve())
    classnames = resolve_classnames(args.test_sets)
    model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
    model = model.cuda(args.gpu).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    clean_tpt_optimizer = None
    clean_tpt_optimizer_state = None
    clean_tpt_scaler = None
    if args.clean_strategy == "tpt":
        for param in model.prompt_learner.parameters():
            param.requires_grad_(True)
        clean_tpt_optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=args.clean_tpt_lr)
        clean_tpt_optimizer_state = deepcopy(clean_tpt_optimizer.state_dict())
        clean_tpt_scaler = torch.amp.GradScaler("cuda", init_scale=1000)

    views_transform = build_views_transform(args)
    dataset_transform = build_base_transform(args) if args.lazy_views else views_transform
    data_root = Path(args.data).resolve()
    dataset = build_dataset(args.test_sets, dataset_transform, str(data_root), mode=args.dataset_mode)

    source_indices = None
    if args.sample_order == "random":
        if args.max_samples is None:
            raise ValueError("--sample-order random requires --max-samples")
        generator = torch.Generator().manual_seed(args.sample_seed)
        source_indices = torch.randperm(len(dataset), generator=generator)[: args.max_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, source_indices)
    elif args.loader_batch_size > 1 and args.max_samples is not None:
        source_indices = list(range(min(args.max_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, source_indices)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.loader_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    fixed_pad = FixedPadResize(args.padding_size, args.padding_pattern).cuda(args.gpu)
    attack = None
    if args.eps > 0:
        if args.steps <= 0:
            raise ValueError("--steps must be positive when --eps > 0")
        alpha = args.alpha if args.alpha is not None else args.eps / 4.0
        args.alpha = alpha
        attack = torchattacks.PGD(model, eps=args.eps / 255.0, alpha=alpha / 255.0, steps=args.steps)

    attack_dir = f"eps_{args.eps:g}_steps_{args.steps}"
    output_dir = Path(args.output_dir) / args.ensemble_mode / args.clean_strategy / arch_slug(args.arch) / args.test_sets / attack_dir / f"shard_{args.shard_id:02d}_of_{args.num_shards:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output_dir}. Choose a new --output_dir.")
    config = vars(args).copy()
    source_digest = hashlib.sha256()
    for source in sorted(ROOT.rglob("*.py")):
        if any(part in ("cache", "output_results", "datasets", ".venv") for part in source.relative_to(ROOT).parts):
            continue
        source_digest.update(str(source.relative_to(ROOT)).encode())
        source_digest.update(source.read_bytes())
    config["source_sha256"] = source_digest.hexdigest()
    config.update(dataset_size=len(dataset), source_indices=source_indices,
                  torch_version=str(torch.__version__), release_version="0.1.0")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    sample_csv = output_dir / "samples.csv"
    start = time.time()
    counts = {
        "seen": 0,
        "evaluated": 0,
        "clip_correct": 0,
        "ttp_correct": 0,
        "det_correct": 0,
        "det_adv": 0,
        "det_clean": 0,
        "adapted": 0,
        "clean_tpt_adapted": 0,
    }

    with sample_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["index", "target", "clip_pred", "ttp_pred", "sim", "detected_adv", "clip_correct", "ttp_correct"],
        )
        writer.writeheader()

        for loader_index, (images, target) in enumerate(loader):
            if args.loader_batch_size > 1:
                batch_images = images.cuda(args.gpu, non_blocking=True)
                batch_targets = target.cuda(args.gpu, non_blocking=True)
                batch_size = batch_images.shape[0]
                counts["seen"] += batch_size

                with torch.no_grad():
                    batch_clip_logits = model(batch_images)
                    batch_z = image_features(model, batch_images)
                    batch_z_pad = image_features(model, fixed_pad(batch_images))
                    batch_sims = cosine_sim(batch_z, batch_z_pad)

                previous_evaluated = counts["evaluated"]
                for row in range(batch_size):
                    position = loader_index * args.loader_batch_size + row
                    index = source_indices[position] if source_indices is not None else position
                    image = batch_images[row : row + 1]
                    target_row = batch_targets[row : row + 1]
                    clip_logits = batch_clip_logits[row : row + 1]
                    sim = float(batch_sims[row].item())
                    detected_adv = sim <= args.threshold

                    counts["det_adv" if detected_adv else "det_clean"] += 1
                    counts["det_correct"] += int(not detected_adv)

                    if detected_adv:
                        counts["adapted"] += 1
                        views = [view.cuda(args.gpu, non_blocking=True) for view in tensor_to_views(image, views_transform)]
                        ttp_logits = adapt_with_ttp(args, model, image, views, fixed_pad)
                    else:
                        ttp_logits = clip_logits

                    clip_pred = int(clip_logits.argmax(dim=-1).item())
                    ttp_pred = int(ttp_logits.argmax(dim=-1).item())
                    clip_correct = int(clip_pred == int(target_row.item()))
                    ttp_correct = int(ttp_pred == int(target_row.item()))
                    counts["evaluated"] += 1
                    counts["clip_correct"] += clip_correct
                    counts["ttp_correct"] += ttp_correct
                    writer.writerow(
                        {
                            "index": index,
                            "target": int(target_row.item()),
                            "clip_pred": clip_pred,
                            "ttp_pred": ttp_pred,
                            "sim": f"{sim:.6f}",
                            "detected_adv": int(detected_adv),
                            "clip_correct": clip_correct,
                            "ttp_correct": ttp_correct,
                        }
                    )

                if counts["evaluated"] // args.print_freq > previous_evaluated // args.print_freq:
                    fp.flush()
                    print(
                        f"[{counts['evaluated']}/{len(dataset)}] "
                        f"clip={100 * counts['clip_correct'] / counts['evaluated']:.2f} "
                        f"ttp={100 * counts['ttp_correct'] / counts['evaluated']:.2f} "
                        f"det={100 * counts['det_correct'] / counts['evaluated']:.2f} "
                        f"adapted={counts['adapted']}",
                        flush=True,
                    )
                continue

            index = source_indices[loader_index] if source_indices is not None else loader_index
            counts["seen"] += 1
            if loader_index % args.num_shards != args.shard_id:
                continue
            if args.max_samples is not None and counts["evaluated"] >= args.max_samples:
                break

            target = target.cuda(args.gpu, non_blocking=True)
            image, views = as_cuda_views(images, args.gpu)

            if attack is not None:
                image = make_or_load_adv(args, attack, model, image, target, index)
                views = [view.cuda(args.gpu, non_blocking=True) for view in tensor_to_views(image, views_transform)]

            with torch.no_grad():
                clip_logits = model(image)
                z = image_features(model, image)
                z_pad = image_features(model, fixed_pad(image))
                sim = cosine_sim(z, z_pad).item()

            detected_adv = sim <= args.threshold
            if detected_adv:
                counts["det_adv"] += 1
            else:
                counts["det_clean"] += 1
            if detected_adv == (args.eps > 0):
                counts["det_correct"] += 1

            if detected_adv:
                counts["adapted"] += 1
                if len(views) == 1:
                    views = [view.cuda(args.gpu, non_blocking=True) for view in tensor_to_views(image, views_transform)]
                ttp_logits = adapt_with_ttp(args, model, image, views, fixed_pad)
            elif args.clean_strategy == "tpt":
                counts["clean_tpt_adapted"] += 1
                if len(views) == 1:
                    views = [view.cuda(args.gpu, non_blocking=True) for view in tensor_to_views(image, views_transform)]
                assert clean_tpt_optimizer is not None and clean_tpt_optimizer_state is not None and clean_tpt_scaler is not None
                ttp_logits = adapt_clean_with_tpt(
                    args,
                    model,
                    image,
                    views,
                    clean_tpt_optimizer,
                    clean_tpt_optimizer_state,
                    clean_tpt_scaler,
                )
            else:
                ttp_logits = clip_logits

            clip_acc = accuracy(clip_logits, target, topk=(1,))[0].item()
            ttp_acc = accuracy(ttp_logits, target, topk=(1,))[0].item()
            clip_pred = int(clip_logits.argmax(dim=-1).item())
            ttp_pred = int(ttp_logits.argmax(dim=-1).item())

            counts["evaluated"] += 1
            counts["clip_correct"] += int(clip_acc > 0)
            counts["ttp_correct"] += int(ttp_acc > 0)
            writer.writerow(
                {
                    "index": index,
                    "target": int(target.item()),
                    "clip_pred": clip_pred,
                    "ttp_pred": ttp_pred,
                    "sim": f"{sim:.6f}",
                    "detected_adv": int(detected_adv),
                    "clip_correct": int(clip_acc > 0),
                    "ttp_correct": int(ttp_acc > 0),
                }
            )
            if counts["evaluated"] % args.print_freq == 0:
                fp.flush()

            if counts["evaluated"] % args.print_freq == 0:
                print(
                    f"[{counts['evaluated']}/{len(loader)}] "
                    f"clip={100 * counts['clip_correct'] / counts['evaluated']:.2f} "
                    f"ttp={100 * counts['ttp_correct'] / counts['evaluated']:.2f} "
                    f"det={100 * counts['det_correct'] / counts['evaluated']:.2f} "
                    f"adapted={counts['adapted']}",
                    flush=True,
                )

    evaluated = max(1, counts["evaluated"])
    result = {
        "ensemble_mode": args.ensemble_mode,
        "dataset": args.test_sets,
        "arch": args.arch,
        "eps": args.eps,
        "steps": args.steps,
        "max_samples": args.max_samples,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "evaluated": counts["evaluated"],
        "clip_acc": 100.0 * counts["clip_correct"] / evaluated,
        "ttp_acc": 100.0 * counts["ttp_correct"] / evaluated,
        "detection_acc": 100.0 * counts["det_correct"] / evaluated,
        "detected_adv": counts["det_adv"],
        "detected_clean": counts["det_clean"],
        "adapted": counts["adapted"],
        "clean_strategy": args.clean_strategy,
        "clean_tpt_adapted": counts["clean_tpt_adapted"],
        "sample_order": args.sample_order,
        "sample_seed": args.sample_seed,
        "loader_batch_size": args.loader_batch_size,
        "elapsed_sec": time.time() - start,
        "sample_csv": str(sample_csv),
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce TTP for CLIP adversarial robustness")
    parser.add_argument("data", help="dataset root following CoOp/R-TPT layout")
    parser.add_argument("--test_sets", default="DTD")
    parser.add_argument("--dataset_mode", default="test")
    parser.add_argument("-a", "--arch", default="ViT-B/32")
    parser.add_argument("--resolution", default=224, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="number of augmented views including the base view")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--n_ctx", default=4, type=int)
    parser.add_argument("--ctx_init", default="a_photo_of_a")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output_dir", default=str(ROOT / "output_results" / "ttp"))
    parser.add_argument("--eps", default=4.0, type=float)
    parser.add_argument("--alpha", default=None, type=float)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--cache-adv", action="store_true", default=True)
    parser.add_argument("--no-cache-adv", action="store_false", dest="cache_adv")
    parser.add_argument("--regen-adv", action="store_true")
    parser.add_argument("--adv-cache-dir", default=str(ROOT / "cache" / "adversarial"))
    parser.add_argument("--threshold", default=0.8, type=float)
    parser.add_argument("--padding-size", default=32, type=int)
    parser.add_argument("--padding-pattern", default="zero", choices=["zero", "white", "random"])
    parser.add_argument("--padding-lr", default=5.0, type=float)
    parser.add_argument("--padding-init-max", default=10.0, type=float, help="max initial padding pixel value in [0, 255] scale")
    parser.add_argument("--selection-p", default=0.1, type=float)
    parser.add_argument("--ensemble-mode", choices=["probability", "legacy_logits"], default="probability")
    parser.add_argument("--model-cache-dir", default=str(ROOT / "cache" / "clip"))
    parser.add_argument("--ensemble-temp", default=1.0, type=float)
    parser.add_argument("--clean-strategy", default="direct", choices=["direct", "tpt"])
    parser.add_argument("--clean-tpt-lr", default=5e-3, type=float)
    parser.add_argument("--clean-tpt-steps", default=1, type=int)
    parser.add_argument("--clean-tpt-selection-p", default=0.1, type=float)
    parser.add_argument("--shard-id", default=0, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--sample-order", default="first", choices=["first", "random"])
    parser.add_argument("--sample-seed", default=2026, type=int)
    parser.add_argument("--lazy-views", action="store_true", help="generate AugMix views only for samples that enter adaptation")
    parser.add_argument("--loader-batch-size", default=1, type=int, help="batched clean/direct evaluation only")
    parser.add_argument("-p", "--print-freq", default=50, type=int)
    args = parser.parse_args()
    if args.eps < 0 or args.padding_size <= 0 or args.ensemble_temp <= 0 or args.print_freq <= 0:
        parser.error("eps must be nonnegative; padding size, ensemble temperature and print frequency must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max-samples must be positive")
    if not 0 < args.selection_p <= 1 or not 0 < args.clean_tpt_selection_p <= 1:
        parser.error("selection fractions must be in (0, 1]")
    if args.resolution != 224:
        parser.error("This release uses 224px AugMix views; resolution must be 224")
    for key in ("data", "output_dir", "adv_cache_dir", "model_cache_dir"):
        setattr(args, key, str(Path(getattr(args, key)).expanduser().resolve()))
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.clean_tpt_steps < 1:
        raise ValueError("--clean-tpt-steps must be positive")
    if args.sample_order == "random" and args.num_shards != 1:
        raise ValueError("random subset evaluation currently requires --num-shards 1")
    if args.loader_batch_size < 1:
        raise ValueError("--loader-batch-size must be positive")
    if args.loader_batch_size > 1 and (args.eps > 0 or args.clean_strategy != "direct" or not args.lazy_views or args.num_shards != 1):
        raise ValueError("batched evaluation requires eps=0, clean-strategy=direct, lazy-views, and num-shards=1")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
