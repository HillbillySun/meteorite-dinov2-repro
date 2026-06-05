from __future__ import annotations

import argparse
import csv
import hashlib
import math
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from transformers import AutoModel

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)
AUGMENTED_ID_RE = re.compile(r"^(.+?)_(?:s|v)\d+$")


@dataclass(frozen=True)
class Example:
    image_id: str
    image_path: Path
    label: int | None
    image_hash: str
    group_key: str


class ImageDataset(Dataset):
    def __init__(self, examples: list[Example], transform):
        self.examples = examples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        with Image.open(ex.image_path) as image:
            image = image.convert("RGB")
        label = -1 if ex.label is None else int(ex.label)
        return self.transform(image), label, ex.image_id


class DinoHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, dropout: float = 0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen DINOv2 feature probe for stage2 meteorite classification.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/dinov2-with-registers-base")
    parser.add_argument("--loader", choices=["hf", "torchhub"], default="hf")
    parser.add_argument("--hub-arch", default="dinov2_vitb14_reg")
    parser.add_argument(
        "--feature-pool",
        choices=["cls_patch_mean", "cls_reg_mean", "cls_reg_patch_mean"],
        default="cls_patch_mean",
        help=(
            "How to pool backbone tokens into probe features: "
            "cls_patch_mean (default), cls_reg_mean, cls_reg_patch_mean."
        ),
    )
    parser.add_argument("--image-size", type=int, default=392)
    parser.add_argument("--resize-size", type=int, default=420)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=1.0,
        help="Cross-entropy weight for positive labels; 1.0 keeps the default unweighted loss.",
    )
    parser.add_argument(
        "--test-like-weight-groups-csv",
        type=Path,
        default=None,
        help="Optional test_like_split_groups.csv; selected test-like train groups get extra loss weight.",
    )
    parser.add_argument(
        "--test-like-weight-frac-per-label",
        type=float,
        default=0.0,
        help="Per-label fraction of most test-like groups to upweight; 0 disables.",
    )
    parser.add_argument(
        "--test-like-sample-weight",
        type=float,
        default=1.0,
        help="Loss multiplier for selected test-like real train samples; 1 disables.",
    )
    parser.add_argument(
        "--hard-example-weights-csv",
        type=Path,
        default=None,
        help="Optional CSV with id,weight columns for hard-example loss reweighting on real train samples.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-ids-csv", type=Path, default=None, help="Optional CSV with an id column defining a fixed validation split.")
    parser.add_argument("--early-stop-metric", choices=["balanced_accuracy", "topk_f1", "fixed_f1", "fixed_balanced_accuracy"], default="topk_f1")
    parser.add_argument("--val-topk", type=int, default=0, help="Validation top-k for topk_f1; 0 uses number of positive validation labels.")
    parser.add_argument("--target-pos-rate", type=float, default=0.0, help="If --val-topk is 0 and this is >0, use round(len(val)*rate) for validation top-k.")
    parser.add_argument("--fixed-threshold", type=float, default=0.5, help="Validation threshold used by fixed_f1/fixed_balanced_accuracy early stopping.")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--test-hflip-tta",
        action="store_true",
        help="Average test features from original and horizontal-flipped images.",
    )
    parser.add_argument(
        "--feature-mixup-alpha",
        type=float,
        default=0.0,
        help="Beta(alpha, alpha) mixup on extracted DINO features; 0 disables mixup.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay for probe weights during training; set within (0, 1) to enable.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="Gradient clipping max norm for probe training; 0 disables clipping.",
    )
    parser.add_argument(
        "--lr-warmup-ratio",
        type=float,
        default=0.0,
        help="Linear warmup ratio before cosine decay, e.g. 0.1; 0 disables warmup.",
    )
    parser.add_argument(
        "--pseudo-label-prob-csv",
        type=Path,
        default=None,
        help="Optional probability CSV for test-set pseudo labels.",
    )
    parser.add_argument(
        "--pseudo-label-top-pos",
        type=int,
        default=0,
        help="Select top-N highest-score test samples as pseudo positives.",
    )
    parser.add_argument(
        "--pseudo-label-bottom-neg",
        type=int,
        default=0,
        help="Select bottom-N lowest-score test samples as pseudo negatives.",
    )
    parser.add_argument(
        "--pseudo-label-weight",
        type=float,
        default=0.25,
        help="Training loss weight for pseudo-labeled samples.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md5_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def infer_group_key(image_id: str, image_hash: str) -> str:
    stem = Path(image_id).stem
    # Mixed style roots may prefix a second view of the same source image,
    # e.g. strict_000001_s00. Strip the source prefix before grouping so
    # relaxed/strict variants cannot leak across fixed validation splits.
    for prefix in ("strict_",):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    match = AUGMENTED_ID_RE.match(stem)
    if match:
        return f"aug:{match.group(1)}"
    return f"hash:{image_hash}"


def find_image_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(path)
    current = path
    while True:
        files = [entry for entry in current.iterdir() if entry.is_file()]
        if files:
            return current
        dirs = [entry for entry in current.iterdir() if entry.is_dir()]
        if len(dirs) != 1:
            raise FileNotFoundError(f"Could not find image files under {path}")
        current = dirs[0]


def load_train_examples(root: Path) -> list[Example]:
    image_dir = find_image_dir(root / "train_images")
    rows = read_csv(root / "train_labels.csv")
    examples = []
    seen_hashes = set()
    for row in rows:
        image_id = row["id"]
        image_path = image_dir / image_id
        image_hash = md5_file(image_path)
        if image_hash in seen_hashes:
            continue
        seen_hashes.add(image_hash)
        examples.append(
            Example(
                image_id=image_id,
                image_path=image_path,
                label=int(row["label"]),
                image_hash=image_hash,
                group_key=infer_group_key(image_id, image_hash),
            )
        )
    return examples


def load_test_examples(root: Path) -> list[Example]:
    image_dir = find_image_dir(root / "test_images")
    rows = read_csv(root / "sample_submission.csv")
    examples = []
    for row in rows:
        image_id = row["id"]
        image_path = image_dir / image_id
        image_hash = md5_file(image_path)
        examples.append(
            Example(
                image_id=image_id,
                image_path=image_path,
                label=None,
                image_hash=image_hash,
                group_key=infer_group_key(image_id, image_hash),
            )
        )
    return examples


def load_pseudo_examples(root: Path, prob_csv: Path, top_pos: int, bottom_neg: int) -> list[Example]:
    if top_pos <= 0 and bottom_neg <= 0:
        return []
    rows = read_csv(prob_csv)
    if not rows:
        raise RuntimeError(f"Pseudo label probability file is empty: {prob_csv}")
    if "id" not in rows[0] or "prob_meteorite" not in rows[0]:
        raise RuntimeError(f"Pseudo label probability file must contain columns id,prob_meteorite: {prob_csv}")

    scores: dict[str, float] = {}
    for row in rows:
        image_id = row["id"]
        if image_id in scores:
            raise RuntimeError(f"Duplicate id in pseudo label file: {image_id}")
        scores[image_id] = float(row["prob_meteorite"])
    ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)

    n = len(ranked)
    top_pos = max(0, min(top_pos, n))
    bottom_neg = max(0, min(bottom_neg, n - top_pos))
    pos_ids = {image_id for image_id, _ in ranked[:top_pos]}
    neg_ids: list[str] = []
    for image_id, _ in reversed(ranked):
        if image_id in pos_ids:
            continue
        neg_ids.append(image_id)
        if len(neg_ids) >= bottom_neg:
            break

    image_dir = find_image_dir(root / "test_images")
    pseudo_examples: list[Example] = []
    for image_id in sorted(pos_ids):
        image_path = image_dir / image_id
        image_hash = md5_file(image_path)
        pseudo_examples.append(
            Example(
                image_id=image_id,
                image_path=image_path,
                label=1,
                image_hash=image_hash,
                group_key=f"pseudo:{image_id}",
            )
        )
    for image_id in sorted(neg_ids):
        image_path = image_dir / image_id
        image_hash = md5_file(image_path)
        pseudo_examples.append(
            Example(
                image_id=image_id,
                image_path=image_path,
                label=0,
                image_hash=image_hash,
                group_key=f"pseudo:{image_id}",
            )
        )
    return pseudo_examples


def stratified_group_split(examples: list[Example], val_ratio: float, seed: int) -> tuple[list[Example], list[Example]]:
    grouped = defaultdict(list)
    for ex in examples:
        grouped[ex.group_key].append(ex)
    by_label = defaultdict(list)
    for group in grouped.values():
        by_label[group[0].label].append(group)

    rng = random.Random(seed)
    train, val = [], []
    for groups in by_label.values():
        groups = list(groups)
        rng.shuffle(groups)
        val_count = max(1, round(len(groups) * val_ratio))
        val_keys = {group[0].group_key for group in groups[:val_count]}
        for group in groups:
            (val if group[0].group_key in val_keys else train).extend(group)
    return train, val




def fixed_id_split(examples: list[Example], val_ids_csv: Path) -> tuple[list[Example], list[Example]]:
    rows = read_csv(val_ids_csv)
    if not rows or "id" not in rows[0]:
        raise RuntimeError(f"Validation id CSV must contain an id column: {val_ids_csv}")
    val_ids = {row["id"] for row in rows}
    id_to_group = {ex.image_id: ex.group_key for ex in examples}
    missing = sorted(image_id for image_id in val_ids if image_id not in id_to_group)
    if missing:
        raise RuntimeError(f"Validation id CSV contains ids not found in train examples: {missing[:5]}")
    val_groups = {id_to_group[image_id] for image_id in val_ids}
    train, val = [], []
    for ex in examples:
        (val if ex.group_key in val_groups else train).append(ex)
    if not train or not val:
        raise RuntimeError(f"Invalid fixed split from {val_ids_csv}: train={len(train)} val={len(val)}")
    return train, val

def select_test_like_weight_groups(groups_csv: Path, frac_per_label: float) -> set[str]:
    if frac_per_label <= 0:
        return set()
    if frac_per_label > 1:
        raise ValueError("--test-like-weight-frac-per-label must be <= 1")
    rows = read_csv(groups_csv)
    required = {"group_key", "label", "test_like_score"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"{groups_csv} must contain columns {sorted(required)}")
    by_label: defaultdict[int, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append((row["group_key"], float(row["test_like_score"])))
    selected: set[str] = set()
    for label, items in sorted(by_label.items()):
        items.sort(key=lambda item: (item[1], item[0]), reverse=True)
        take = min(len(items), max(1, round(len(items) * frac_per_label)))
        selected.update(group_key for group_key, _ in items[:take])
        print(f"[dinov2] test_like_weight label={label} groups={len(items)} selected={take} cutoff={items[take - 1][1]:.6f}")
    return selected


def load_hard_example_weights(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    if not rows or "id" not in rows[0] or "weight" not in rows[0]:
        raise RuntimeError(f"{path} must contain id and weight columns")
    weights: dict[str, float] = {}
    for row in rows:
        weight = float(row["weight"])
        if weight <= 0:
            raise ValueError(f"Hard-example weight must be positive: {row}")
        weights[row["id"]] = weight
    return weights


def build_transform(image_size: int, resize_size: int):
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
        ]
    )


def load_dino_model(args: argparse.Namespace) -> nn.Module:
    if args.loader == "torchhub":
        return torch.hub.load("facebookresearch/dinov2", args.hub_arch)
    return AutoModel.from_pretrained(args.model_name)


def dino_forward(model: nn.Module, images: torch.Tensor, feature_pool: str = "cls_patch_mean") -> torch.Tensor:
    if hasattr(model, "forward_features"):
        feats = model.forward_features(images)
        if isinstance(feats, dict):
            cls = feats.get("x_norm_clstoken")
            regs = feats.get("x_norm_regtokens")
            patches = feats.get("x_norm_patchtokens")
            if cls is not None:
                if feature_pool == "cls_reg_patch_mean":
                    parts = [cls]
                    if regs is not None and regs.ndim == 3 and regs.shape[1] > 0:
                        parts.append(regs.mean(dim=1))
                    if patches is not None and patches.ndim == 3 and patches.shape[1] > 0:
                        parts.append(patches.mean(dim=1))
                    if len(parts) > 1:
                        return torch.cat(parts, dim=1)
                elif feature_pool == "cls_reg_mean":
                    if regs is not None and regs.ndim == 3 and regs.shape[1] > 0:
                        return torch.cat([cls, regs.mean(dim=1)], dim=1)
                if patches is not None and patches.ndim == 3 and patches.shape[1] > 0:
                    return torch.cat([cls, patches.mean(dim=1)], dim=1)
            if cls is not None:
                return cls
        if torch.is_tensor(feats):
            return feats

    try:
        outputs = model(pixel_values=images, interpolate_pos_encoding=True)
    except TypeError:
        outputs = model(pixel_values=images)
    hidden = outputs.last_hidden_state
    cls = hidden[:, 0]
    n_registers = int(getattr(model.config, "num_register_tokens", 0) or 0)
    regs = hidden[:, 1 : 1 + n_registers] if n_registers > 0 else None
    patches = hidden[:, 1 + n_registers :]

    if feature_pool == "cls_reg_patch_mean":
        parts = [cls]
        if regs is not None and regs.shape[1] > 0:
            parts.append(regs.mean(dim=1))
        if patches.shape[1] > 0:
            parts.append(patches.mean(dim=1))
        if len(parts) > 1:
            return torch.cat(parts, dim=1)

    if feature_pool == "cls_reg_mean" and regs is not None and regs.shape[1] > 0:
        return torch.cat([cls, regs.mean(dim=1)], dim=1)

    if patches.shape[1] > 0:
        return torch.cat([cls, patches.mean(dim=1)], dim=1)
    return cls


@torch.no_grad()
def extract_features(
    model: nn.Module,
    examples: list[Example],
    args: argparse.Namespace,
    device: torch.device,
    test_hflip_tta: bool = False,
):
    dataset = ImageDataset(examples, transform=build_transform(args.image_size, args.resize_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model.eval()
    features, labels, ids = [], [], []
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    for images, batch_labels, batch_ids in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            feats = dino_forward(model, images, feature_pool=args.feature_pool)
            if test_hflip_tta:
                feats_flip = dino_forward(model, torch.flip(images, dims=[3]), feature_pool=args.feature_pool)
                feats = 0.5 * (feats + feats_flip)
        features.append(feats.float().cpu())
        labels.append(batch_labels.long().cpu())
        ids.extend(batch_ids)
    return torch.cat(features), torch.cat(labels), ids


def binary_metrics(labels: torch.Tensor, probs: torch.Tensor, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).long()
    labels = labels.long()
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    accuracy = (tp + tn) / max(1, len(labels))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def metrics_at_topk(labels: torch.Tensor, probs: torch.Tensor, top_k: int) -> dict[str, float]:
    labels = labels.long()
    top_k = max(1, min(int(top_k), len(labels)))
    preds = torch.zeros_like(labels)
    preds[torch.argsort(probs, descending=True)[:top_k]] = 1
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    accuracy = (tp + tn) / max(1, len(labels))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "top_k": top_k,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def resolve_val_topk(labels: torch.Tensor, args: argparse.Namespace) -> int:
    if int(getattr(args, "val_topk", 0)) > 0:
        return int(args.val_topk)
    target_pos_rate = float(getattr(args, "target_pos_rate", 0.0))
    if target_pos_rate > 0:
        return max(1, round(len(labels) * target_pos_rate))
    return max(1, int(labels.long().sum().item()))


def find_best_threshold(labels: torch.Tensor, probs: torch.Tensor) -> tuple[float, dict[str, float]]:
    best_t = 0.5
    best = binary_metrics(labels, probs, best_t)
    best_score = best["balanced_accuracy"]
    for step in range(5, 96):
        threshold = step / 100
        metrics = binary_metrics(labels, probs, threshold)
        if metrics["balanced_accuracy"] > best_score:
            best_t, best, best_score = threshold, metrics, metrics["balanced_accuracy"]
    return best_t, best


def train_head(train_x, train_y, train_w, val_x, val_y, args: argparse.Namespace, device: torch.device):
    train_ds = TensorDataset(train_x, train_y, train_w)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    head = DinoHead(train_x.shape[1], dropout=args.dropout).to(device)
    positive_class_weight = float(getattr(args, "positive_class_weight", 1.0))
    if positive_class_weight <= 0:
        raise ValueError("--positive-class-weight must be > 0")
    class_weights = torch.tensor([1.0, positive_class_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
        reduction="none",
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * max(0.0, float(args.lr_warmup_ratio)))

    def lr_lambda(step: int) -> float:
        if total_steps <= 1:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    mixup_alpha = max(0.0, float(args.feature_mixup_alpha))
    use_ema = 0.0 < float(args.ema_decay) < 1.0
    ema_decay = float(args.ema_decay)
    grad_clip = max(0.0, float(args.grad_clip_norm))
    ema_state = {k: v.detach().clone() for k, v in head.state_dict().items()} if use_ema else None

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x_gpu = val_x.to(device)
    val_y_gpu = val_y.to(device)
    val_topk = resolve_val_topk(val_y, args)
    fixed_threshold = float(getattr(args, "fixed_threshold", 0.5))
    if not 0.0 <= fixed_threshold <= 1.0:
        raise ValueError("--fixed-threshold must be in [0, 1]")
    print(
        f"[dinov2] early_stop_metric={args.early_stop_metric} val_topk={val_topk} "
        f"fixed_threshold={fixed_threshold:.6f} val_pos={int(val_y.sum().item())} val_total={len(val_y)}"
    )
    best = {"score": -1.0, "epoch": 0, "threshold": 0.5, "metrics": None, "topk_metrics": None, "fixed_metrics": None, "state": None}
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        head.train()
        total_loss = 0.0
        total = 0
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad(set_to_none=True)

            if mixup_alpha > 0.0:
                lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample(()).item())
                perm = torch.randperm(len(batch_x), device=batch_x.device)
                mixed_x = lam * batch_x + (1.0 - lam) * batch_x[perm]
                logits = head(mixed_x)
                mixed_w = lam * batch_w + (1.0 - lam) * batch_w[perm]
                per_sample_loss = lam * criterion(logits, batch_y) + (1.0 - lam) * criterion(logits, batch_y[perm])
                loss = (per_sample_loss * mixed_w).sum() / mixed_w.sum().clamp_min(1e-12)
            else:
                logits = head(batch_x)
                per_sample_loss = criterion(logits, batch_y)
                loss = (per_sample_loss * batch_w).sum() / batch_w.sum().clamp_min(1e-12)

            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()

            if use_ema:
                with torch.no_grad():
                    for name, param in head.state_dict().items():
                        ema_state[name].mul_(ema_decay).add_(param.detach(), alpha=1.0 - ema_decay)

            scheduler.step()
            total_loss += float(loss.item()) * len(batch_x)
            total += len(batch_x)

        eval_from_ema = False
        raw_state_backup = None
        if use_ema:
            raw_state_backup = {k: v.detach().clone() for k, v in head.state_dict().items()}
            head.load_state_dict(ema_state, strict=True)
            eval_from_ema = True

        head.eval()
        with torch.no_grad():
            val_logits = head(val_x_gpu)
            val_loss = float(criterion(val_logits, val_y_gpu).mean().item())
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu()

        if eval_from_ema:
            head.load_state_dict(raw_state_backup, strict=True)

        threshold, metrics = find_best_threshold(val_y, val_probs)
        topk_metrics = metrics_at_topk(val_y, val_probs, val_topk)
        fixed_metrics = binary_metrics(val_y, val_probs, fixed_threshold)
        if args.early_stop_metric == "topk_f1":
            score = topk_metrics["f1"]
        elif args.early_stop_metric == "fixed_f1":
            score = fixed_metrics["f1"]
        elif args.early_stop_metric == "fixed_balanced_accuracy":
            score = fixed_metrics["balanced_accuracy"]
        else:
            score = metrics["balanced_accuracy"]
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, total),
            "val_loss": val_loss,
            "threshold_best": threshold,
            "metrics_best": metrics,
            "topk_metrics": topk_metrics,
            "fixed_threshold": fixed_threshold,
            "fixed_metrics": fixed_metrics,
            "early_stop_metric": args.early_stop_metric,
            "early_stop_score": score,
            "lr": optimizer.param_groups[0]["lr"],
            "eval_with_ema": eval_from_ema,
            "seconds": time.time() - start,
        })
        print(
            f"Epoch {epoch:02d} | train_loss={history[-1]['train_loss']:.4f} | "
            f"val_loss={val_loss:.4f} | bal_acc@best={metrics['balanced_accuracy']:.4f} | "
            f"topk_f1={topk_metrics['f1']:.4f} topk_recall={topk_metrics['recall']:.4f} "
            f"topk_precision={topk_metrics['precision']:.4f} | fixed_f1={fixed_metrics['f1']:.4f} "
            f"fixed_fp={fixed_metrics['fp']} fixed_fn={fixed_metrics['fn']} | threshold={threshold:.2f} | lr={history[-1]['lr']:.6f}"
        )
        if score > best["score"]:
            best_state = ema_state if eval_from_ema else head.state_dict()
            best.update({
                "score": score,
                "epoch": epoch,
                "threshold": threshold,
                "metrics": metrics,
                "topk_metrics": topk_metrics,
                "fixed_threshold": fixed_threshold,
                "fixed_metrics": fixed_metrics,
                "state": {k: v.detach().cpu().clone() for k, v in best_state.items()},
            })
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    head.load_state_dict(best["state"])
    return head, best, history


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dinov2] device={device} model={args.model_name} image_size={args.image_size}")
    print(
        f"[dinov2] tricks mixup_alpha={args.feature_mixup_alpha} ema_decay={args.ema_decay} "
        f"warmup_ratio={args.lr_warmup_ratio} grad_clip={args.grad_clip_norm} hflip_tta={args.test_hflip_tta} "
        f"feature_pool={args.feature_pool} positive_class_weight={args.positive_class_weight}"
    )
    print(
        f"[dinov2] pseudo prob_csv={args.pseudo_label_prob_csv} top_pos={args.pseudo_label_top_pos} "
        f"bottom_neg={args.pseudo_label_bottom_neg} weight={args.pseudo_label_weight}"
    )
    print(
        f"[dinov2] test_like_weight groups_csv={args.test_like_weight_groups_csv} "
        f"frac={args.test_like_weight_frac_per_label} sample_weight={args.test_like_sample_weight}"
    )
    if args.pseudo_label_weight <= 0:
        raise ValueError("--pseudo-label-weight must be > 0")
    if args.test_like_sample_weight <= 0:
        raise ValueError("--test-like-sample-weight must be > 0")

    train_examples_all = load_train_examples(args.data_root)
    if args.val_ids_csv is not None:
        train_examples, val_examples = fixed_id_split(train_examples_all, args.val_ids_csv)
        print(f"[dinov2] using fixed validation ids: {args.val_ids_csv}")
    else:
        train_examples, val_examples = stratified_group_split(train_examples_all, args.val_ratio, args.seed)
    real_train_examples = list(train_examples)
    test_like_weight_group_keys: set[str] = set()
    if args.test_like_weight_groups_csv is not None and args.test_like_weight_frac_per_label > 0 and args.test_like_sample_weight > 1.0:
        test_like_weight_group_keys = select_test_like_weight_groups(
            args.test_like_weight_groups_csv,
            args.test_like_weight_frac_per_label,
        )
    hard_example_weights = load_hard_example_weights(args.hard_example_weights_csv) if args.hard_example_weights_csv else {}
    real_train_weights = []
    hard_weighted_real = 0
    for ex in real_train_examples:
        weight = float(args.test_like_sample_weight) if ex.group_key in test_like_weight_group_keys else 1.0
        hard_weight = hard_example_weights.get(ex.image_id)
        if hard_weight is None:
            hard_weight = hard_example_weights.get(ex.group_key)
        if hard_weight is not None:
            weight *= float(hard_weight)
            hard_weighted_real += 1
        real_train_weights.append(weight)
    test_like_weighted_real = sum(1 for weight in real_train_weights if weight > 1.0)
    if test_like_weight_group_keys:
        print(
            f"[dinov2] test_like_weight selected_groups={len(test_like_weight_group_keys)} "
            f"weighted_real_examples={test_like_weighted_real}/{len(real_train_examples)}"
        )
    if hard_example_weights:
        print(
            f"[dinov2] hard_example_weight csv={args.hard_example_weights_csv} "
            f"weighted_real_examples={hard_weighted_real}/{len(real_train_examples)}"
        )
    pseudo_examples: list[Example] = []
    if args.pseudo_label_prob_csv is not None:
        pseudo_examples = load_pseudo_examples(
            args.data_root,
            args.pseudo_label_prob_csv,
            args.pseudo_label_top_pos,
            args.pseudo_label_bottom_neg,
        )
        real_hashes = {ex.image_hash for ex in train_examples_all}
        kept_pseudo: list[Example] = []
        skipped_overlap = 0
        for ex in pseudo_examples:
            if ex.image_hash in real_hashes:
                skipped_overlap += 1
                continue
            kept_pseudo.append(ex)
        pseudo_examples = kept_pseudo
        train_examples.extend(pseudo_examples)
        print(f"[dinov2] pseudo selected={len(kept_pseudo)} skipped_overlap={skipped_overlap}")

    train_weights = real_train_weights + [float(args.pseudo_label_weight)] * len(pseudo_examples)
    if len(train_weights) != len(train_examples):
        raise RuntimeError("train_weights size mismatch")
    print(
        f"[dinov2] train_real={len(real_train_examples)} train_pseudo={len(pseudo_examples)} "
        f"train_total={len(train_examples)} val={len(val_examples)} total_real={len(train_examples_all)}"
    )

    model = load_dino_model(args).to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    print("[dinov2] extract train features")
    train_x, train_y, _ = extract_features(model, train_examples, args, device)
    print("[dinov2] extract val features")
    val_x, val_y, _ = extract_features(model, val_examples, args, device)
    train_y = train_y.clamp_min(0)
    train_w = torch.tensor(train_weights, dtype=torch.float32)
    val_y = val_y.clamp_min(0)

    head, best, history = train_head(train_x, train_y, train_w, val_x, val_y, args, device)
    torch.save(
        {
            "model_name": args.model_name,
            "loader": args.loader,
            "hub_arch": args.hub_arch,
            "image_size": args.image_size,
            "resize_size": args.resize_size,
            "feature_pool": str(args.feature_pool),
            "feature_dim": train_x.shape[1],
            "head_state": best["state"],
            "threshold": best["threshold"],
            "epoch": best["epoch"],
            "feature_mixup_alpha": float(args.feature_mixup_alpha),
            "positive_class_weight": float(args.positive_class_weight),
            "ema_decay": float(args.ema_decay),
            "grad_clip_norm": float(args.grad_clip_norm),
            "lr_warmup_ratio": float(args.lr_warmup_ratio),
            "test_hflip_tta": bool(args.test_hflip_tta),
            "pseudo_label_prob_csv": str(args.pseudo_label_prob_csv) if args.pseudo_label_prob_csv else None,
            "pseudo_label_top_pos": int(args.pseudo_label_top_pos),
            "pseudo_label_bottom_neg": int(args.pseudo_label_bottom_neg),
            "pseudo_label_weight": float(args.pseudo_label_weight),
            "pseudo_selected": len(pseudo_examples),
            "test_like_weight_groups_csv": str(args.test_like_weight_groups_csv) if args.test_like_weight_groups_csv else None,
            "test_like_weight_frac_per_label": float(args.test_like_weight_frac_per_label),
            "test_like_sample_weight": float(args.test_like_sample_weight),
            "test_like_weighted_real_examples": int(test_like_weighted_real),
            "hard_example_weights_csv": str(args.hard_example_weights_csv) if args.hard_example_weights_csv else None,
            "hard_weighted_real_examples": int(hard_weighted_real),
            "val_ids_csv": str(args.val_ids_csv) if args.val_ids_csv else None,
            "early_stop_metric": str(args.early_stop_metric),
            "val_topk": int(resolve_val_topk(val_y, args)),
            "target_pos_rate": float(args.target_pos_rate),
            "fixed_threshold": float(args.fixed_threshold),
        },
        args.output_dir / "best_probe.pt",
    )

    print("[dinov2] extract test features")
    test_examples = load_test_examples(args.data_root)
    test_x, _, test_ids = extract_features(
        model,
        test_examples,
        args,
        device,
        test_hflip_tta=args.test_hflip_tta,
    )
    head.eval()
    with torch.no_grad():
        probs = torch.softmax(head(test_x.to(device)), dim=1)[:, 1].cpu().tolist()

    write_csv(
        args.output_dir / "test_probabilities.csv",
        [{"id": image_id, "prob_meteorite": f"{prob:.8f}"} for image_id, prob in zip(test_ids, probs)],
        ["id", "prob_meteorite"],
    )
    threshold = float(best["threshold"])
    write_csv(
        args.output_dir / "submission.csv",
        [{"id": image_id, "label": int(prob >= threshold)} for image_id, prob in zip(test_ids, probs)],
        ["id", "label"],
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": args.model_name,
                "loader": args.loader,
                "hub_arch": args.hub_arch,
                "image_size": args.image_size,
                "resize_size": args.resize_size,
                "feature_pool": str(args.feature_pool),
                "best_epoch": best["epoch"],
                "best_threshold": best["threshold"],
                "best_metrics": best["metrics"],
                "best_topk_metrics": best.get("topk_metrics"),
                "best_fixed_metrics": best.get("fixed_metrics"),
                "fixed_threshold": float(args.fixed_threshold),
                "best_score": best["score"],
                "early_stop_metric": str(args.early_stop_metric),
                "val_topk": int(resolve_val_topk(val_y, args)),
                "target_pos_rate": float(args.target_pos_rate),
                "test_hflip_tta": bool(args.test_hflip_tta),
                "feature_mixup_alpha": float(args.feature_mixup_alpha),
                "positive_class_weight": float(args.positive_class_weight),
                "ema_decay": float(args.ema_decay),
                "grad_clip_norm": float(args.grad_clip_norm),
                "lr_warmup_ratio": float(args.lr_warmup_ratio),
                "pseudo_label_prob_csv": str(args.pseudo_label_prob_csv) if args.pseudo_label_prob_csv else None,
                "pseudo_label_top_pos": int(args.pseudo_label_top_pos),
                "pseudo_label_bottom_neg": int(args.pseudo_label_bottom_neg),
                "pseudo_label_weight": float(args.pseudo_label_weight),
                "pseudo_selected": len(pseudo_examples),
                "history": history,
                "train_examples": len(train_examples),
                "train_real_examples": len(real_train_examples),
                "train_pseudo_examples": len(pseudo_examples),
                "val_examples": len(val_examples),
                "hard_example_weights_csv": str(args.hard_example_weights_csv) if args.hard_example_weights_csv else None,
                "hard_weighted_real_examples": int(hard_weighted_real),
                "val_ids_csv": str(args.val_ids_csv) if args.val_ids_csv else None,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"[dinov2] saved probabilities to {args.output_dir / 'test_probabilities.csv'}")


if __name__ == "__main__":
    main()
