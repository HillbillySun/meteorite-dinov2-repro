#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from train_dinov2_probe import metrics_at_topk, set_seed, train_head, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the known-best DINOv2 probe head from cached features and save checkpoint.")
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=90)
    parser.add_argument("--reference-csv", type=Path, default=Path("kaggle_online_results.csv"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--feature-mixup-alpha", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--lr-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-metric", default="topk_f1", choices=["topk_f1", "balanced_accuracy", "fixed_f1", "fixed_balanced_accuracy"])
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    return parser.parse_args()


def topk_submission_rows(ids: list[str], scores: torch.Tensor, top_k: int) -> list[dict[str, object]]:
    selected = set(torch.argsort(scores, descending=True)[:top_k].tolist())
    return [{"id": image_id, "label": int(index in selected)} for index, image_id in enumerate(ids)]


def write_probabilities(path: Path, ids: list[str], scores: torch.Tensor) -> None:
    write_csv(
        path,
        [{"id": image_id, "prob_meteorite": f"{float(score):.8f}"} for image_id, score in zip(ids, scores)],
        ["id", "prob_meteorite"],
    )


def read_labels(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    label_col = [c for c in rows[0].keys() if c != "id"][0]
    return {r["id"]: str(r[label_col]) for r in rows}


def main() -> None:
    args = parse_args()
    if not args.feature_cache.exists():
        raise FileNotFoundError(f"Missing feature cache: {args.feature_cache}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.feature_cache, map_location="cpu")

    train_x = data["train_x"]
    train_y = data["train_y"].long()
    val_x = data["val_x"]
    val_y = data["val_y"].long()
    test_x = data["test_x"]
    test_ids = list(data["test_ids"])
    train_w = torch.ones(len(train_y), dtype=torch.float32)
    val_topk = max(1, int(val_y.long().sum().item()))

    head_args = SimpleNamespace(
        dropout=args.dropout,
        positive_class_weight=args.positive_class_weight,
        label_smoothing=args.label_smoothing,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        feature_mixup_alpha=args.feature_mixup_alpha,
        ema_decay=args.ema_decay,
        grad_clip_norm=args.grad_clip_norm,
        lr_warmup_ratio=args.lr_warmup_ratio,
        early_stop_metric=args.early_stop_metric,
        val_topk=val_topk,
        target_pos_rate=0.0,
        fixed_threshold=args.fixed_threshold,
    )

    print(f"[best_ckpt] device={device}")
    print(f"[best_ckpt] feature_cache={args.feature_cache}")
    print(f"[best_ckpt] train={len(train_y)} val={len(val_y)} test={len(test_ids)} val_topk={val_topk}")
    print(
        "[best_ckpt] hparams="
        f"seed={args.seed} pw={args.positive_class_weight} mixup={args.feature_mixup_alpha} "
        f"dropout={args.dropout} ls={args.label_smoothing} lr={args.lr} wd={args.weight_decay} "
        f"ema={args.ema_decay} warmup={args.lr_warmup_ratio} grad_clip={args.grad_clip_norm}"
    )

    head, best, history = train_head(train_x, train_y, train_w, val_x, val_y, head_args, device)
    head.eval()
    with torch.no_grad():
        val_probs = torch.softmax(head(val_x.to(device)), dim=1)[:, 1].float().cpu()
        test_probs = torch.softmax(head(test_x.to(device)), dim=1)[:, 1].float().cpu()

    val_topk_metrics = metrics_at_topk(val_y, val_probs, val_topk)
    tag = "s3407_pw1p00_mx0p05_do0p45_ls0p03_lr0p0005_wd0p01"
    record = {
        "tag": tag,
        "seed": args.seed,
        "positive_class_weight": args.positive_class_weight,
        "feature_mixup_alpha": args.feature_mixup_alpha,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "lr_warmup_ratio": args.lr_warmup_ratio,
        "grad_clip_norm": args.grad_clip_norm,
        "best_epoch": int(best["epoch"]),
        "best_score": float(best["score"]),
        "best_threshold": float(best["threshold"]),
        "best_balanced_accuracy": float(best["metrics"]["balanced_accuracy"]),
        "val_topk_f1": float(val_topk_metrics["f1"]),
        "val_topk_precision": float(val_topk_metrics["precision"]),
        "val_topk_recall": float(val_topk_metrics["recall"]),
        "val_topk_fp": int(val_topk_metrics["fp"]),
        "val_topk_fn": int(val_topk_metrics["fn"]),
    }

    checkpoint = {
        "checkpoint_type": "dinov2_frozen_feature_mlp_probe_head",
        "backbone": {
            "loader": data.get("loader", "torchhub"),
            "hub_arch": data.get("hub_arch", "dinov2_vitb14_reg"),
            "model_name": data.get("model_name", "facebook/dinov2-with-registers-base"),
            "image_size": data.get("image_size", 392),
            "resize_size": data.get("resize_size", 420),
            "feature_pool": data.get("feature_pool", "cls_patch_mean"),
        },
        "feature_dim": int(train_x.shape[1]),
        "head_state": best["state"],
        "record": record,
        "best": {k: v for k, v in best.items() if k != "state"},
    }
    torch.save(checkpoint, args.output_dir / "best_probe.pt")
    write_probabilities(args.output_dir / "test_probabilities.csv", test_ids, test_probs)
    write_csv(args.output_dir / f"topk_{args.top_k}.csv", topk_submission_rows(test_ids, test_probs, args.top_k), ["id", "label"])

    summary = {
        "record": record,
        "best": {k: v for k, v in best.items() if k != "state"},
        "history": history,
        "feature_cache": str(args.feature_cache),
        "checkpoint": str(args.output_dir / "best_probe.pt"),
        "probabilities": str(args.output_dir / "test_probabilities.csv"),
        "topk_submission": str(args.output_dir / f"topk_{args.top_k}.csv"),
    }
    pred_path = args.output_dir / f"topk_{args.top_k}.csv"
    if args.reference_csv.exists():
        pred = read_labels(pred_path)
        ref = read_labels(args.reference_csv)
        diff = [(k, pred.get(k), ref.get(k)) for k in sorted(set(pred) | set(ref)) if pred.get(k) != ref.get(k)]
        summary["reference"] = str(args.reference_csv)
        summary["matches_reference"] = len(diff) == 0
        summary["diff_count"] = len(diff)
        summary["diff_preview"] = diff[:20]
        if diff:
            print(f"[best_ckpt] WARNING: {pred_path} differs from {args.reference_csv}; diff_count={len(diff)}")
            print("[best_ckpt] diff_preview=", diff[:20])
        else:
            print(f"[best_ckpt] OK: {pred_path} matches {args.reference_csv}")

    with (args.output_dir / "checkpoint_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"[best_ckpt] checkpoint={args.output_dir / 'best_probe.pt'}")
    print(f"[best_ckpt] probabilities={args.output_dir / 'test_probabilities.csv'}")
    print(f"[best_ckpt] topk_submission={pred_path}")


if __name__ == "__main__":
    main()
