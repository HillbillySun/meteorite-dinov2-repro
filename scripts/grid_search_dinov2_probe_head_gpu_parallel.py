#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import torch

from train_dinov2_probe import (
    extract_features,
    fixed_id_split,
    load_dino_model,
    load_test_examples,
    load_train_examples,
    metrics_at_topk,
    set_seed,
    train_head,
    write_csv,
)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split() if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split() if x.strip()]


def rank01(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ranks[order] = torch.linspace(1.0, 0.0, len(scores))
    return ranks


def topk_submission_rows(ids: list[str], scores: torch.Tensor, top_k: int) -> list[dict[str, object]]:
    selected = set(torch.argsort(scores, descending=True)[:top_k].tolist())
    return [{"id": image_id, "label": int(index in selected)} for index, image_id in enumerate(ids)]


def write_probabilities(path: Path, ids: list[str], scores: torch.Tensor) -> None:
    write_csv(
        path,
        [{"id": image_id, "prob_meteorite": f"{float(score):.8f}"} for image_id, score in zip(ids, scores)],
        ["id", "prob_meteorite"],
    )


def make_tag(seed: int, pos_w: float, mixup: float, dropout: float, smoothing: float, lr: float, wd: float) -> str:
    return f"s{seed}_pw{pos_w:.2f}_mx{mixup:.2f}_do{dropout:.2f}_ls{smoothing:.2f}_lr{lr:g}_wd{wd:g}".replace('.', 'p')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU-parallel grid search for DINOv2 frozen-feature probe heads.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--val-ids-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, default=None)
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--top-k", type=int, default=86)
    p.add_argument("--target-pos-rate", type=float, default=0.4432989691)
    p.add_argument("--model-name", default="facebook/dinov2-with-registers-base")
    p.add_argument("--loader", choices=["hf", "torchhub"], default="torchhub")
    p.add_argument("--hub-arch", default="dinov2_vitb14_reg")
    p.add_argument("--feature-pool", choices=["cls_patch_mean", "cls_reg_mean", "cls_reg_patch_mean"], default="cls_patch_mean")
    p.add_argument("--image-size", type=int, default=392)
    p.add_argument("--resize-size", type=int, default=420)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--test-hflip-tta", action="store_true")
    p.add_argument("--head-seeds", default="3407")
    p.add_argument("--positive-class-weights", default="1.00 1.05 1.10")
    p.add_argument("--feature-mixup-alphas", default="0.03 0.05 0.07")
    p.add_argument("--dropouts", default="0.40 0.45 0.50")
    p.add_argument("--label-smoothings", default="0.02 0.03 0.04")
    p.add_argument("--lrs", default="0.0004 0.0005 0.0007")
    p.add_argument("--weight-decays", default="0.01")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--lr-warmup-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--early-stop-metric", choices=["topk_f1", "balanced_accuracy", "fixed_f1", "fixed_balanced_accuracy"], default="topk_f1")
    p.add_argument("--fixed-threshold", type=float, default=0.5)
    p.add_argument("--hard-example-weights-csv", type=Path, default=None, help="Optional CSV with id,weight or group_key,weight columns for training sample weights.")
    p.add_argument("--max-runs", type=int, default=0)
    p.add_argument("--save-top-n", type=int, default=12)
    p.add_argument("--parallel-workers", type=int, default=4)
    p.add_argument("--worker-threads", type=int, default=1)
    return p.parse_args()


def build_or_load_feature_cache(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.feature_cache or (args.output_dir / "features_cache.pt")
    if cache.exists() and not args.overwrite_cache:
        try:
            meta = torch.load(cache, map_location="cpu")
            same_data = str(meta.get("data_root", "")) == str(args.data_root)
            same_val = str(meta.get("val_ids_csv", "")) == str(args.val_ids_csv)
            same_image = int(meta.get("image_size", -1)) == int(args.image_size)
            same_resize = int(meta.get("resize_size", -1)) == int(args.resize_size)
            same_pool = str(meta.get("feature_pool", "")) == str(args.feature_pool)
            same_loader = str(meta.get("loader", "")) == str(args.loader)
            same_model = str(meta.get("model_name", "")) == str(args.model_name)
            same_arch = str(meta.get("hub_arch", "")) == str(args.hub_arch)
            if same_data and same_val and same_image and same_resize and same_pool and same_loader and same_model and same_arch:
                print(f"[gpu_grid] using existing feature cache: {cache}")
                return cache
            print(f"[gpu_grid] cache metadata mismatch; rebuilding: {cache}")
            print(f"[gpu_grid] cache data_root={meta.get('data_root')} val={meta.get('val_ids_csv')} image={meta.get('image_size')} resize={meta.get('resize_size')} pool={meta.get('feature_pool')} loader={meta.get('loader')} model={meta.get('model_name')} arch={meta.get('hub_arch')}")
            print(f"[gpu_grid] args  data_root={args.data_root} val={args.val_ids_csv} image={args.image_size} resize={args.resize_size} pool={args.feature_pool} loader={args.loader} model={args.model_name} arch={args.hub_arch}")
        except Exception as exc:
            print(f"[gpu_grid] failed to inspect cache; rebuilding {cache}: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gpu_grid] extract features on {device}: {args.hub_arch} {args.image_size}:{args.resize_size}")
    all_examples = load_train_examples(args.data_root)
    train_examples, val_examples = fixed_id_split(all_examples, args.val_ids_csv)
    test_examples = load_test_examples(args.data_root)
    print(f"[gpu_grid] train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}")

    set_seed(3407)
    model = load_dino_model(args).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    print("[gpu_grid] extract train features once")
    train_x, train_y, _ = extract_features(model, train_examples, args, device)
    print("[gpu_grid] extract val features once")
    val_x, val_y, _ = extract_features(model, val_examples, args, device)
    print("[gpu_grid] extract test features once")
    test_x, _, test_ids = extract_features(model, test_examples, args, device, test_hflip_tta=args.test_hflip_tta)

    payload = {
        "train_x": train_x.float().cpu(),
        "train_y": train_y.clamp_min(0).long().cpu(),
        "train_ids": [ex.image_id for ex in train_examples],
        "train_group_keys": [ex.group_key for ex in train_examples],
        "val_x": val_x.float().cpu(),
        "val_y": val_y.clamp_min(0).long().cpu(),
        "test_x": test_x.float().cpu(),
        "test_ids": list(test_ids),
        "data_root": str(args.data_root),
        "val_ids_csv": str(args.val_ids_csv),
        "image_size": args.image_size,
        "resize_size": args.resize_size,
        "feature_pool": args.feature_pool,
        "loader": args.loader,
        "model_name": args.model_name,
        "hub_arch": args.hub_arch,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    print(f"[gpu_grid] saved feature cache: {cache}")
    return cache


def load_sample_weights_csv(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    weights: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "weight" not in rows[0]:
        raise RuntimeError(f"{path} must contain a weight column")
    key_col = "id" if "id" in rows[0] else "group_key" if "group_key" in rows[0] else None
    if key_col is None:
        raise RuntimeError(f"{path} must contain id or group_key column")
    for row in rows:
        weight = float(row["weight"])
        if weight <= 0:
            raise ValueError(f"Sample weight must be positive: {row}")
        weights[row[key_col]] = weight
    return weights


def run_one_gpu(task: dict[str, object]) -> tuple[dict[str, object], list[float], dict[str, object], list[dict[str, object]]]:
    cache_path = Path(task["cache_path"])
    combo = task["combo"]
    seed, pos_w, mixup, dropout, smoothing, lr, wd = combo
    torch.set_num_threads(int(task["worker_threads"]))
    set_seed(int(seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(cache_path, map_location="cpu")
    train_x = data["train_x"]
    train_y = data["train_y"]
    val_x = data["val_x"]
    val_y = data["val_y"]
    test_x = data["test_x"]
    weight_map = load_sample_weights_csv(Path(task["hard_example_weights_csv"]) if task.get("hard_example_weights_csv") else None)
    train_ids = data.get("train_ids", [""] * len(train_y))
    train_group_keys = data.get("train_group_keys", [""] * len(train_y))
    train_w = torch.tensor(
        [float(weight_map.get(image_id, weight_map.get(group_key, 1.0))) for image_id, group_key in zip(train_ids, train_group_keys)],
        dtype=torch.float32,
    )
    val_topk = int(task["val_topk"])
    tag = make_tag(int(seed), float(pos_w), float(mixup), float(dropout), float(smoothing), float(lr), float(wd))

    head_args = SimpleNamespace(
        dropout=float(dropout),
        positive_class_weight=float(pos_w),
        label_smoothing=float(smoothing),
        lr=float(lr),
        weight_decay=float(wd),
        epochs=int(task["epochs"]),
        patience=int(task["patience"]),
        feature_mixup_alpha=float(mixup),
        ema_decay=float(task["ema_decay"]),
        grad_clip_norm=float(task["grad_clip_norm"]),
        lr_warmup_ratio=float(task["lr_warmup_ratio"]),
        early_stop_metric=str(task["early_stop_metric"]),
        val_topk=val_topk,
        target_pos_rate=0.0,
        fixed_threshold=float(task["fixed_threshold"]),
    )

    # Parallel workers would otherwise spam per-epoch logs. Keep main progress compact.
    with contextlib.redirect_stdout(io.StringIO()):
        head, best, history = train_head(train_x, train_y, train_w, val_x, val_y, head_args, device)
    head.eval()
    with torch.no_grad():
        val_probs = torch.softmax(head(val_x.to(device)), dim=1)[:, 1].float().cpu()
        test_probs = torch.softmax(head(test_x.to(device)), dim=1)[:, 1].float().cpu()
    val_topk_metrics = metrics_at_topk(val_y, val_probs, val_topk)
    record = {
        "rank": 0,
        "tag": tag,
        "seed": int(seed),
        "positive_class_weight": float(pos_w),
        "feature_mixup_alpha": float(mixup),
        "dropout": float(dropout),
        "label_smoothing": float(smoothing),
        "lr": float(lr),
        "weight_decay": float(wd),
        "hard_example_weights_csv": str(task.get("hard_example_weights_csv") or ""),
        "best_epoch": int(best["epoch"]),
        "best_score": float(best["score"]),
        "best_balanced_accuracy": float(best["metrics"]["balanced_accuracy"]),
        "best_threshold": float(best["threshold"]),
        "val_topk_f1": float(val_topk_metrics["f1"]),
        "val_topk_precision": float(val_topk_metrics["precision"]),
        "val_topk_recall": float(val_topk_metrics["recall"]),
        "val_topk_fp": int(val_topk_metrics["fp"]),
        "val_topk_fn": int(val_topk_metrics["fn"]),
    }
    best_no_state = {k: v for k, v in best.items() if k != "state"}
    return record, test_probs.tolist(), best_no_state, history


def write_ranked(output_dir: Path, results: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(results, key=lambda r: (r["val_topk_f1"], r["best_balanced_accuracy"]), reverse=True)
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    if ranked:
        write_csv(output_dir / "grid_results_live.csv", ranked, list(ranked[0].keys()))
    return ranked


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = build_or_load_feature_cache(args)
    cached = torch.load(cache, map_location="cpu")
    val_y = cached["val_y"]
    test_ids = list(cached["test_ids"])
    if args.target_pos_rate > 0:
        val_topk = max(1, round(len(val_y) * args.target_pos_rate))
    else:
        val_topk = max(1, int(val_y.long().sum().item()))
    print(f"[gpu_grid] val_topk={val_topk} val_pos={int(val_y.sum())} val_total={len(val_y)}")

    combos = list(itertools.product(
        parse_int_list(args.head_seeds),
        parse_float_list(args.positive_class_weights),
        parse_float_list(args.feature_mixup_alphas),
        parse_float_list(args.dropouts),
        parse_float_list(args.label_smoothings),
        parse_float_list(args.lrs),
        parse_float_list(args.weight_decays),
    ))
    if args.max_runs > 0:
        combos = combos[: args.max_runs]
    print(f"[gpu_grid] runs={len(combos)} parallel_workers={args.parallel_workers}")

    base_task = {
        "cache_path": str(cache),
        "val_topk": int(val_topk),
        "epochs": args.epochs,
        "patience": args.patience,
        "ema_decay": args.ema_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "lr_warmup_ratio": args.lr_warmup_ratio,
        "early_stop_metric": args.early_stop_metric,
        "fixed_threshold": args.fixed_threshold,
        "hard_example_weights_csv": str(args.hard_example_weights_csv) if args.hard_example_weights_csv else "",
        "worker_threads": args.worker_threads,
    }
    tasks = [dict(base_task, combo=combo) for combo in combos]

    results: list[dict[str, object]] = []
    candidate_probs: list[tuple[dict[str, object], torch.Tensor]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, args.parallel_workers), mp_context=ctx) as executor:
        futures = [executor.submit(run_one_gpu, task) for task in tasks]
        for done_idx, future in enumerate(as_completed(futures), start=1):
            record, probs, best_no_state, history = future.result()
            test_probs = torch.tensor(probs, dtype=torch.float32)
            results.append(record)
            candidate_probs.append((record, test_probs))
            run_dir = args.output_dir / "runs" / str(record["tag"])
            run_dir.mkdir(parents=True, exist_ok=True)
            write_probabilities(run_dir / "test_probabilities.csv", test_ids, test_probs)
            write_csv(run_dir / f"topk_{args.top_k}.csv", topk_submission_rows(test_ids, test_probs, args.top_k), ["id", "label"])
            with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump({"record": record, "best": best_no_state, "history": history}, handle, indent=2, sort_keys=True)
            ranked_live = write_ranked(args.output_dir, results)
            best = ranked_live[0]
            print(
                f"[gpu_grid] {done_idx}/{len(combos)} done {record['tag']} "
                f"f1={record['val_topk_f1']:.6f}; best={best['tag']} f1={best['val_topk_f1']:.6f}"
            )

    ranked = write_ranked(args.output_dir, results)
    write_csv(args.output_dir / "grid_results.csv", ranked, list(ranked[0].keys()))

    keep_tags = {r["tag"] for r in ranked[: args.save_top_n]}
    top_probs = [(r, p) for r, p in candidate_probs if r["tag"] in keep_tags]
    if top_probs:
        mean_scores = torch.stack([p for _, p in top_probs]).mean(dim=0)
        rank_scores = torch.stack([rank01(p) for _, p in top_probs]).mean(dim=0)
        ens_dir = args.output_dir / "ensemble_top"
        ens_dir.mkdir(parents=True, exist_ok=True)
        write_probabilities(ens_dir / "test_probabilities_mean_top.csv", test_ids, mean_scores)
        write_probabilities(ens_dir / "test_probabilities_rank_top.csv", test_ids, rank_scores)
        write_csv(ens_dir / f"topk_{args.top_k}_mean_top.csv", topk_submission_rows(test_ids, mean_scores, args.top_k), ["id", "label"])
        write_csv(ens_dir / f"topk_{args.top_k}_rank_top.csv", topk_submission_rows(test_ids, rank_scores, args.top_k), ["id", "label"])
        with (ens_dir / "members.json").open("w", encoding="utf-8") as handle:
            json.dump([r for r, _ in top_probs], handle, indent=2, sort_keys=True)
    print(f"[gpu_grid] done output={args.output_dir}")
    print("[gpu_grid] top5")
    for r in ranked[:5]:
        print(r)


if __name__ == "__main__":
    main()
