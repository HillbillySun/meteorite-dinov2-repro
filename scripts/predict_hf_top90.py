#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageClassification

warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate top-90 submission with the Hugging Face full model.")
    p.add_argument("--model-id", default="Eki734/meteorite-dinov2-b14-direct")
    p.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    p.add_argument("--output-csv", type=Path, default=Path("runs/hf_direct_top90/topk_90.csv"))
    p.add_argument("--prob-csv", type=Path, default=Path("runs/hf_direct_top90/test_probabilities.csv"))
    p.add_argument("--reference-csv", type=Path, default=Path("kaggle_online_results.csv"))
    p.add_argument("--top-k", type=int, default=90)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def read_labels(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    label_col = [c for c in rows[0].keys() if c != "id"][0]
    return {r["id"]: str(r[label_col]) for r in rows}


def main() -> None:
    args = parse_args()
    if not args.test_dir.exists():
        raise FileNotFoundError(f"Missing test image directory: {args.test_dir}")

    image_paths = sorted([
        p for p in args.test_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])
    if not image_paths:
        raise RuntimeError(f"No images found in {args.test_dir}")

    print(f"[hf_top90] load model: {args.model_id}")
    model = AutoModelForImageClassification.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval().to(args.device)

    rows: list[tuple[str, float]] = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Predict", ncols=80):
            image = Image.open(path).convert("RGB")
            prob = float(model.predict(image)["prob_meteorite"][0])
            rows.append((path.name, prob))

    ranked = sorted(rows, key=lambda x: x[1], reverse=True)
    positive_ids = {name for name, _ in ranked[:args.top_k]}

    args.prob_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.prob_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prob_meteorite"])
        for name, prob in rows:
            writer.writerow([name, f"{prob:.8f}"])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for name, _ in rows:
            writer.writerow([name, 1 if name in positive_ids else 0])

    cutoff = ranked[args.top_k - 1]
    next_item = ranked[args.top_k] if args.top_k < len(ranked) else None
    print(f"[hf_top90] saved probabilities: {args.prob_csv}")
    print(f"[hf_top90] saved submission: {args.output_csv}")
    print(f"[hf_top90] top_k={args.top_k} cutoff={cutoff} next={next_item}")

    if args.reference_csv.exists():
        ref = read_labels(args.reference_csv)
        cur = read_labels(args.output_csv)
        diff = [(k, ref.get(k), cur.get(k)) for k in sorted(ref) if ref.get(k) != cur.get(k)]
        print(f"[hf_top90] diff_vs_reference={len(diff)}")
        if diff:
            print("[hf_top90] first_diffs=", diff[:20])
        else:
            print("[hf_top90] matches reference exactly")


if __name__ == "__main__":
    main()
