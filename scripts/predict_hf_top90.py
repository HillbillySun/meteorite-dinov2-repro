#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageClassification

warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a top-k submission from Hugging Face, ModelScope, or a local model directory."
    )
    parser.add_argument(
        "--model-source",
        choices=("huggingface", "modelscope", "local"),
        default="huggingface",
        help="Where to resolve --model-id. Default: huggingface.",
    )
    parser.add_argument("--model-id", default="Eki734/meteorite-dinov2-b14-direct")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face or ModelScope revision.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional model download cache directory.")
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--output-csv", type=Path, default=Path("runs/hf_direct_top90/topk_90.csv"))
    parser.add_argument("--prob-csv", type=Path, default=Path("runs/hf_direct_top90/test_probabilities.csv"))
    parser.add_argument("--reference-csv", type=Path, default=Path("kaggle_online_results.csv"))
    parser.add_argument("--top-k", type=int, default=90)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_model_path(
    source: Literal["huggingface", "modelscope", "local"],
    model_id: str,
    revision: str | None,
    cache_dir: Path | None,
) -> str:
    if source == "local":
        model_path = Path(model_id).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
        return str(model_path)

    if source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "ModelScope support requires the 'modelscope' package. "
                "Install the repository requirements or run: pip install modelscope"
            ) from exc

        kwargs: dict[str, object] = {}
        if revision:
            kwargs["revision"] = revision
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)
        print(f"[top90] download from ModelScope: {model_id}")
        return str(snapshot_download(model_id=model_id, **kwargs))

    return model_id


def read_labels(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    label_col = next(column for column in rows[0] if column != "id")
    return {row["id"]: str(row[label_col]) for row in rows}


def main() -> None:
    args = parse_args()
    if not args.test_dir.exists():
        raise FileNotFoundError(f"Missing test image directory: {args.test_dir}")

    image_paths = sorted(
        path
        for path in args.test_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {args.test_dir}")
    if not 0 < args.top_k <= len(image_paths):
        raise ValueError(f"--top-k must be between 1 and {len(image_paths)}, got {args.top_k}")

    resolved_model = resolve_model_path(
        args.model_source,
        args.model_id,
        args.revision,
        args.cache_dir,
    )
    load_kwargs: dict[str, object] = {"trust_remote_code": True}
    if args.model_source == "huggingface":
        if args.revision:
            load_kwargs["revision"] = args.revision
        if args.cache_dir:
            load_kwargs["cache_dir"] = str(args.cache_dir)

    print(f"[top90] source={args.model_source} model={args.model_id}")
    print(f"[top90] resolved_model={resolved_model}")
    model = AutoModelForImageClassification.from_pretrained(resolved_model, **load_kwargs)
    model.eval().to(args.device)

    rows: list[tuple[str, float]] = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Predict", ncols=80):
            image = Image.open(path).convert("RGB")
            probability = float(model.predict(image)["prob_meteorite"][0])
            rows.append((path.name, probability))

    ranked = sorted(rows, key=lambda item: item[1], reverse=True)
    positive_ids = {name for name, _ in ranked[: args.top_k]}

    args.prob_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.prob_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prob_meteorite"])
        for name, probability in rows:
            writer.writerow([name, f"{probability:.8f}"])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        for name, _ in rows:
            writer.writerow([name, 1 if name in positive_ids else 0])

    cutoff = ranked[args.top_k - 1]
    next_item = ranked[args.top_k] if args.top_k < len(ranked) else None
    print(f"[top90] saved probabilities: {args.prob_csv}")
    print(f"[top90] saved submission: {args.output_csv}")
    print(f"[top90] top_k={args.top_k} cutoff={cutoff} next={next_item}")

    if args.reference_csv.exists():
        reference = read_labels(args.reference_csv)
        current = read_labels(args.output_csv)
        all_ids = sorted(set(reference) | set(current))
        differences = [
            (image_id, reference.get(image_id), current.get(image_id))
            for image_id in all_ids
            if reference.get(image_id) != current.get(image_id)
        ]
        print(f"[top90] diff_vs_reference={len(differences)}")
        if differences:
            print("[top90] first_diffs=", differences[:20])
        else:
            print("[top90] matches reference exactly")


if __name__ == "__main__":
    main()
