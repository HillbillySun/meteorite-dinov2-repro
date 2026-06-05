from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class TestProfile:
    width: int
    height: int
    bbox_area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build stage2_style_v3: a conservative train style closer to stage2 test "
            "by filtering blank rembg failures, enlarging foreground, and adding mild background variation."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-csv", type=Path, default=Path("data/train_labels.csv"))
    parser.add_argument("--original-dir", type=Path, default=Path("data/train_images"))
    parser.add_argument("--rembg-dir", type=Path, default=Path("data/train_images_stage2_rembg_all"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test_images"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/train_images_stage2_style_v3"))
    parser.add_argument("--output-labels", type=Path, default=Path("data/train_labels_stage2_style_v3.csv"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/train_images_stage2_style_v3_meta.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/stage2_style_v3_dino_root"))
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=0, help="Debug only: limit source train images.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--canvas-max-side", type=int, default=1024)
    parser.add_argument("--white-threshold", type=int, default=245)
    parser.add_argument("--min-source-area", type=float, default=0.006)
    parser.add_argument("--max-source-area", type=float, default=0.86)
    parser.add_argument("--allow-original-fallback", action="store_true")
    parser.add_argument("--disable-component-filter", action="store_true")
    parser.add_argument("--min-target-area", type=float, default=0.22)
    parser.add_argument("--max-target-area", type=float, default=0.96)
    parser.add_argument("--target-area-multiplier", type=float, default=1.10)
    parser.add_argument("--max-shift", type=float, default=0.07)
    parser.add_argument("--background-jitter-prob", type=float, default=0.35)
    parser.add_argument("--shadow-prob", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scaled_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = max_side / max(width, height)
    if scale >= 1:
        return width, height
    return max(1, round(width * scale)), max(1, round(height * scale))


def save_rgb(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        rgb.save(out_path, quality=95)
    else:
        rgb.save(out_path)


def foreground_bbox(
    image: Image.Image,
    white_threshold: int,
    probe_max_side: int = 768,
) -> tuple[int, int, int, int] | None:
    width, height = image.size
    probe = image.convert("RGB")
    probe.thumbnail((probe_max_side, probe_max_side), Image.Resampling.BILINEAR)
    arr = np.asarray(probe)
    mask = ~np.all(arr >= white_threshold, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    scale_x = width / probe.width
    scale_y = height / probe.height
    pad_x = max(2, round(scale_x * 3))
    pad_y = max(2, round(scale_y * 3))
    x0 = max(0, math.floor(float(xs.min()) * scale_x) - pad_x)
    y0 = max(0, math.floor(float(ys.min()) * scale_y) - pad_y)
    x1 = min(width, math.ceil(float(xs.max() + 1) * scale_x) + pad_x)
    y1 = min(height, math.ceil(float(ys.max() + 1) * scale_y) + pad_y)
    return x0, y0, x1, y1


def bbox_area_fraction(bbox: tuple[int, int, int, int] | None, size: tuple[int, int]) -> float:
    if bbox is None:
        return 0.0
    width, height = size
    x0, y0, x1, y1 = bbox
    return max(0.0, ((x1 - x0) * (y1 - y0)) / max(1, width * height))


def marker_like_component(
    area_frac: float,
    bbox_frac: float,
    aspect: float,
    fill: float,
    center_y: float,
    rel_height: float,
) -> bool:
    # Calibration cubes / scale rulers are usually dense geometric objects,
    # while meteorites tend to have irregular masks. Keep this conservative:
    # only remove small-to-medium, high-fill, boxy/strip-like components.
    square_cube = 0.55 <= aspect <= 1.55 and fill >= 0.62 and bbox_frac <= 0.26 and area_frac <= 0.20
    long_ruler = (aspect >= 2.2 or aspect <= 0.45) and fill >= 0.40 and bbox_frac <= 0.32 and area_frac <= 0.22
    bottom_paper_scale = (
        aspect >= 2.0
        and rel_height <= 0.22
        and center_y >= 0.58
        and bbox_frac <= 0.24
        and area_frac <= 0.18
        and fill >= 0.22
    )
    return square_cube or long_ruler or bottom_paper_scale


def foreground_bbox_component_filtered(
    image: Image.Image,
    white_threshold: int,
    probe_max_side: int = 768,
) -> tuple[tuple[int, int, int, int] | None, str]:
    width, height = image.size
    probe = image.convert("RGB")
    probe.thumbnail((probe_max_side, probe_max_side), Image.Resampling.BILINEAR)
    arr = np.asarray(probe)
    mask = ~np.all(arr >= white_threshold, axis=2)
    if not mask.any():
        return None, "blank"

    labels, n_labels = ndi.label(mask)
    if n_labels <= 0:
        return None, "blank"

    probe_area = max(1, probe.width * probe.height)
    candidates = []
    objects = ndi.find_objects(labels)
    for label_idx, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        ys, xs = slc
        comp = labels[slc] == label_idx
        area = int(comp.sum())
        if area < 16:
            continue
        bw = int(xs.stop - xs.start)
        bh = int(ys.stop - ys.start)
        bbox_pix = max(1, bw * bh)
        area_frac = area / probe_area
        bbox_frac = bbox_pix / probe_area
        aspect = bw / max(1, bh)
        fill = area / bbox_pix
        center_y = ((ys.start + ys.stop) / 2) / max(1, probe.height)
        rel_height = bh / max(1, probe.height)
        marker = marker_like_component(area_frac, bbox_frac, aspect, fill, center_y, rel_height)
        candidates.append(
            {
                "label": label_idx,
                "area": area,
                "bbox": (xs.start, ys.start, xs.stop, ys.stop),
                "marker": marker,
                "area_frac": area_frac,
                "bbox_frac": bbox_frac,
                "aspect": aspect,
                "fill": fill,
                "center_y": center_y,
                "rel_height": rel_height,
            }
        )

    if not candidates:
        return None, "blank"

    non_marker = [c for c in candidates if not c["marker"]]
    if non_marker:
        selected = max(non_marker, key=lambda c: c["area"])
        removed_markers = len(candidates) - len(non_marker)
        if removed_markers > 0:
            status = f"component_filtered_removed{removed_markers}"
        else:
            status = "component_filtered" if len(candidates) > 1 else "single_component"
    else:
        # A single dense square/strip component is more likely to be a calibration marker
        # than a meteorite. Skipping is safer than teaching the classifier marker features.
        return None, "marker_only"

    x0, y0, x1, y1 = selected["bbox"]
    scale_x = width / probe.width
    scale_y = height / probe.height
    pad_x = max(2, round(scale_x * 4))
    pad_y = max(2, round(scale_y * 4))
    out = (
        max(0, math.floor(float(x0) * scale_x) - pad_x),
        max(0, math.floor(float(y0) * scale_y) - pad_y),
        min(width, math.ceil(float(x1) * scale_x) + pad_x),
        min(height, math.ceil(float(y1) * scale_y) + pad_y),
    )
    return out, status


def image_stats(image: Image.Image, white_threshold: int) -> dict[str, float]:
    probe = image.convert("RGB")
    probe.thumbnail((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(probe).astype(np.float32)
    white = np.all(arr >= white_threshold, axis=2)
    gray = arr.mean(axis=2)
    edge = (np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean()) / 255.0
    return {
        "nearwhite_ratio": float(white.mean()),
        "dark_ratio": float((gray < 70).mean()),
        "edge_density": float(edge),
    }


def collect_test_profiles(test_dir: Path, white_threshold: int, canvas_max_side: int) -> list[TestProfile]:
    profiles: list[TestProfile] = []
    for path in sorted(test_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = scaled_size(image.width, image.height, canvas_max_side)
            bbox = foreground_bbox(image, white_threshold=white_threshold)
            area = bbox_area_fraction(bbox, image.size)
        profiles.append(TestProfile(width=width, height=height, bbox_area=area if area > 0 else 0.60))
    if not profiles:
        raise ValueError(f"No test images found in {test_dir}")
    return profiles


def crop_foreground(
    rembg_path: Path,
    original_path: Path,
    white_threshold: int,
    min_source_area: float,
    max_source_area: float,
    allow_original_fallback: bool = False,
    component_filter: bool = True,
) -> tuple[Image.Image | None, str, float]:
    with Image.open(rembg_path) as image:
        rembg = image.convert("RGB")
    if component_filter:
        bbox, comp_status = foreground_bbox_component_filtered(rembg, white_threshold=white_threshold)
    else:
        bbox = foreground_bbox(rembg, white_threshold=white_threshold)
        comp_status = "whole_foreground"
    rembg_area = bbox_area_fraction(bbox, rembg.size)
    if bbox is not None and rembg_area >= min_source_area:
        if rembg_area <= max_source_area:
            return rembg.crop(bbox), f"rembg_{comp_status}", rembg_area
        return None, "skipped_full_scene_or_connected_marker", rembg_area

    if not allow_original_fallback:
        # Original images often contain rulers, cubes, labels, dark tables, or support stands.
        # Using them as fallback reintroduces exactly the shortcuts we are trying to remove.
        return None, "skipped_no_clean_rembg_foreground", rembg_area

    # Optional recovery path: disabled by default because it can preserve rulers/cubes.
    with Image.open(original_path) as image:
        original = image.convert("RGB")
    if component_filter:
        orig_bbox, orig_status = foreground_bbox_component_filtered(original, white_threshold=white_threshold)
    else:
        orig_bbox = foreground_bbox(original, white_threshold=white_threshold)
        orig_status = "whole_foreground"
    orig_area = bbox_area_fraction(orig_bbox, original.size)
    if orig_bbox is not None and min_source_area <= orig_area <= max_source_area:
        return original.crop(orig_bbox), f"fallback_original_{orig_status}", orig_area
    return None, "skipped_blank_tiny_marker_or_full_scene", max(rembg_area, orig_area)


def resize_crop_to_target(crop: Image.Image, canvas_size: tuple[int, int], target_area: float) -> Image.Image:
    crop_w, crop_h = crop.size
    canvas_w, canvas_h = canvas_size
    aspect = crop_w / max(1, crop_h)
    target_pixels = target_area * canvas_w * canvas_h
    new_h = math.sqrt(target_pixels / max(aspect, 1e-6))
    new_w = new_h * aspect
    max_w = canvas_w * 0.98
    max_h = canvas_h * 0.98
    scale = min(max_w / max(new_w, 1), max_h / max(new_h, 1), 1.0)
    new_w = max(1, round(new_w * scale))
    new_h = max(1, round(new_h * scale))
    return crop.resize((new_w, new_h), Image.Resampling.LANCZOS)


def make_background(size: tuple[int, int], rng: random.Random, jitter: bool) -> Image.Image:
    width, height = size
    if not jitter:
        return Image.new("RGB", size, "white")

    base = rng.randint(238, 252)
    y_grad = np.linspace(rng.randint(-5, 2), rng.randint(0, 7), height, dtype=np.float32)[:, None]
    x_grad = np.linspace(rng.randint(-3, 3), rng.randint(-3, 3), width, dtype=np.float32)[None, :]
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(0.8, 2.2), (height, width))
    gray = np.clip(base + y_grad + x_grad + noise, 226, 255).astype(np.uint8)
    arr = np.stack([gray, gray, gray], axis=2)
    return Image.fromarray(arr, mode="RGB")


def blend_crop_background(crop: Image.Image, bg_patch: Image.Image, white_threshold: int) -> Image.Image:
    crop_arr = np.asarray(crop.convert("RGB")).copy()
    bg_arr = np.asarray(bg_patch.convert("RGB"))
    white = np.all(crop_arr >= white_threshold, axis=2)
    # Keep the meteorite pixels, but let white rembg margins inherit the canvas color.
    crop_arr[white] = bg_arr[white]
    return Image.fromarray(crop_arr, mode="RGB")


def add_soft_shadow(canvas: Image.Image, box: tuple[int, int, int, int], rng: random.Random) -> Image.Image:
    x, y, w, h = box
    shadow = Image.new("L", canvas.size, 0)
    sx = x + rng.randint(-max(1, w // 18), max(1, w // 18))
    sy = y + rng.randint(max(1, h // 18), max(2, h // 10))
    sw = max(1, round(w * rng.uniform(0.75, 1.08)))
    sh = max(1, round(h * rng.uniform(0.18, 0.34)))
    tmp = Image.new("L", (sw, sh), rng.randint(16, 34))
    tmp = tmp.filter(ImageFilter.GaussianBlur(radius=max(3, round(min(w, h) * 0.035))))
    shadow.paste(tmp, (max(0, sx), min(canvas.height - 1, sy)))
    arr = np.asarray(canvas.convert("RGB")).astype(np.int16)
    sh_arr = np.asarray(shadow).astype(np.int16)
    arr = np.clip(arr - sh_arr[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def compose_variant(
    crop: Image.Image,
    profile: TestProfile,
    rng: random.Random,
    min_target_area: float,
    max_target_area: float,
    target_area_multiplier: float,
    max_shift: float,
    background_jitter_prob: float,
    shadow_prob: float,
    white_threshold: int,
) -> Image.Image:
    canvas_size = (profile.width, profile.height)
    jitter = rng.uniform(0.94, 1.20)
    target_area = min(max_target_area, max(min_target_area, profile.bbox_area * target_area_multiplier * jitter))
    resized = resize_crop_to_target(crop, canvas_size=canvas_size, target_area=target_area)

    canvas_w, canvas_h = canvas_size
    max_dx = round(canvas_w * max_shift)
    max_dy = round(canvas_h * max_shift)
    center_x = canvas_w // 2 + rng.randint(-max_dx, max_dx)
    center_y = canvas_h // 2 + rng.randint(-max_dy, max_dy)
    x = min(max(0, center_x - resized.width // 2), max(0, canvas_w - resized.width))
    y = min(max(0, center_y - resized.height // 2), max(0, canvas_h - resized.height))

    jitter_bg = rng.random() < background_jitter_prob
    canvas = make_background(canvas_size, rng=rng, jitter=jitter_bg)
    if jitter_bg and rng.random() < shadow_prob:
        canvas = add_soft_shadow(canvas, (x, y, resized.width, resized.height), rng)

    bg_patch = canvas.crop((x, y, x + resized.width, y + resized.height))
    pasted = blend_crop_background(resized, bg_patch, white_threshold=white_threshold)
    canvas.paste(pasted, (x, y))
    return canvas


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_path(src: Path, dst: Path) -> None:
    remove_existing(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    train_csv = resolve(args.train_csv, root)
    original_dir = resolve(args.original_dir, root)
    rembg_dir = resolve(args.rembg_dir, root)
    test_dir = resolve(args.test_dir, root)
    sample_submission = resolve(args.sample_submission, root)
    output_dir = resolve(args.output_dir, root)
    output_labels = resolve(args.output_labels, root)
    metadata_csv = resolve(args.metadata_csv, root)
    output_root = resolve(args.output_root, root)

    rows = read_rows(train_csv)
    if args.max_images > 0:
        rows = rows[: args.max_images]
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = collect_test_profiles(test_dir, args.white_threshold, args.canvas_max_side)

    rng = random.Random(args.seed)
    label_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for row in tqdm(rows, desc="Building stage2_style_v3"):
        image_id = row["id"]
        label = row["label"]
        original_path = original_dir / image_id
        rembg_path = rembg_dir / image_id
        if not original_path.exists():
            raise FileNotFoundError(f"Missing original image: {original_path}")
        if not rembg_path.exists():
            raise FileNotFoundError(f"Missing rembg image: {rembg_path}")

        crop, source_status, source_bbox_area = crop_foreground(
            rembg_path=rembg_path,
            original_path=original_path,
            white_threshold=args.white_threshold,
            min_source_area=args.min_source_area,
            max_source_area=args.max_source_area,
            allow_original_fallback=args.allow_original_fallback,
            component_filter=not args.disable_component_filter,
        )
        if crop is None:
            skipped_rows.append(
                {
                    "base_id": image_id,
                    "label": label,
                    "status": source_status,
                    "source_bbox_area": f"{source_bbox_area:.8f}",
                }
            )
            continue

        stem = Path(image_id).stem
        suffix = Path(image_id).suffix.lower()
        for variant_idx in range(args.variants):
            out_id = f"{stem}_s{variant_idx:02d}{suffix}"
            out_path = output_dir / out_id
            profile = rng.choice(profiles)
            if args.overwrite or not out_path.exists():
                variant = compose_variant(
                    crop=crop,
                    profile=profile,
                    rng=rng,
                    min_target_area=args.min_target_area,
                    max_target_area=args.max_target_area,
                    target_area_multiplier=args.target_area_multiplier,
                    max_shift=args.max_shift,
                    background_jitter_prob=args.background_jitter_prob,
                    shadow_prob=args.shadow_prob,
                    white_threshold=args.white_threshold,
                )
                save_rgb(variant, out_path)
            else:
                with Image.open(out_path) as image:
                    variant = image.convert("RGB")

            stats = image_stats(variant, args.white_threshold)
            label_rows.append({"id": out_id, "label": label})
            metadata_rows.append(
                {
                    "id": out_id,
                    "base_id": image_id,
                    "label": label,
                    "status": source_status,
                    "source_bbox_area": f"{source_bbox_area:.8f}",
                    "canvas_width": variant.width,
                    "canvas_height": variant.height,
                    "nearwhite_ratio_245": f"{stats['nearwhite_ratio']:.8f}",
                    "dark_ratio": f"{stats['dark_ratio']:.8f}",
                    "edge_density": f"{stats['edge_density']:.8f}",
                }
            )

    write_rows(output_labels, label_rows, ["id", "label"])
    write_rows(
        metadata_csv,
        metadata_rows,
        [
            "id",
            "base_id",
            "label",
            "status",
            "source_bbox_area",
            "canvas_width",
            "canvas_height",
            "nearwhite_ratio_245",
            "dark_ratio",
            "edge_density",
        ],
    )
    skipped_csv = metadata_csv.with_name(metadata_csv.stem + "_skipped.csv")
    write_rows(skipped_csv, skipped_rows, ["base_id", "label", "status", "source_bbox_area"])

    output_root.mkdir(parents=True, exist_ok=True)
    copy_path(output_dir, output_root / "train_images")
    copy_path(output_labels, output_root / "train_labels.csv")
    copy_path(test_dir, output_root / "test_images")
    copy_path(sample_submission, output_root / "sample_submission.csv")

    print(f"Saved stage2_style_v3 dataset: {output_dir} images={len(label_rows)}")
    print(f"Saved labels: {output_labels}")
    print(f"Saved metadata: {metadata_csv}")
    print(f"Saved skipped audit: {skipped_csv} skipped_base_images={len(skipped_rows)}")
    print(f"Saved dino root: {output_root}")


if __name__ == "__main__":
    main()
