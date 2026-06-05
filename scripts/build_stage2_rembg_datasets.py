from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from rembg import new_session, remove
from tqdm import tqdm


IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build unified-rembg stage2-style train datasets.')
    parser.add_argument('--project-root', type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument('--train-csv', type=Path, default=Path('data/train_labels.csv'))
    parser.add_argument('--original-dir', type=Path, default=Path('data/train_images'))
    parser.add_argument('--rembg-dir', type=Path, default=Path('data/train_images_stage2_rembg_all'))
    parser.add_argument('--rembg-labels', type=Path, default=Path('data/train_labels_stage2_rembg_all.csv'))
    parser.add_argument('--mix-dir', type=Path, default=Path('data/train_images_stage2_rembg_mix_original'))
    parser.add_argument('--mix-labels', type=Path, default=Path('data/train_labels_stage2_rembg_mix_original.csv'))
    parser.add_argument('--metadata-csv', type=Path, default=Path('data/train_images_stage2_rembg_all_meta.csv'))
    parser.add_argument('--max-side', type=int, default=1024)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resize_longest(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    width, height = image.size
    scale = max_side / max(width, height)
    if scale >= 1:
        return image.copy(), 1.0
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS), scale


def save_rgb(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert('RGB')
    if out_path.suffix.lower() in {'.jpg', '.jpeg'}:
        rgb.save(out_path, quality=95)
    else:
        rgb.save(out_path)


def copy_or_convert(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.suffix.lower() == dst_path.suffix.lower():
        shutil.copy2(src_path, dst_path)
        return
    with Image.open(src_path) as image:
        save_rgb(image, dst_path)


def white_ratio(image: Image.Image, value: int = 245) -> float:
    probe = image.copy()
    probe.thumbnail((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(probe.convert('RGB'))
    return float(np.all(arr >= value, axis=2).mean())


def rembg_position_preserving(src_path: Path, out_path: Path, session, max_side: int) -> tuple[str, str, float]:
    with Image.open(src_path) as image:
        image = image.convert('RGB')

    small, scale = resize_longest(image, max_side=max_side)
    small_mask = remove(small, session=session, only_mask=True).convert('L')
    small_mask = small_mask.filter(ImageFilter.MaxFilter(5))
    small_mask = small_mask.filter(ImageFilter.MinFilter(5))
    small_mask = small_mask.filter(ImageFilter.GaussianBlur(1.2))
    # rembg usually preserves the input mask size, but some images can return a
    # mask with a different size even when we did not downscale the input.
    # Always align the mask to the original image before compositing.
    mask = small_mask
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.BICUBIC)

    bbox = mask.getbbox()
    if bbox is None:
        canvas = image.copy()
        bbox_text = ''
        status = 'no_mask_fallback_original'
    else:
        canvas = Image.new('RGB', image.size, 'white')
        canvas.paste(image, (0, 0), mask)
        bbox_text = ','.join(map(str, bbox))
        status = 'rembg_position_preserving'

    ratio = white_ratio(canvas, value=245)
    save_rgb(canvas, out_path)
    return status, bbox_text, ratio


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    train_csv = resolve(args.train_csv, root)
    original_dir = resolve(args.original_dir, root)
    rembg_dir = resolve(args.rembg_dir, root)
    rembg_labels = resolve(args.rembg_labels, root)
    mix_dir = resolve(args.mix_dir, root)
    mix_labels = resolve(args.mix_labels, root)
    metadata_csv = resolve(args.metadata_csv, root)

    rows = read_rows(train_csv)
    rembg_dir.mkdir(parents=True, exist_ok=True)
    mix_dir.mkdir(parents=True, exist_ok=True)

    session = new_session('u2net')
    metadata_rows: list[dict[str, object]] = []
    rembg_label_rows: list[dict[str, object]] = []
    mix_label_rows: list[dict[str, object]] = []

    for row in tqdm(rows, desc='Building unified rembg datasets'):
        image_id = row['id']
        label = row['label']
        src_path = original_dir / image_id
        if not src_path.exists():
            raise FileNotFoundError(f'Missing original image: {src_path}')

        rembg_path = rembg_dir / image_id
        if args.overwrite or not rembg_path.exists():
            status, bbox, ratio = rembg_position_preserving(src_path, rembg_path, session=session, max_side=args.max_side)
        else:
            with Image.open(rembg_path) as image:
                ratio = white_ratio(image, value=245)
            status, bbox = 'skipped_existing', ''

        rembg_label_rows.append({'id': image_id, 'label': label})
        metadata_rows.append({
            'id': image_id,
            'label': label,
            'src_path': str(src_path),
            'rembg_path': str(rembg_path),
            'status': status,
            'bbox': bbox,
            'nearwhite_ratio_245': f'{ratio:.8f}',
        })

        stem = src_path.stem
        suffix = src_path.suffix.lower()
        orig_id = f'{stem}_orig{suffix}'
        rembg_id = f'{stem}_rembg{suffix}'
        copy_or_convert(src_path, mix_dir / orig_id)
        copy_or_convert(rembg_path, mix_dir / rembg_id)
        mix_label_rows.append({'id': orig_id, 'label': label})
        mix_label_rows.append({'id': rembg_id, 'label': label})

    write_rows(rembg_labels, rembg_label_rows, ['id', 'label'])
    write_rows(mix_labels, mix_label_rows, ['id', 'label'])
    write_rows(metadata_csv, metadata_rows, ['id', 'label', 'src_path', 'rembg_path', 'status', 'bbox', 'nearwhite_ratio_245'])

    print(f'Saved rembg dataset: {rembg_dir} images={len(rembg_label_rows)} labels={rembg_labels}')
    print(f'Saved mixed dataset: {mix_dir} images={len(mix_label_rows)} labels={mix_labels}')
    print(f'Saved metadata: {metadata_csv}')


if __name__ == '__main__':
    main()
