#!/usr/bin/env bash
set -euo pipefail

# Build the processed style-v3 relaxed dataset from official raw images.
# Input:
#   data/train_images/
#   data/test_images/
#   data/train_labels.csv
#   data/sample_submission.csv
# Output:
#   data/${OUT_TAG}/stage2_style_v3_relaxed_dino_root/

OUT_TAG="${OUT_TAG:-zero_raw_v1}"
WORK_ROOT="data/${OUT_TAG}"
mkdir -p "${WORK_ROOT}"

echo "[build_zero] out_tag=${OUT_TAG}"
echo "[build_zero] work_root=${WORK_ROOT}"

for required in data/train_images data/test_images data/train_labels.csv data/sample_submission.csv; do
  if [[ ! -e "${required}" ]]; then
    echo "[build_zero] missing required input: ${required}"
    exit 1
  fi
done

echo "[build_zero] Step 1/2: rembg from raw train_images"
python scripts/build_stage2_rembg_datasets.py \
  --train-csv data/train_labels.csv \
  --original-dir data/train_images \
  --rembg-dir "${WORK_ROOT}/train_images_stage2_rembg_all" \
  --rembg-labels "${WORK_ROOT}/train_labels_stage2_rembg_all.csv" \
  --mix-dir "${WORK_ROOT}/train_images_stage2_rembg_mix_original" \
  --mix-labels "${WORK_ROOT}/train_labels_stage2_rembg_mix_original.csv" \
  --metadata-csv "${WORK_ROOT}/train_images_stage2_rembg_all_meta.csv" \
  --overwrite

echo "[build_zero] Step 2/2: build style-v3 relaxed dataset"
python scripts/build_stage2_style_v3_dataset.py \
  --train-csv data/train_labels.csv \
  --original-dir data/train_images \
  --rembg-dir "${WORK_ROOT}/train_images_stage2_rembg_all" \
  --test-dir data/test_images \
  --sample-submission data/sample_submission.csv \
  --output-dir "${WORK_ROOT}/train_images_stage2_style_v3_relaxed" \
  --output-labels "${WORK_ROOT}/train_labels_stage2_style_v3_relaxed.csv" \
  --metadata-csv "${WORK_ROOT}/train_images_stage2_style_v3_relaxed_meta.csv" \
  --min-source-area 0.003 \
  --max-source-area 0.90 \
  --min-target-area 0.20 \
  --max-target-area 0.96 \
  --target-area-multiplier 1.08 \
  --background-jitter-prob 0.35 \
  --shadow-prob 0.25 \
  --output-root "${WORK_ROOT}/stage2_style_v3_relaxed_dino_root" \
  --overwrite

echo "[build_zero] done"
echo "[build_zero] processed root: ${WORK_ROOT}/stage2_style_v3_relaxed_dino_root"
