#!/usr/bin/env bash
set -euo pipefail

# Train one probe with the known best hyperparameters from the original grid
# feature cache and save a checkpoint.
#
# Outputs:
#   ${OUT_DIR}/best_probe.pt
#   ${OUT_DIR}/checkpoint_summary.json
#   ${OUT_DIR}/test_probabilities.csv
#   ${OUT_DIR}/topk_90.csv

FEATURE_CACHE="${FEATURE_CACHE:-runs/grid_original_like_topkf1_zero_raw_v1/features_cache_tta.pt}"
OUT_DIR="${OUT_DIR:-runs/train_best_known_hparams_checkpoint_zero_raw_v1}"
TOP_K="${TOP_K:-90}"
REF="${REF:-kaggle_online_results.csv}"

if [[ ! -f "${FEATURE_CACHE}" ]]; then
  echo "[best_ckpt] missing FEATURE_CACHE=${FEATURE_CACHE}"
  echo "[best_ckpt] run the full grid first or pass FEATURE_CACHE=/path/to/features_cache_tta.pt"
  exit 1
fi

python scripts/train_best_known_hparams_checkpoint.py \
  --feature-cache "${FEATURE_CACHE}" \
  --output-dir "${OUT_DIR}" \
  --top-k "${TOP_K}" \
  --reference-csv "${REF}" \
  --seed 3407 \
  --positive-class-weight 1.00 \
  --feature-mixup-alpha 0.05 \
  --dropout 0.45 \
  --label-smoothing 0.03 \
  --lr 0.0005 \
  --weight-decay 0.01 \
  --epochs 70 \
  --patience 10 \
  --ema-decay 0.995 \
  --lr-warmup-ratio 0.1 \
  --grad-clip-norm 1.0 \
  --early-stop-metric topk_f1 \
  --fixed-threshold 0.5
