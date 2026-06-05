#!/usr/bin/env bash
set -euo pipefail

# Original-repo-style DINOv2-B relaxed head grid.
# Mirrors MTR-cls-2026/scripts/run_dinov2_b14_relaxed_data_head_ablation.sh
# for the grid stage:
#   - selection/early stopping: topk_f1
#   - val_topk: number of positive samples in val set (--target-pos-rate 0)
#   - test hflip TTA enabled
#   - 3 seeds x 2 pos weights x 4 mixups x 3 dropouts x 3 smoothings x 2 lrs = 432 runs
# This script does NOT use outputs/best_submission/topk_90.csv as a selection signal.

DATA_ROOT="${DATA_ROOT:-data/zero_raw_v1/stage2_style_v3_relaxed_dino_root}"
OUT_DIR="${OUT_DIR:-runs/grid_original_like_topkf1_zero_raw_v1}"
FEATURE_CACHE="${FEATURE_CACHE:-${OUT_DIR}/features_cache_tta.pt}"
VAL_IDS_CSV="${VAL_IDS_CSV:-data/splits/hardval_ablation/val_mild_400g.csv}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
SAVE_TOP_N="${SAVE_TOP_N:-12}"
MAX_RUNS="${MAX_RUNS:-0}"
TOP_K="${TOP_K:-90}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[orig_like_grid] missing DATA_ROOT=${DATA_ROOT}"
  echo "[orig_like_grid] run first: OUT_TAG=zero_raw_v1 bash scripts/build_zero_raw_dataset.sh"
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[orig_like_grid] data_root=${DATA_ROOT}"
echo "[orig_like_grid] val_ids_csv=${VAL_IDS_CSV}"
echo "[orig_like_grid] feature_cache=${FEATURE_CACHE}"
echo "[orig_like_grid] output_dir=${OUT_DIR}"
echo "[orig_like_grid] top_k=${TOP_K}"
echo "[orig_like_grid] selection=val_topk_f1"

python scripts/grid_search_dinov2_probe_head_gpu_parallel.py \
  --loader torchhub \
  --hub-arch dinov2_vitb14_reg \
  --data-root "${DATA_ROOT}" \
  --val-ids-csv "${VAL_IDS_CSV}" \
  --top-k "${TOP_K}" \
  --output-dir "${OUT_DIR}" \
  --feature-cache "${FEATURE_CACHE}" \
  --feature-pool cls_patch_mean \
  --image-size 392 \
  --resize-size 420 \
  --batch-size 20 \
  --num-workers 8 \
  --test-hflip-tta \
  --head-seeds "3407 42 2024" \
  --positive-class-weights "1.00 1.05" \
  --feature-mixup-alphas "0.00 0.05 0.10 0.20" \
  --dropouts "0.35 0.45 0.55" \
  --label-smoothings "0.00 0.03 0.05" \
  --lrs "0.0005 0.001" \
  --weight-decays "0.01" \
  --epochs 70 \
  --patience 10 \
  --early-stop-metric topk_f1 \
  --target-pos-rate 0 \
  --parallel-workers "${PARALLEL_WORKERS}" \
  --save-top-n "${SAVE_TOP_N}" \
  --max-runs "${MAX_RUNS}"

python - <<PY
from pathlib import Path
import csv
import json
import shutil

out = Path('${OUT_DIR}')
grid = out / 'grid_results.csv'
best_dir = out / 'grid_best_result'
print('[orig_like_grid] grid_results=', grid)
if grid.exists():
    with grid.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    print('[orig_like_grid] top candidates:')
    for r in rows[:10]:
        print({k: r[k] for k in ['rank','tag','val_topk_f1','best_balanced_accuracy','best_epoch','val_topk_fp','val_topk_fn'] if k in r})
    if rows:
        best = rows[0]
        tag = best['tag']
        run_dir = out / 'runs' / tag
        top_path = run_dir / f'topk_${TOP_K}.csv'
        prob_path = run_dir / 'test_probabilities.csv'
        metrics_path = run_dir / 'metrics.json'
        best_dir.mkdir(parents=True, exist_ok=True)

        if top_path.exists():
            shutil.copy2(top_path, best_dir / f'topk_${TOP_K}.csv')
        if prob_path.exists():
            shutil.copy2(prob_path, best_dir / 'test_probabilities.csv')
        if metrics_path.exists():
            shutil.copy2(metrics_path, best_dir / 'metrics.json')

        with (best_dir / 'best_record.json').open('w', encoding='utf-8') as handle:
            json.dump(best, handle, ensure_ascii=False, indent=2, sort_keys=True)
        with (best_dir / 'README.txt').open('w', encoding='utf-8') as handle:
            handle.write('Grid best result selected by val_topk_f1.\n')
            handle.write(f'output_dir: {out}\n')
            handle.write(f'tag: {tag}\n')
            handle.write(f'top_k: ${TOP_K}\n')
            handle.write(f'submission: grid_best_result/topk_${TOP_K}.csv\n')
            handle.write('selection_metric: val_topk_f1\n')
            handle.write(f'val_topk_f1: {best.get("val_topk_f1", "")}\n')
            handle.write(f'best_balanced_accuracy: {best.get("best_balanced_accuracy", "")}\n')
            handle.write(f'best_epoch: {best.get("best_epoch", "")}\n')

        print('[orig_like_grid] rank1_submission=', top_path)
        print('[orig_like_grid] ensemble_rank_top=', out / 'ensemble_top' / f'topk_${TOP_K}_rank_top.csv')
        print('[orig_like_grid] grid_best_result=', best_dir)
PY
