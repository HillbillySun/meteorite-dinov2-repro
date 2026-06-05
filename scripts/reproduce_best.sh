#!/usr/bin/env bash
set -euo pipefail

python scripts/grid_search_dinov2_probe_head_gpu_parallel.py \
  --loader torchhub \
  --hub-arch dinov2_vitb14_reg \
  --data-root data/stage2_style_v3_relaxed_dino_root \
  --val-ids-csv data/splits/hardval_ablation/val_mild_400g.csv \
  --top-k 90 \
  --output-dir runs/reproduce_dinov2_b14_best_exact \
  --feature-cache artifacts/features_cache_dinov2_b14_relaxed_mild.pt \
  --feature-pool cls_patch_mean \
  --image-size 392 \
  --resize-size 420 \
  --batch-size 20 \
  --num-workers 8 \
  --head-seeds "3407" \
  --positive-class-weights "1.00" \
  --feature-mixup-alphas "0.03" \
  --dropouts "0.45" \
  --label-smoothings "0.02" \
  --lrs "0.0005" \
  --weight-decays "0.01" \
  --epochs 60 \
  --patience 10 \
  --early-stop-metric topk_f1 \
  --parallel-workers 1 \
  --save-top-n 1

PRED="runs/reproduce_dinov2_b14_best_exact/runs/s3407_pw1p00_mx0p03_do0p45_ls0p02_lr0p0005_wd0p01/topk_90.csv"
REF="kaggle_online_results.csv"

if [[ -f "${REF}" ]]; then
  if cmp -s "${PRED}" "${REF}"; then
    echo "[reproduce_best] OK: ${PRED} matches ${REF}"
  else
    echo "[reproduce_best] WARNING: ${PRED} differs from ${REF}"
    python - <<'PY'
import csv
from pathlib import Path
pred = Path('runs/reproduce_dinov2_b14_best_exact/runs/s3407_pw1p00_mx0p03_do0p45_ls0p02_lr0p0005_wd0p01/topk_90.csv')
ref = Path('kaggle_online_results.csv')
def read_labels(p):
    rows = list(csv.DictReader(p.open('r', encoding='utf-8-sig', newline='')))
    label_col = [c for c in rows[0].keys() if c != 'id'][0]
    return {r['id']: str(r[label_col]) for r in rows}
a = read_labels(pred)
b = read_labels(ref)
diff = [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
print('diff_count=', len(diff))
print('diff_preview=', diff[:20])
PY
  fi
else
  echo "[reproduce_best] reference ${REF} not found; skip comparison"
fi
