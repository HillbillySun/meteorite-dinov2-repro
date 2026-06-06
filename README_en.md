<div align="center">

# Meteorite Image Classification with DINOv2

**Reproduction repository for the STA326 course project**

[中文](README.md) · [Hugging Face Model](https://huggingface.co/Eki734/meteorite-dinov2-b14-direct)

</div>

---

## Overview

This repository reproduces the meteorite image binary classification result for the STA326 course project. The final solution uses **DINOv2-B/14 with registers** as the feature extractor and trains a lightweight MLP probe head. The best public submission is a **top90** file with a Public Score of **0.79545**.

| Item | Value |
| --- | --- |
| Task | Binary meteorite image classification |
| Main model | DINOv2-B/14 with registers + MLP probe head |
| Best submission | `outputs/best_submission/topk_90.csv` |
| Online result record | `kaggle_online_results.csv` |
| Public Score | `0.79545` |
| Recommended GPU | NVIDIA L20 or better |
| Recommended Python | Python 3.10 |

## Reproduction Options

| Option | Description | Command |
| --- | --- | --- |
| Quick verification | Generate top90 directly with the full Hugging Face model | `python scripts/predict_hf_top90.py ...` |
| Fixed-cache reproduction | Reproduce the training output using the provided feature cache | `bash scripts/reproduce_best.sh` |
| Full pipeline | Build the dataset from raw images and select the best model through grid search | `bash scripts/build_zero_raw_dataset.sh` + `bash scripts/grid_search_original_like_topkf1.sh` |

## Repository Layout

```text
.
├── data/
│   ├── train_images/                 # Place original training images here
│   ├── test_images/                  # Place original test images here
│   ├── train_labels.csv              # Training labels
│   ├── sample_submission.csv         # Submission template
│   └── splits/                       # Fixed validation splits
├── scripts/
│   ├── build_zero_raw_dataset.sh     # Build the training dataset from raw images
│   ├── grid_search_original_like_topkf1.sh
│   │                                  # Original-style grid search selected by val topk_f1
│   ├── reproduce_best.sh             # Fast reproduction with fixed feature cache
│   ├── predict_hf_top90.py           # Generate top90 using the Hugging Face model
│   ├── train_best_known_hparams.sh   # Optional single-setting retraining
│   └── train_best_known_hparams_checkpoint.py
├── outputs/
│   └── best_submission/              # Best submission and probability files
├── artifacts/                        # Fixed feature cache
├── docs/                             # Experiment notes and documentation
├── requirements.txt
├── README.md
└── README_en.md
```

## Data Preparation

Place the course data as follows:

```text
data/train_images/*.jpg
data/test_images/*.jpg
data/train_labels.csv
data/sample_submission.csv
```

Note: `data/test_images` should contain the original test images. No additional background removal or style conversion is required for the test set.

## Environment Setup

Create a Python 3.10 virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check CUDA availability:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Option 1: Hugging Face Quick Verification

This is the simplest verification path. The script loads the packaged model from Hugging Face and generates the top90 submission directly from `data/test_images`.

```bash
python scripts/predict_hf_top90.py \
  --model-id Eki734/meteorite-dinov2-b14-direct \
  --test-dir data/test_images \
  --output-csv runs/hf_direct_top90/topk_90.csv \
  --prob-csv runs/hf_direct_top90/test_probabilities.csv \
  --reference-csv kaggle_online_results.csv \
  --top-k 90
```

Output files:

```text
runs/hf_direct_top90/test_probabilities.csv
runs/hf_direct_top90/topk_90.csv
```

If the local data matches the online submission record, the terminal should show:

```text
[hf_top90] diff_vs_reference=0
[hf_top90] matches reference exactly
```

## Option 2: Fast Reproduction with Fixed Feature Cache

To reproduce the final training output without extracting DINOv2 features again, run:

```bash
bash scripts/reproduce_best.sh
```

The script automatically compares its output against `kaggle_online_results.csv` and prints the number of differences. Its main output is:

```text
runs/reproduce_dinov2_b14_best_exact/runs/
└── s3407_pw1p00_mx0p03_do0p45_ls0p02_lr0p0005_wd0p01/
    └── topk_90.csv
```

An exact match produces:

```text
[reproduce_best] OK: ... matches kaggle_online_results.csv
```

## Option 3: Full Pipeline Reproduction

To reproduce the full workflow from raw course data, first build the training dataset:

```bash
bash scripts/build_zero_raw_dataset.sh
```

Then run the original-style grid search. The script trains multiple probe heads over candidate hyperparameters and automatically selects the rank1 model by validation `topk_f1`:

```bash
bash scripts/grid_search_original_like_topkf1.sh
```

This path rebuilds the training data, extracts DINOv2 features, performs grid search, trains the MLP probe heads, and writes the top90 result. It is slower than the fixed-cache path because both feature extraction and multiple hyperparameter trials are performed.

After the grid search finishes, check:

```text
runs/grid_original_like_topkf1_zero_raw_v1/grid_results.csv
runs/grid_original_like_topkf1_zero_raw_v1/grid_best_result/topk_90.csv
```

`grid_best_result` is the automatically collected rank1 result directory and contains the final submission file.

After the grid finishes, compare the automatically selected rank1 submission against the online record sample by sample:

```bash
python - <<'PY'
import csv
from pathlib import Path

pred = Path("runs/grid_original_like_topkf1_zero_raw_v1/grid_best_result/topk_90.csv")
ref = Path("kaggle_online_results.csv")

def read_labels(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    label_col = next(c for c in rows[0] if c != "id")
    return {row["id"]: str(row[label_col]) for row in rows}

pred_labels = read_labels(pred)
ref_labels = read_labels(ref)
all_ids = sorted(set(pred_labels) | set(ref_labels))
diff = [(i, pred_labels.get(i), ref_labels.get(i))
        for i in all_ids if pred_labels.get(i) != ref_labels.get(i)]

print("prediction:", pred)
print("reference:", ref)
print("diff_count:", len(diff))
print("diff_preview:", diff[:20])
raise SystemExit(1 if diff else 0)
PY
```

`diff_count: 0` means that the grid rank1 top90 exactly matches `kaggle_online_results.csv`.

## Outputs and Comparison

Main files:

```text
outputs/best_submission/topk_90.csv
outputs/best_submission/test_probabilities.csv
kaggle_online_results.csv
```

`kaggle_online_results.csv` stores the online submission record. To verify reproduction, compare the generated `topk_90.csv` with `kaggle_online_results.csv`.
