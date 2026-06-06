<div align="center">

# Meteorite Image Classification with DINOv2

**Reproduction repository for the STA326 course project**

[中文](README.md) · [Hugging Face Model](https://huggingface.co/Eki734/meteorite-dinov2-b14-direct) · [ModelScope Model](https://modelscope.cn/models/QiSunSiu/meteorite-dinov2-b14-direct)

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
| Quick verification | Load the full model from Hugging Face or ModelScope and generate top90 | `python scripts/predict_hf_top90.py ...` |
| Fixed-cache reproduction | Reproduce the training output using the provided feature cache | `bash scripts/reproduce_best.sh` |
| Full pipeline | Build the dataset from raw images and select the best model through grid search | `bash scripts/build_zero_raw_dataset.sh` + `bash scripts/grid_search_original_like_topkf1.sh` |

## Core Repository Layout

```text
.
├── artifacts/
│   └── features_cache_dinov2_b14_relaxed_mild.pt
│                                      # Fixed feature cache for fast reproduction
├── configs/
│   └── best_dinov2_b14_relaxed.yaml   # Experiment configuration record
├── data/
│   ├── README.md                      # Data placement instructions
│   ├── train_images/                  # Place original training images here
│   ├── test_images/                   # Place original test images here
│   ├── train_labels.csv               # Training labels
│   ├── sample_submission.csv          # Submission template
│   └── splits/hardval_ablation/
│       └── val_mild_400g.csv          # Fixed validation split
├── scripts/
│   ├── build_stage2_rembg_datasets.py # rembg training-data preprocessing
│   ├── build_stage2_style_v3_dataset.py
│   ├── build_zero_raw_dataset.sh      # Build the full dataset from raw images
│   ├── grid_search_dinov2_probe_head_gpu_parallel.py
│   ├── grid_search_original_like_topkf1.sh
│   │                                  # Grid search selected by validation topk_f1
│   ├── predict_hf_top90.py            # HF, ModelScope, and local inference
│   ├── reproduce_best.sh              # Fast reproduction with the fixed cache
│   ├── train_dinov2_probe.py          # DINOv2 frozen-probe training
│   └── train_best_known_hparams_checkpoint.py
├── kaggle_online_results.csv          # Online submission record for comparison
├── requirements.txt                   # Python dependencies
├── setup_gpu_env.sh                   # Configure pip CUDA/cuDNN library paths
├── README.md                          # Chinese documentation
└── README_en.md                       # English documentation
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

Create a Python 3.10 virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

After installation, load the GPU runtime environment:

```bash
source setup_gpu_env.sh
```

> Use `source`, not `bash setup_gpu_env.sh`. The script must keep the pip-installed CUDA/cuDNN library paths in the current shell for `onnxruntime-gpu` and the rembg dataset-building step. Run it again after activating `.venv` in each new terminal when rebuilding the dataset.

The script checks both PyTorch CUDA and the ONNX Runtime providers. A working GPU setup should print output similar to:

```text
torch cuda available: True
onnxruntime providers: ['CUDAExecutionProvider', ...]
```

## Option 1: Hugging Face / ModelScope Quick Verification

The same inference script supports Hugging Face, ModelScope, and local model directories. After resolution, all three sources use the same Transformers inference path.

### Hugging Face

```bash
python scripts/predict_hf_top90.py \
  --model-source huggingface \
  --model-id Eki734/meteorite-dinov2-b14-direct \
  --test-dir data/test_images \
  --output-csv runs/hf_direct_top90/topk_90.csv \
  --prob-csv runs/hf_direct_top90/test_probabilities.csv \
  --reference-csv kaggle_online_results.csv \
  --top-k 90
```

### ModelScope

The same complete model can also be loaded directly from ModelScope:

```bash
python scripts/predict_hf_top90.py \
  --model-source modelscope \
  --model-id QiSunSiu/meteorite-dinov2-b14-direct \
  --test-dir data/test_images \
  --output-csv runs/modelscope_direct_top90/topk_90.csv \
  --prob-csv runs/modelscope_direct_top90/test_probabilities.csv \
  --reference-csv kaggle_online_results.csv \
  --top-k 90
```

Main outputs:

```text
runs/hf_direct_top90/topk_90.csv
runs/hf_direct_top90/test_probabilities.csv
runs/modelscope_direct_top90/topk_90.csv
runs/modelscope_direct_top90/test_probabilities.csv
```

An exact match against the online submission record produces:

```text
[top90] diff_vs_reference=0
[top90] matches reference exactly
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

After the grid finishes, compare the selected rank1 submission directly against the online record:

```bash
cmp \
  runs/grid_original_like_topkf1_zero_raw_v1/grid_best_result/topk_90.csv \
  kaggle_online_results.csv \
  && echo "OK: grid best result matches Kaggle online result"
```

An `OK` message means that the automatically selected rank1 top90 exactly matches `kaggle_online_results.csv`.

## Outputs and Comparison

Main files:

```text
outputs/best_submission/topk_90.csv
outputs/best_submission/test_probabilities.csv
kaggle_online_results.csv
```

`kaggle_online_results.csv` stores the online submission record. To verify reproduction, compare the generated `topk_90.csv` with `kaggle_online_results.csv`.
