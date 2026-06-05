# 基于DINOv2的陨石图像分类

本仓库用于复现 STA326 课程项目的最佳提交结果。

- Public Score: 0.79545
- 线上提交记录: kaggle_online_results.csv
- 模型: DINOv2-B with registers + MLP probe head

本仓库提供两条复现路径：

- 快速验证：使用仓库提供的固定 DINOv2 feature cache，直接复现最佳提交。

- 完整 Pipeline：从原始图片构造数据，按原仓库网格搜索流程重新训练，按 val topk_f1 选择 rank1。
## 1. 仓库结构

```text
artifacts/
  features_cache_dinov2_b14_relaxed_mild.pt

configs/
  best_dinov2_b14_relaxed.yaml

data/
  README.md
  splits/hardval_ablation/val_mild_400g.csv

outputs/best_submission/
  topk_90.csv
  test_probabilities.csv

kaggle_online_results.csv

scripts/
  reproduce_best.sh
  grid_search_original_like_topkf1.sh
  grid_search_dinov2_probe_head_gpu_parallel.py
  train_dinov2_probe.py
  build_stage2_rembg_datasets.py
  build_stage2_style_v3_dataset.py

requirements.txt
setup_gpu_env.sh
```

## 2. 硬件环境

本结果在 NVIDIA L20 GPU 上完成和验证。完整网格搜索建议使用同等级或更高显存的 GPU；如果显存较小，可以降低 `PARALLEL_WORKERS` 或改为单进程运行。

## 3. 创建环境

推荐使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

如果 PyTorch CUDA wheel 安装失败，请先按机器 CUDA 版本从 PyTorch 官网安装 `torch` 和 `torchvision`，再运行：

```bash
pip install -r requirements.txt
```

## 4. 配置 GPU 运行库

每次新开 shell 后运行：

```bash
source .venv/bin/activate
source setup_gpu_env.sh
```

正常情况下会看到：

```text
torch cuda available: True
onnxruntime providers: ['CUDAExecutionProvider', ...]
```

如果出现下面错误，通常是没有运行 `source setup_gpu_env.sh`：

```text
libcudnn.so.9: cannot open shared object file
```

## 5. 快速验证：固定 cache 复现最佳提交

该路径不需要原始图片，也不需要先构建数据。本仓库包含固定的 DINOv2 feature cache：

```text
artifacts/features_cache_dinov2_b14_relaxed_mild.pt
```

如果只需要快速验证最终提交文件，可以直接运行：

```bash
bash scripts/reproduce_best.sh
```

输出文件：

```text
runs/reproduce_dinov2_b14_best_exact/runs/s3407_pw1p00_mx0p03_do0p45_ls0p02_lr0p0005_wd0p01/topk_90.csv
```

检查是否与线上公开榜提交记录完全一致：

```bash
cmp \
  runs/reproduce_dinov2_b14_best_exact/runs/s3407_pw1p00_mx0p03_do0p45_ls0p02_lr0p0005_wd0p01/topk_90.csv \
  kaggle_online_results.csv
```

如果没有输出，说明复现结果与线上提交文件完全一致。`outputs/best_submission/topk_90.csv` 是仓库中保存的一份同样内容的复现输出副本。

## 6. 准备原始数据

请将课程提供的数据放到 `data/` 下：

```text
data/
├── train_images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── test_images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── train_labels.csv
└── sample_submission.csv
```

要求：

```text
train_labels.csv 和 sample_submission.csv 中的 id 必须与图片文件名一致。
```

## 7. 构造训练数据

助教从原始数据开始复现时，需要先构造本项目使用的 style-v3 relaxed 数据。

运行：

```bash
OUT_TAG=zero_raw_v1 bash scripts/build_zero_raw_dataset.sh
```

该脚本会执行：

```text
原始 train_images -> rembg 抠图 -> style-v3 relaxed 数据
```

主要生成：

```text
data/zero_raw_v1/stage2_style_v3_relaxed_dino_root/
```

该目录会作为后续完整网格搜索的数据根目录。


## 8. 完整 Pipeline：网格搜索复现

数据构造完成后，按原仓库网格搜索流程重新训练 probe head：

```bash
PARALLEL_WORKERS=4 \
OUT_DIR=runs/grid_original_like_topkf1_zero_raw_v1 \
bash scripts/grid_search_original_like_topkf1.sh
```

该脚本使用 `topk_f1` 做早停和模型选择。完整网格如下：

```text
head_seeds = 3407 42 2024
positive_class_weights = 1.00 1.05
feature_mixup_alphas = 0.00 0.05 0.10 0.20
dropouts = 0.35 0.45 0.55
label_smoothings = 0.00 0.03 0.05
lrs = 0.0005 0.001
weight_decays = 0.01
epochs = 70
patience = 10
early_stop_metric = topk_f1
test_hflip_tta = on
top_k = 90
```

网格结束后，脚本会自动整理 rank1 到：

```text
runs/grid_original_like_topkf1_zero_raw_v1/grid_best_result/
├── README.txt
├── best_record.json
├── metrics.json
├── test_probabilities.csv
└── topk_90.csv
```

其中 `grid_best_result/topk_90.csv` 是按 `val_topk_f1` 选择出的 rank1 submission。

检查是否与线上公开榜提交记录完全一致：

```bash
cmp \
  runs/grid_original_like_topkf1_zero_raw_v1/grid_best_result/topk_90.csv \
  kaggle_online_results.csv
```

如果没有输出，说明完整网格选出的 rank1 与线上提交文件完全一致。

## 9. 可选：只重新构建预处理数据

如果只希望从原始图片重新生成训练数据，可以先运行 rembg：

```bash
python scripts/build_stage2_rembg_datasets.py
```

然后生成 style-v3 relaxed 数据：

```bash
python scripts/build_stage2_style_v3_dataset.py \
  --output-dir data/train_images_stage2_style_v3_relaxed \
  --output-labels data/train_labels_stage2_style_v3_relaxed.csv \
  --metadata-csv data/train_images_stage2_style_v3_relaxed_meta.csv \
  --min-source-area 0.003 \
  --max-source-area 0.90 \
  --min-target-area 0.20 \
  --max-target-area 0.96 \
  --target-area-multiplier 1.08 \
  --background-jitter-prob 0.35 \
  --shadow-prob 0.25 \
  --output-root data/stage2_style_v3_relaxed_dino_root
```

该步骤会生成：

```text
data/stage2_style_v3_relaxed_dino_root/
├── train_images/
├── test_images/
├── train_labels.csv
└── sample_submission.csv
```

## 10. 可选：手动测试 DINOv2 下载

如果不使用固定 feature cache，而是重新提取 DINOv2 特征，可以先测试 PyTorch Hub：

```bash
python - <<'PY'
import torch
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
print("DINOv2-B with registers loaded")
PY
```

