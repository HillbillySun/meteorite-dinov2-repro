# 基于DINOv2的陨石图像分类

本仓库用于复现 STA326 课程项目的最佳提交结果。

```text
Public Score: 0.79545
线上提交记录: kaggle_online_results.csv
模型: DINOv2-B/14 with registers + MLP probe head
主要显卡: NVIDIA L20
```

本仓库提供三种使用方式：

1. 快速验证：使用仓库内固定 DINOv2 feature cache，直接复现最佳 `topk_90.csv`。
2. Hugging Face 直调：直接加载已上传的完整模型做单图推理，并生成 `topk_90.csv`。
3. 完整 Pipeline：从课程原始图片开始构造数据、提特征、网格搜索并选择最佳提交。

## 1. 仓库结构

当前 Git 仓库中主要包含：

```text
artifacts/
  features_cache_dinov2_b14_relaxed_mild.pt   # 固定 feature cache，用于快速复现

configs/
  best_dinov2_b14_relaxed.yaml                # 最佳配置记录

data/
  README.md
  train_labels.csv                            # 官方标签表
  sample_submission.csv                       # 官方提交模板
  splits/hardval_ablation/val_mild_400g.csv   # 固定 validation split

outputs/best_submission/
  topk_90.csv                                 # 最佳提交文件副本
  test_probabilities.csv                      # 最佳模型测试概率副本

scripts/
  reproduce_best.sh                           # 固定 cache 快速复现
  predict_hf_top90.py                         # Hugging Face 直调生成 top90
  build_zero_raw_dataset.sh                   # 从原始图构造训练数据
  grid_search_original_like_topkf1.sh         # 完整网格搜索入口
  grid_search_dinov2_probe_head_gpu_parallel.py
  train_dinov2_probe.py
  build_stage2_rembg_datasets.py
  build_stage2_style_v3_dataset.py

kaggle_online_results.csv                     # 线上公开榜对应提交文件
requirements.txt
setup_gpu_env.sh
```

## 2. 环境配置

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

每次新开 shell 后建议运行：

```bash
source .venv/bin/activate
source setup_gpu_env.sh
```

正常情况下会看到：

```text
torch cuda available: True
onnxruntime providers: ['CUDAExecutionProvider', ...]
```

如果出现下面错误，通常是没有运行 `source setup_gpu_env.sh`，或 CUDA/cuDNN 运行库没有被正确找到：

```text
libcudnn.so.9: cannot open shared object file
```
## 3. 快速验证
### 3.1 固定 cache 复现最佳提交

该路径不需要原始图片，也不需要先构建数据。本仓库包含固定的 DINOv2 feature cache：

```text
artifacts/features_cache_dinov2_b14_relaxed_mild.pt
```

运行：

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

如果没有输出，说明复现结果与线上提交文件完全一致。

### 3.2 Hugging Face 直调模型验证

完整模型已上传到 Hugging Face，仓库地址：

```text
https://huggingface.co/Eki734/meteorite-dinov2-b14-direct
```

该模型包含完整 DINOv2-B/14-register backbone 和 MLP probe head 权重。

如果希望直接生成 Hugging Face 模型对应的 `topk_90.csv`，运行：

```bash
python scripts/predict_hf_top90.py
```

默认输出：

```text
runs/hf_direct_top90/test_probabilities.csv
runs/hf_direct_top90/topk_90.csv
```

脚本会自动与 `kaggle_online_results.csv` 比对。预期最后会看到：

```text
[hf_top90] diff_vs_reference=0
[hf_top90] matches reference exactly
```

这说明生成的 `topk_90.csv` 与线上提交文件完全一致。

## 4. 完整pipeline复现
### 4.1 准备原始数据

如需从零开始运行完整 Pipeline，请将课程提供的数据放到 `data/` 下：

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

其中 `train_images/` 和 `test_images/` 已写入 `.gitignore`，不会被提交到 Git。

### 4.2 构造训练数据

若要完整复现 pipeline，须从原始数据开始构造本项目使用的 style-v3 relaxed 数据。

运行：

```bash
OUT_TAG=zero_raw_v1 bash scripts/build_zero_raw_dataset.sh
```

该脚本会执行：

```text
原始 train_images -> rembg 抠图 -> style-v3 relaxed 训练数据
原始 test_images  -> 直接复制到构造后的 test_images
```

主要生成：

```text
data/zero_raw_v1/stage2_style_v3_relaxed_dino_root/
├── train_images/
├── test_images/
├── train_labels.csv
└── sample_submission.csv
```

`data/zero_raw_v1/` 是生成目录，已写入 `.gitignore`。

### 4.3 网格搜索

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

## 5. 可选：测试 DINOv2 下载

如果不使用固定 feature cache，而是重新提取 DINOv2 特征，可以先测试 PyTorch Hub：

```bash
python - <<'PY'
import torch
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
print("DINOv2-B with registers loaded")
PY
```
