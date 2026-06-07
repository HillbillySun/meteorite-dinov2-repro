<div align="center">

# 基于 DINOv2 的陨石图像分类

**STA326 课程项目复现仓库**

[English](README_en.md) · [Hugging Face Model](https://huggingface.co/Eki734/meteorite-dinov2-b14-direct) · [ModelScope Model](https://modelscope.cn/models/QiSunSiu/meteorite-dinov2-b14-direct)

</div>

---

## 项目简介

本仓库用于复现 STA326 课程项目中的陨石图像二分类结果。最终方案采用 **DINOv2-B/14 with registers** 作为特征提取器，并训练轻量 MLP probe head。最佳公开榜提交为 **top90**，Public Score 为 **0.79545**。

| 项目 | 内容 |
| --- | --- |
| 任务 | 陨石图像二分类 |
| 主模型 | DINOv2-B/14 with registers + MLP probe head |
| 线上结果记录 | `kaggle_online_results.csv` |
| Public Score | `0.79545` |
| 推荐显卡 | NVIDIA L20 或更高 |
| 推荐 Python | Python 3.10 |

## 复现路径

| 路径 | 说明 | 命令 |
| --- | --- | --- |
| 快速验证 | 从 Hugging Face 或 ModelScope 加载完整模型并生成 top90 | `python scripts/predict_hf_top90.py ...` |
| 固定 cache 复现 | 使用仓库内固定 feature cache 快速复现训练输出 | `bash scripts/reproduce_best.sh` |
| 完整 pipeline | 从原始数据构建数据集，并通过网格搜索自动选择最佳模型 | `bash scripts/build_zero_raw_dataset.sh` + `bash scripts/grid_search_original_like_topkf1.sh` |

## 核心仓库结构

```text
.
├── artifacts/
│   └── features_cache_dinov2_b14_relaxed_mild.pt
│                                      # 固定特征 cache，用于快速复现
├── configs/
│   └── best_dinov2_b14_relaxed.yaml   # 实验配置记录
├── data/
│   ├── README.md                      # 数据放置说明
│   ├── train_images/                  # 原始训练图片，需自行放置
│   ├── test_images/                   # 原始测试图片，需自行放置
│   ├── train_labels.csv               # 训练标签
│   ├── sample_submission.csv          # 提交模板
│   └── splits/hardval_ablation/
│       └── val_mild_400g.csv          # 固定验证集划分
├── scripts/
│   ├── build_stage2_rembg_datasets.py # rembg 训练集预处理
│   ├── build_stage2_style_v3_dataset.py
│   ├── build_zero_raw_dataset.sh      # 从原始数据构建完整训练数据
│   ├── grid_search_dinov2_probe_head_gpu_parallel.py
│   ├── grid_search_original_like_topkf1.sh
│   │                                  # 网格搜索并按 val topk_f1 选择 rank1
│   ├── predict_hf_top90.py            # HF、ModelScope 和本地模型推理
│   ├── reproduce_best.sh              # 使用固定 cache 快速复现
│   ├── train_dinov2_probe.py          # DINOv2 frozen probe 训练
│   └── train_best_known_hparams_checkpoint.py
├── kaggle_online_results.csv          # 线上提交记录，用于结果比对
├── requirements.txt                   # Python 依赖
├── setup_gpu_env.sh                   # 配置 pip CUDA/cuDNN 动态库路径
├── README.md                          # 中文说明
└── README_en.md                       # English documentation
```

## 数据准备

请将课程提供的数据放到以下位置：

```text
data/train_images/*.jpg
data/test_images/*.jpg
data/train_labels.csv
data/sample_submission.csv
```

说明：`data/test_images` 直接使用原始测试图片，不需要额外抠图或风格化处理。

## 环境配置

建议使用 Python 3.10 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

依赖安装完成后，加载 GPU 运行库环境：

```bash
source setup_gpu_env.sh
```

> 必须使用 `source`，不要使用 `bash setup_gpu_env.sh`。脚本需要把 pip 安装的 CUDA/cuDNN 动态库路径保留在当前终端，供后续 `onnxruntime-gpu` 和 rembg 数据构建使用。每次新开终端并激活 `.venv` 后，如需构建数据，请重新执行一次该命令。

脚本会同时检查 PyTorch CUDA 和 ONNX Runtime provider。正常情况下应看到：

```text
torch cuda available: True
onnxruntime providers: ['CUDAExecutionProvider', ...]
```

## 路径一：Hugging Face / ModelScope 快速验证

同一个推理脚本支持 Hugging Face、ModelScope 和本地模型目录。三种来源下载完成后都会进入同一套 Transformers 推理流程。

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

也可以直接从 ModelScope 加载同一份完整模型：

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

主要输出：

```text
runs/hf_direct_top90/topk_90.csv
runs/hf_direct_top90/test_probabilities.csv
runs/modelscope_direct_top90/topk_90.csv
runs/modelscope_direct_top90/test_probabilities.csv
```

如果生成结果与线上提交记录一致，终端会显示：

```text
[top90] diff_vs_reference=0
[top90] matches reference exactly
```

## 路径二：固定 feature cache 快速复现

如果只想快速复现最终训练输出，而不重新提取 DINOv2 特征，可以运行：

```bash
bash scripts/reproduce_best.sh
```

该脚本会在训练结束后自动与 `kaggle_online_results.csv` 比较，并打印差异数量。主要输出为：

```text
runs/reproduce_dinov2_b14_best_exact/runs/
└── s3407_pw1p00_mx0p03_do0p45_ls0p02_lr0p0005_wd0p01/
    └── topk_90.csv
```

若完全一致，终端会显示：

```text
[reproduce_best] OK: ... matches kaggle_online_results.csv
```

## 路径三：完整 Pipeline 复现

如果希望从课程原始数据完整复现，请先构建训练用数据集：

```bash
bash scripts/build_zero_raw_dataset.sh
```

然后运行原始风格的网格搜索。该脚本会在候选超参数中训练多个 probe head，并按照验证集 `topk_f1` 自动选择 rank1 作为最终模型：

```bash
bash scripts/grid_search_original_like_topkf1.sh
```

该路径会重新构建训练数据、提取 DINOv2 特征、执行网格搜索、训练 MLP probe head，并输出 top90 结果。由于包含特征提取和多组超参数搜索，耗时会明显长于固定 cache 路径。

网格搜索完成后，重点查看：

```text
runs/grid_original_like_topkf1_zero_raw_v1/grid_results.csv
runs/grid_original_like_topkf1_zero_raw_v1/grid_best_result/topk_90.csv
```

其中 `grid_best_result` 是脚本自动整理出的 rank1 结果目录，方便直接检查最终提交文件。

网格完成后，可以直接与线上提交记录比较：

```bash
cmp \
  runs/grid_original_like_topkf1_zero_raw_v1/grid_best_result/topk_90.csv \
  kaggle_online_results.csv \
  && echo "OK: grid best result matches Kaggle online result"
```

命令输出 `OK` 时，说明网格自动选择的 rank1 top90 与 `kaggle_online_results.csv` 完全一致。

## 输出与比对

主要文件：

```text
outputs/best_submission/topk_90.csv
outputs/best_submission/test_probabilities.csv
kaggle_online_results.csv
```

`kaggle_online_results.csv` 是线上提交记录。若需要检查复现是否一致，请比较复现生成的 `topk_90.csv` 与 `kaggle_online_results.csv`。
