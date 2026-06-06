<div align="center">

# 基于 DINOv2 的陨石图像分类

**STA326 课程项目复现仓库**

[English](README_en.md) · [Hugging Face Model](https://huggingface.co/Eki734/meteorite-dinov2-b14-direct)

</div>

---

## 项目简介

本仓库用于复现 STA326 课程项目中的陨石图像二分类结果。最终方案采用 **DINOv2-B/14 with registers** 作为特征提取器，并训练轻量 MLP probe head。最佳公开榜提交为 **top90**，Public Score 为 **0.79545**。

| 项目 | 内容 |
| --- | --- |
| 任务 | 陨石图像二分类 |
| 主模型 | DINOv2-B/14 with registers + MLP probe head |
| 最佳提交 | `outputs/best_submission/topk_90.csv` |
| 线上结果记录 | `kaggle_online_results.csv` |
| Public Score | `0.79545` |
| 推荐显卡 | NVIDIA L20 或更高 |
| 推荐 Python | Python 3.10 |

## 复现路径

| 路径 | 说明 | 命令 |
| --- | --- | --- |
| 快速验证 | 直接调用 Hugging Face 上的完整模型生成 top90 | `python scripts/predict_hf_top90.py ...` |
| 固定 cache 复现 | 使用仓库内固定 feature cache 快速复现训练输出 | `bash scripts/reproduce_best.sh` |
| 完整 pipeline | 从原始数据构建数据集，并通过网格搜索自动选择最佳模型 | `bash scripts/build_zero_raw_dataset.sh` + `bash scripts/grid_search_original_like_topkf1.sh` |

## 仓库结构

```text
.
├── data/
│   ├── train_images/                 # 原始训练图片，请自行放置
│   ├── test_images/                  # 原始测试图片，请自行放置
│   ├── train_labels.csv              # 训练标签
│   ├── sample_submission.csv         # 提交模板
│   └── splits/                       # 固定验证集划分
├── scripts/
│   ├── build_zero_raw_dataset.sh     # 从原始数据构建训练用数据集
│   ├── grid_search_original_like_topkf1.sh
│   │                                  # 原始风格网格搜索，按 val topk_f1 选择最佳模型
│   ├── reproduce_best.sh             # 使用固定 feature cache 快速复现
│   ├── predict_hf_top90.py           # 直接调用 Hugging Face 模型生成 top90
│   ├── train_best_known_hparams.sh   # 可选：单组超参数复训
│   └── train_best_known_hparams_checkpoint.py
├── outputs/
│   └── best_submission/              # 最佳提交与概率文件
├── artifacts/                        # 固定 feature cache
├── docs/                             # 实验记录与说明
├── requirements.txt
├── README.md
└── README_en.md
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

建议使用 Python 3.10 创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

检查 CUDA 是否可用：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## 路径一：Hugging Face 快速验证

这是最省事的验证方式。脚本会从 Hugging Face 加载已经打包好的完整模型，并直接对 `data/test_images` 生成 top90 提交。

```bash
python scripts/predict_hf_top90.py \
  --model-id Eki734/meteorite-dinov2-b14-direct \
  --test-dir data/test_images \
  --output-csv runs/hf_direct_top90/topk_90.csv \
  --prob-csv runs/hf_direct_top90/test_probabilities.csv \
  --reference-csv kaggle_online_results.csv \
  --top-k 90
```

输出文件：

```text
runs/hf_direct_top90/test_probabilities.csv
runs/hf_direct_top90/topk_90.csv
```

如果本地数据与线上提交记录一致，终端会显示：

```text
[hf_top90] diff_vs_reference=0
[hf_top90] matches reference exactly
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

网格完成后，运行下面的命令，将自动选出的 rank1 提交与线上记录逐样本比较：

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

当输出 `diff_count: 0` 时，说明该网格 rank1 的 top90 与 `kaggle_online_results.csv` 完全一致。

## 输出与比对

主要文件：

```text
outputs/best_submission/topk_90.csv
outputs/best_submission/test_probabilities.csv
kaggle_online_results.csv
```

`kaggle_online_results.csv` 是线上提交记录。若需要检查复现是否一致，请比较复现生成的 `topk_90.csv` 与 `kaggle_online_results.csv`。
