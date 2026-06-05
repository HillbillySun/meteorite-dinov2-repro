# Data Layout

Raw images are not included in this repository.

## Official Raw Dataset

Place the official course dataset as:

```text
data/
├── train_images/
├── test_images/
├── train_labels.csv
└── sample_submission.csv
```

`train_images/` and `test_images/` should contain image files whose names match the `id` column in the CSV files.

## Required Preprocessing

The teaching assistant does not need to prepare processed images manually. Starting from the raw dataset above, run:

```bash
OUT_TAG=zero_raw_v1 bash scripts/build_zero_raw_dataset.sh
```

This creates:

```text
data/zero_raw_v1/train_images_stage2_rembg_all/
data/zero_raw_v1/train_images_stage2_style_v3_relaxed/
data/zero_raw_v1/stage2_style_v3_relaxed_dino_root/
```

The full grid-search reproduction uses:

```text
data/zero_raw_v1/stage2_style_v3_relaxed_dino_root/
```

## Validation Split

The fixed validation split used for grid selection is included at:

```text
data/splits/hardval_ablation/val_mild_400g.csv
```

## Preprocessing Scripts

The preprocessing scripts are:

```text
scripts/build_stage2_rembg_datasets.py
scripts/build_stage2_style_v3_dataset.py
```

`build_zero_raw_dataset.sh` calls both scripts with the parameters used by the final pipeline.
