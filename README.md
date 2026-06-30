# MSPF-Net

MSPF-Net is a thesis codebase for robust fault diagnosis on high-speed train running gear datasets. It contains the cleaned pipeline used to build catalogs, assign leakage-safe file-level splits, preprocess multichannel vibration windows, train baseline models, evaluate MSPF-Net variants, and reproduce the final thesis result tables and figures.

The active thesis benchmark datasets are:

- D1
- D2
- D4
- D5
- D8

D3 is excluded from the thesis benchmark. D6 and D7 remain supported by parts of the codebase but are not part of the default evaluation flow.

## What Is Included

```text
MSPF-Net/
├── configs/                 # Global and model-specific YAML configs
├── data/
│   ├── metadata/            # Dataset schema and label mapping
│   └── interim/             # Catalog CSVs used by the pipeline
├── embedded/                # Raspberry Pi 5 export and benchmark tools
├── outputs/                 # Dataset metadata summaries
├── reports/                 # EDA report summaries and figures
├── results/
│   ├── figures/             # Final thesis figures
│   └── tables/              # Final thesis result tables
├── scripts/                 # Reproduction entry points
├── src/mspf_net/            # Package source code
├── environment.yml          # Conda environment
└── pyproject.toml           # Editable package metadata
```

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate mspf_net
pip install -e .
```

If the environment already exists:

```bash
conda activate mspf_net
pip install -e .
```

## Data

The dataset files are available from Google Drive:

[MSPF-Net data folder](https://drive.google.com/drive/folders/1QaCOWctCfAHH4ixc7Dq2EWfnnWCmI_Xo?usp=sharing)

Download the data and place the raw datasets under:

```text
data/raw/dataset_1/
data/raw/dataset_2/
data/raw/dataset_4/
data/raw/dataset_5/
data/raw/dataset_8/
```

The repository includes lightweight metadata, catalogs, final tables, and final figures. Raw data and generated arrays should be restored locally before running the full pipeline.

## Recreate The Thesis Pipeline

Run commands from the repository root.

### 1. Verify Raw Data

```bash
python scripts/run_verify_datasets.py
```

Optional checks:

```bash
python scripts/run_verify_datasets.py --dataset 2
python scripts/run_verify_datasets.py --peek 1
```

Main output:

```text
results/tables/dataset_verification.csv
```

### 2. Build The File Catalog

```bash
python scripts/run_build_catalog.py
```

Main outputs:

```text
data/interim/catalog.csv
data/interim/catalog_summary.csv
```

### 3. Standardize Labels And Assign Splits

```bash
python scripts/run_standardize_catalog.py
```

Main outputs:

```text
data/interim/catalog_standardized.csv
results/tables/naming_convention_report.csv
```

The split protocol is file-level, not window-level. This avoids leakage from overlapping windows of the same source recording.

### 4. Run EDA And Time-Series Analysis

```bash
python scripts/run_phase2_eda.py
python scripts/run_timeseries_analysis.py
```

Main outputs:

```text
reports/
results/figures/eda/
results/figures/eda_timeseries/
results/tables/eda_stats.csv
results/tables/eda_class_separation.csv
results/tables/eda_split_diagnostics.csv
results/tables/timeseries_channel_analysis.csv
results/tables/timeseries_window_recommendations.csv
```

### 5. Preprocess Windows

```bash
python scripts/run_phase3_preprocess.py
```

Main output:

```text
data/processed/
```

Window sizes are configured in:

```text
configs/config.yaml
```

### 6. Build Unified Dataset Exports

```bash
python scripts/run_unify_dataset.py
```

Main output:

```text
data/unified/
```

The unified export builds component-focused groups:

```text
bearing_md
gearbox_md
```

### 7. Run Baseline Evaluation

Full baseline matrix:

```bash
python scripts/run_full_evaluation.py
```

Or run processed baselines directly:

```bash
python scripts/baselines/run_all_baselines.py --data-mode processed --datasets 1 2 4 5 8
```

Aggregate baseline results:

```bash
python scripts/aggregate_results.py
python scripts/aggregate_cv_results.py
python scripts/audit_baseline_matrix.py
```

Main outputs:

```text
results/tables/phase4_baseline_comparison.csv
results/tables/phase4_cv_comparison.csv
results/figures/phase4/
results/figures/phase4_cv/
```

### 8. Run MSPF-Net Evaluation

Run the MSPF-Net thesis suite:

```bash
python scripts/run_mspf_full_evaluation.py
```

Aggregate MSPF-Net results:

```bash
python scripts/aggregate_mspf_results.py
python scripts/audit_baseline_matrix.py
```

Main outputs:

```text
results/tables/mspf_net_all_results.csv
results/tables/mspf_net_comparison.csv
results/tables/mspf_net_ablation.csv
results/figures/mspf_net/
```

### 9. Plot Robustness Comparisons

```bash
python scripts/plot_robustness_comparison.py
```

Main outputs:

```text
results/tables/phase4_robustness_comparison.csv
results/figures/phase4/robustness_comparison_window.png
results/figures/phase4/robustness_comparison_file.png
```

## Useful Single-Model Commands

Run one model on one dataset:

```bash
python scripts/baselines/run_timesnet.py --datasets 1
python scripts/baselines/run_wdcnn.py --datasets 1
python scripts/baselines/run_mspf_net.py --datasets 1
```

Run any MSPF config:

```bash
python scripts/baselines/run_mspf_net_config.py configs/baselines/mspf_net_no_periodic_path.yaml --datasets 2
```

Run an inference smoke/latency check from saved training outputs:

```bash
python scripts/baselines/run_inference_check.py
```

## Test Different Models And Datasets

The training code is designed around three pieces:

```text
data/processed/                         # preprocessed windows
configs/baselines/<model>.yaml          # model/training config
scripts/baselines/run_<model>.py        # small runner wrapper
```

### Run Built-In Models On Different Datasets

Use the wrapper for a single model:

```bash
python scripts/baselines/run_timesnet.py --datasets 1
python scripts/baselines/run_timesnet.py --datasets 1 2 4 5 8
```

Use the sweep runner for one model across many datasets:

```bash
python scripts/baselines/run_model_sweep.py \
  --model timesnet \
  --data-mode processed \
  --datasets 1 2 4 5 8
```

Run all supported baselines:

```bash
python scripts/baselines/run_all_baselines.py \
  --data-mode processed \
  --datasets 1 2 4 5 8
```

Common options:

```bash
--epochs 50
--batch_size 128
--device cpu
--device cuda
--label-space fine
--label-space coarse
--cv-folds 5
```

Examples:

```bash
python scripts/baselines/run_wdcnn.py --datasets 5 --epochs 20 --device auto
python scripts/baselines/run_model_sweep.py --model resnet1d --datasets 2 8 --epochs 100
python scripts/baselines/run_all_baselines.py --datasets 1 2 4 5 8 --cv-folds 5
```

After runs finish:

```bash
python scripts/aggregate_results.py
python scripts/aggregate_cv_results.py
python scripts/audit_baseline_matrix.py
```

### Change Model Hyperparameters

Edit the model YAML under:

```text
configs/baselines/
```

For example:

```text
configs/baselines/timesnet.yaml
configs/baselines/wdcnn.yaml
configs/baselines/mspf_net.yaml
```

Typical fields:

```yaml
model:
  name: timesnet
  d_model: 64
  d_ff: 128
  n_layers: 3
  dropout: 0.1

training:
  batch_size: 256
  lr: 0.001
  weight_decay: 0.0001
```

Dataset-specific overrides can be placed in the same YAML:

```yaml
dataset_overrides:
  D2:
    training:
      batch_size: 64
  D8:
    training:
      batch_size: 64
```

Global defaults live in:

```text
configs/config.yaml
configs/baselines/default.yaml
```

### Add A New Model

Use this path when you want to test your own architecture with the existing training, metrics, aggregation, and dataset loaders.

1. Add the model file:

```text
src/mspf_net/models/baselines/my_model.py
```

The class should accept `in_channels` and `num_classes`, and return logits shaped `(batch, num_classes)`.

Minimal example:

```python
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).squeeze(-1)
        return self.head(z)
```

2. Register it in:

```text
src/mspf_net/models/baselines/factory.py
```

Add the import:

```python
from .my_model import MyModel
```

Add it to `registry`:

```python
"my_model": MyModel,
```

3. Create a config:

```text
configs/baselines/my_model.yaml
```

Example:

```yaml
model:
  name: my_model
  hidden: 64

training:
  batch_size: 256
  lr: 0.001
  weight_decay: 0.0001
```

4. Create a runner:

```text
scripts/baselines/run_my_model.py
```

Use this template:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.training.baseline_runner import run_baseline


if __name__ == "__main__":
    run_baseline("my_model", "configs/baselines/my_model.yaml")
```

5. Run it:

```bash
python scripts/baselines/run_my_model.py --datasets 1 --epochs 20
python scripts/baselines/run_model_sweep.py --model my_model --datasets 1 2 4 5 8
```

6. Aggregate results:

```bash
python scripts/aggregate_results.py
```

### Add A New Dataset

The code expects each dataset to become catalog rows, then processed windows. The cleanest route is to adapt the existing catalog and standardization flow.

1. Put files under a new raw folder:

```text
data/raw/dataset_9/
```

2. Add or update metadata rules:

```text
data/metadata/dataset_schema.json
data/metadata/label_mapping.json
```

Use these files to describe dataset names, label mappings, components, and canonical fault codes.

3. Update active dataset config if you want the new dataset in default runs:

```text
configs/config.yaml
```

Add the dataset id to the relevant lists, for example:

```yaml
datasets:
  active_thesis: [1, 2, 4, 5, 8, 9]
  primary_scratch: [1, 2, 4, 5, 8, 9]
```

4. Add window settings:

```yaml
windowing:
  per_dataset:
    "9": { window_size: 2048, hop_size: 1024 }
```

5. Rebuild the data pipeline:

```bash
python scripts/run_verify_datasets.py --dataset 9
python scripts/run_build_catalog.py
python scripts/run_standardize_catalog.py
python scripts/run_phase2_eda.py --dataset 9
python scripts/run_phase3_preprocess.py --dataset 9
```

6. Run models on the new dataset:

```bash
python scripts/baselines/run_timesnet.py --datasets 9
python scripts/baselines/run_model_sweep.py --model wdcnn --datasets 9
```

7. If you want unified experiments too:

```bash
python scripts/run_unify_dataset.py
python scripts/baselines/run_all_baselines.py --data-mode unified --label-space fine
```

### Add A New Split Protocol

File-level splits are assigned during standardization:

```text
src/mspf_net/data/standardize_catalog.py
```

Use this file when a new dataset needs a paper-specific split, such as speed holdout, condition holdout, or variable-speed robustness holdout. After changing the split logic, rerun:

```bash
python scripts/run_standardize_catalog.py
python scripts/run_phase3_preprocess.py --dataset <id>
```

### Expected Data Format After Preprocessing

Training uses arrays under:

```text
data/processed/
```

Each dataset split is expected to provide windows shaped:

```text
(N, C, L)
```

Where:

- `N` is the number of windows
- `C` is the number of channels
- `L` is the window length

Models should therefore accept tensors shaped:

```text
(batch, channels, length)
```

and return:

```text
(batch, num_classes)
```

## Model Code Map

```text
src/mspf_net/models/mspf_net.py       # public MSPF-Net facade
src/mspf_net/models/mspf/             # MSPF-Net core blocks
src/mspf_net/models/baselines/        # baseline architectures
src/mspf_net/training/baseline_runner.py
src/mspf_net/training/trainer.py
src/mspf_net/training/eval_utils.py
src/mspf_net/training/metrics.py
```

## Embedded Deployment

Raspberry Pi 5 export and benchmark instructions are in:

```text
embedded/README.md
```
