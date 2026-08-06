# Raspberry Pi 5 Embedded Benchmark

This folder contains the embedded export and benchmark tools used to evaluate trained Phase 4 torch models on a Raspberry Pi 5 CPU.

The embedded flow supports:

- Exporting trained model bundles from desktop results
- Running inference smoke checks without test data
- Running full latency, FLOPs, memory, and accuracy benchmarks when `data/processed/` is available
- Writing final Pi benchmark CSVs under `embedded/results/tables/`

## Layout

```text
embedded/
├── README.md
├── requirements-pi.txt
├── configs/
│   └── pi5_softmax.yaml
├── mspf_embedded/
│   ├── benchmark.py
│   ├── bundle.py
│   └── _paths.py
├── scripts/
│   ├── export_bundle.py
│   ├── export_all_bundles.py
│   ├── run_benchmark.py
│   ├── run_inference_matrix.py
│   └── run_matrix.py
├── bundles/                 # ignored exported bundles
└── results/tables/          # final embedded CSV summaries
```

`embedded/bundles/` is ignored by Git because bundles contain model checkpoints.

## Setup On Raspberry Pi

Use a 64-bit Raspberry Pi OS or another aarch64 Linux environment.

```bash
cd ~/MSPF-Net
python3 -m venv .venv
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r embedded/requirements-pi.txt
pip install -e .
```

## Files Needed On The Pi

For inference-only checks:

```text
src/
embedded/
pyproject.toml
```

For full accuracy benchmarks, also sync:

```text
data/processed/
```

Example sync from the development machine:

```bash
rsync -av src/ pi@<host>:~/MSPF-Net/src/
rsync -av embedded/ pi@<host>:~/MSPF-Net/embedded/
rsync -av pyproject.toml pi@<host>:~/MSPF-Net/pyproject.toml

# Optional, only for full accuracy benchmarks
rsync -av data/processed/ pi@<host>:~/MSPF-Net/data/processed/
```

## Recreate Embedded Results

Run these commands from the repository root.

### 1. Export Bundles On The Development Machine

Export all torch model bundles from saved training results:

```bash
export PYTHONPATH=src
python embedded/scripts/export_all_bundles.py
```

This creates ignored bundle folders under:

```text
embedded/bundles/
```

Each bundle contains:

```text
checkpoint.pt
model.yaml
manifest.json
```

Export one MSPF-Net bundle manually:

```bash
python embedded/scripts/export_bundle.py \
  --results-json results/baselines/mspf_net/processed/d1/mspf_net_d1_softmax_results.json \
  --out-dir embedded/bundles/mspf_net_d1_softmax
```

### 2. Run Inference Smoke Checks On The Pi

This does not require `data/processed/`.

```bash
export PYTHONPATH=src
python embedded/scripts/run_inference_matrix.py \
  --latency-runs 100 \
  --latency-warmup 10
```

Main output:

```text
embedded/results/tables/pi5_inference_check.csv
```

### 3. Run Full Accuracy Benchmarks On The Pi

This requires `data/processed/`.

Single bundle:

```bash
python embedded/scripts/run_benchmark.py \
  --bundle-dir embedded/bundles/mspf_net_d1_softmax \
  --processed-dir data/processed \
  --config embedded/configs/pi5_softmax.yaml
```

All bundles:

```bash
python embedded/scripts/run_matrix.py \
  --bundles-root embedded/bundles \
  --processed-dir data/processed \
  --config embedded/configs/pi5_softmax.yaml
```

Main output:

```text
embedded/results/tables/pi5_benchmark.csv
```

## Configuration

Edit:

```text
embedded/configs/pi5_softmax.yaml
```

Important keys:

```yaml
embedded:
  device: cpu
  batch_size: 1
  latency_runs: 1000
  benchmark_latency_runs: 100
  latency_warmup: 50
  torch_num_threads: 4
  processed_dir: data/processed
  results_dir: embedded/results
```

## Final Embedded Tables

The cleaned thesis repo keeps only final embedded CSV summaries:

```text
embedded/results/tables/pi5_inference_check.csv
embedded/results/tables/pi5_benchmark.csv
embedded/results/tables/pi5_benchmark_latest.csv
embedded/results/tables/pi5_complexity_canonical.csv
```

Generated JSON run logs and checkpoint bundles are intentionally ignored.
