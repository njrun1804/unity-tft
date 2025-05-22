# UnityPrice-GPU-Sandbox

A high-performance, reproducible PyTorch Temporal Fusion Transformer (TFT) training pipeline, optimized for Apple Silicon (M-series) Macs. This project focuses on local ML experimentation, robust training, and best practices for collaborative research and development.

## Features
- **Apple Silicon/MPS Optimizations:**
  - Automatic device selection (MPS/CPU)
  - Fast matmul precision for Apple M-series
  - OMP/MKL thread scaling for maximum CPU utilization
- **Robust Training Pipeline:**
  - PyTorch Lightning Trainer with Apple Silicon support
  - Mixed precision (16-mixed), gradient clipping, early stopping
  - Hybrid Quantile+MSE loss, top-k checkpointing
  - Cosine annealing LR scheduler with warm restarts
  - DataLoader with persistent workers, prefetching, and optimal num_workers
- **Reproducibility & Experimentation:**
  - Hydra/OMEGACONF config, MLflow logging, deterministic seeding
  - Optuna/Ray stub for hyperparameter search
- **Collaboration Ready:**
  - GitHub repository, pre-commit hooks, CI/CD via GitHub Actions
  - Modular codebase (src/, scripts/, tests/)

## Quickstart
### 1. Clone and set up environment
```bash
git clone https://github.com/njrun1804/unity-tft.git
cd unity-tft
# Recommended: create conda env
conda env create -f env.yml
conda activate unity-gpu
# Or use pip (see env.yml for requirements)
```

### 2. Prepare data
- Place your time series data (e.g., `U_5min.parquet`) in the `data/` directory.
- Data should be a Parquet file with OHLCV columns. See `src/unity_gpu/data.py` for feature engineering details.

### 3. Train the model
```bash
python train_tft.py
```
- Configurable via `src/config/default.yaml` or Hydra CLI overrides.
- Checkpoints and logs are saved in `models/` and `mlruns/`.

### 4. Run tests/diagnostics
```bash
pytest tests/
python tests/quick_diag.py
```

### 5. Hyperparameter search (Optuna/Ray)
```bash
python run_optuna.py  # (stubbed, see file for details)
```

## Usage

### Training (Hydra config mode)
Run with the default config:
```bash
python train_tft.py
```
Override config values via CLI:
```bash
python train_tft.py data.train_csv=data/pre_train.parquet data.val_csv=data/pre_val.parquet model_dir=models/tft-exp1
```

### Training (Legacy CLI mode)
For backward compatibility:
```bash
python train_tft.py --params_json params.json --train_csv data/pre_train.parquet --val_csv data/pre_val.parquet --model_dir models/tmp
```

### Ensemble Prediction
Average predictions from top-5 checkpoints:
```bash
python ensemble_predict.py --input_csv data/pre_val.parquet --model_dir models/tft-exp1
```

### Auto-Resume Training
Automatically resume interrupted training:
```bash
bash scripts/auto_resume_train.sh
```

## Project Structure
- `train_tft.py` — Main training script (TFT, Apple Silicon optimized)
- `src/unity_gpu/` — Data, model, and training utilities
- `src/utils/data_utils.py` — Dataset construction helpers
- `scripts/dev_env.sh` — Shell helper for Apple Silicon thread/env setup
- `tests/` — Quick diagnostics and test scripts
- `env.yml` — Conda environment (Python 3.11, PyTorch, Lightning, etc.)
- `.github/workflows/` — CI/CD workflows (see below)

## Contribution Guidelines
- Follow code style enforced by Ruff and Flake8 (`pyproject.toml`, `.flake8`).
- Add/modify tests in `tests/` for new features or bugfixes.
- Document new modules and functions with clear docstrings.
- Use feature branches and submit pull requests for review.

## Troubleshooting
- For Apple Silicon, ensure you are using the correct Python and PyTorch builds (see `env.yml`).
- If you encounter missing package errors, run:
  ```bash
  pip install -r requirements.txt  # or conda install --file env.yml
  ```
- For MLflow UI: `mlflow ui --backend-store-uri file:mlruns`

## CI/CD
- GitHub Actions will lint and test all PRs and pushes to `main`
- See `.github/workflows/ci.yml` for details

## License
MIT License (see `LICENSE` file)

## Acknowledgements
- Built with PyTorch, PyTorch Lightning, and Apple Silicon MPS support
- Inspired by best practices in time series forecasting and ML engineering

## Ray Tune Parallel Sweeps

To launch a Ray Tune sweep:

```bash
pip install "ray[tune]" optuna
python tune_train.py
```

To get the best config after a sweep:
```python
from ray.tune import Analysis
best = Analysis("ray_results/ray_sweep").best_config
print(best)
```

---
For questions or collaboration, open an issue or contact the maintainer via GitHub.
