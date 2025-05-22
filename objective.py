from train_tft import train_once
from pathlib import Path
import numpy as np, tempfile
from cv_utils import rolling_cv
from hyper_utils import sample_tft_params
import pandas as pd

def objective(trial, full_df):
    params = sample_tft_params(trial)
    fold_scores = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i, (train_df, val_df) in enumerate(rolling_cv(full_df, n_folds=4)):
            fold_rmse = train_once(
                train_df, val_df,
                params,
                model_dir = tmp_path / f"fold{i}"
            )
            fold_scores.append(fold_rmse)
    return float(np.mean(fold_scores))
