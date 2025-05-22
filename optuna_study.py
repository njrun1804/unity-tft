import optuna
import pandas as pd
from objective import objective

full_df = pd.read_parquet("data/U_5min.parquet")

study = optuna.create_study(
    study_name="unity_tft_walkforward",
    direction="minimize",
    storage="sqlite:///optuna_unity.db",
    load_if_exists=True,
)
study.optimize(objective, n_trials=60, timeout=3*60*60)   # ~3h GPU wall
print("Best params:", study.best_params, "RMSE:", study.best_value)
