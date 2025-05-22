import os
import optuna
import pandas as pd
from pathlib import Path
from objective import objective
import ray
from optuna.integration import RayTuneSampler, RayTunePruner

# Initialize Ray for parallelism on all CPU cores
ray.init(num_cpus=os.cpu_count())

full_df = pd.read_parquet(Path("data") / "U_5min.parquet")

study = optuna.create_study(
    study_name      = "unity_tft_walkforward",
    direction       = "minimize",
    storage         = "sqlite:///optuna_unity.db",
    load_if_exists  = True,
    sampler         = RayTuneSampler(),
    pruner          = RayTunePruner(),
)

study.optimize(lambda t: objective(t, full_df),
               n_trials=60,
               timeout=3*60*60)   # ≈3 h with one modest GPU

print("Best params:", study.best_params)
print("Best mean RMSE:", study.best_value)
