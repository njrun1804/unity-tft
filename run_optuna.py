import os
os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
from pathlib import Path
from urllib.parse import quote
import ray
from ray import tune
from ray.tune import TuneConfig, RunConfig
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import MedianStoppingRule
from training.objectives import lstm_objective

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
N_TRIALS = 60
EPOCH_BUDGET = 50
TRIAL_CPUS = 3          # 4 trials at once on 12-core M4 Pro
TOTAL_CPUS = 12

# ---------------------------------------------------------------------------
# Hyper-parameter space
# ---------------------------------------------------------------------------
search_space = {
    "hidden_size":  tune.choice([64, 128, 256]),
    "lstm_layers":  tune.randint(1, 4),
    "dropout":      tune.uniform(0.1, 0.5),
    "lr":           tune.loguniform(1e-4, 1e-2),
    "seed":         tune.randint(1, 10_000),
    "epochs":       EPOCH_BUDGET,         # fixed for all trials
}

# ---------------------------------------------------------------------------
# Searcher & scheduler (no pruning)
# ---------------------------------------------------------------------------
search_alg = OptunaSearch(metric="loss", mode="min")
scheduler  = MedianStoppingRule(metric="loss", mode="min", grace_period=10, min_samples_required=5)

# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------
run_cfg = RunConfig(
    name="unity_sweep",
    storage_path=f"file://{quote(str(Path('ray_results').resolve()))}",
    log_to_file=("stdout.log", "stderr.log"),
    verbose=1,
)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ray.init(include_dashboard=True)

    tuner = tune.Tuner(
        tune.with_resources(lstm_objective, {"cpu": TRIAL_CPUS}),
        param_space=search_space,
        tune_config=TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=N_TRIALS,
            max_concurrent_trials=TOTAL_CPUS // TRIAL_CPUS,
        ),
        run_config=run_cfg,
    )

    results = tuner.fit()
    best = results.get_best_result(metric="loss", mode="min")
    print("Best metrics:", best.metrics)
    print("Best config :", best.config)

    ray.shutdown()
