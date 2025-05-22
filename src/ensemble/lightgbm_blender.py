import lightgbm as lgb
import pandas as pd
import optuna

def train_blender(oof_path="oof_preds.parquet",
                  objective="quantile",
                  alpha=0.5,
                  n_trials=50,
                  seed=42):
    df = pd.read_parquet(oof_path)
    X  = df[["pred_tft","pred_gnn"]]
    y  = df["y"]
    def objective_fn(trial):
        params = {
            "objective": objective,
            "metric": "quantile",
            "alpha": alpha,
            "learning_rate": trial.suggest_float("lr", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("leaves", 4, 64),
            "feature_fraction": trial.suggest_float("ff", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bf", 0.6, 1.0),
            "bagging_freq": 1,
            "seed": seed,
        }
        gbm = lgb.train(params,
                        lgb.Dataset(X, y),
                        num_boost_round=500,
                        verbose_eval=False)
        pred = gbm.predict(X)
        loss = (abs(pred - y)).mean()      # CRPS proxy
        return loss
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_fn, n_trials=n_trials)
    best_params = study.best_params | {
        "objective": objective,
        "metric": "quantile",
        "alpha": alpha,
        "seed": seed,
    }
    model = lgb.train(best_params, lgb.Dataset(X, y), num_boost_round=500)
    return model, study.best_value, best_params
