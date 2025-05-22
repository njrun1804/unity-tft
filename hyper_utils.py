def sample_tft_params(trial):
    return {
        # ─ network depth/width
        "hidden_size"      : trial.suggest_int("hidden_size", 32, 192, step=32),
        "lstm_layers"      : trial.suggest_int("lstm_layers", 1, 3),
        "dropout"          : trial.suggest_float("dropout", 0.05, 0.30),
        # ─ optimiser
        "lr"               : trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        # ─ data / training
        "batch_size"       : trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "min_encoder_len"  : 48,
        "max_encoder_len"  : 48,
        "predict_len"      : 39,
        "early_stop_patience": 5,
        "num_workers"      : 4,      # easy to down-shift on CPU boxes
        "max_epochs"       : 30,
        "seed"             : 42,
    }
