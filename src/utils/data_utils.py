import numpy as np
from pytorch_forecasting import TimeSeriesDataSet

def make_dataset(df, params):
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="up_next_day",
        group_ids=["grp"],
        min_encoder_length=params["min_encoder_len"],
        max_encoder_length=params["max_encoder_len"],
        min_prediction_length=params["predict_len"],
        max_prediction_length=params["predict_len"],
        static_categoricals=[],
        time_varying_known_reals=["close"],
        time_varying_unknown_reals=["close"],
        target_normalizer=None,
        categorical_encoders={},
        add_relative_time_idx=True,
        add_target_scales=False,
        allow_missing_timesteps=True,
    )
