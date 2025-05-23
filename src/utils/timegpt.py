import os
import pandas as pd
from nixtla import NixtlaClient

# Use the provided API key directly
_client = NixtlaClient(api_key="nixak-xYTMFfVTyqCkdF0HqjaMhfw6CH3NoAbH1G1BCtgHaJFxhFYDXkOyWRgFLzu0T6PHGfJHPAhQ5drdz4NI")

def timegpt_forecast(df: pd.DataFrame, horizon: int, freq: str = "D",
                     time_col: str = "date", target_col: str = "close",
                     n_samples: int = 100) -> pd.DataFrame:
    """
    Zero-shot forecast via Nixtla TimeGPT.
      df: historical frame containing at least {time_col, target_col}
      horizon: forecast length (same as params['predict_len'])
      freq: pandas offset alias ('D', 'H', etc.)
    Returns a tidy dataframe with columns:
      {time_col, 'y_hat', 'quantile', 'sample_id'}
    """
    fcst = _client.forecast(
        df=df[[time_col, target_col]].rename(columns={time_col: "timestamp",
                                                     target_col: "value"}),
        h=horizon,
        freq=freq,
        time_col="timestamp",
        target_col="value",
        n_samples=n_samples
    )
    # Nixtla returns one row per sample; melt so we keep them all
    return (fcst
            .melt(id_vars=["timestamp"], var_name="sample_id",
                  value_name="y_hat")
            .rename(columns={"timestamp": time_col}))

def blend_and_score_crps(y_true, current_ensemble_samples, tg_samples):
    """
    Optionally blend TimeGPT samples with the current ensemble and compute blended CRPS.
    """
    import numpy as np
    from properscoring import crps_ensemble
    blended = np.concatenate([current_ensemble_samples, tg_samples], axis=0)
    blended_crps = crps_ensemble(y_true, blended).mean()
    return blended_crps
