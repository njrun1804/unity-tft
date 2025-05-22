import os
import pytest
import torch
import pandas as pd
import numpy as np
from unity_gpu.data import fetch_intraday, make_tensor


@pytest.mark.skipif(
    "AV_KEY" not in os.environ,
    reason="Set the Alpha Vantage key in $AV_KEY to run this test",
)
def test_fetch_intraday(tmp_path):
    """Pull one intraday file and make sure it’s non-empty and cached."""
    df = fetch_intraday("U", os.environ["AV_KEY"], cache_dir=tmp_path)
    assert not df.empty, "DataFrame came back empty"
    assert "4. close" in df.columns  # Alpha Vantage naming


def test_make_tensor_vectorized():
    # Create dummy DataFrame with enough rows and required columns
    n_rows = 60
    n_features = 3
    data = np.random.randn(n_rows, n_features)
    columns = [f"feat_{i}" for i in range(n_features - 1)] + ["ret_pp"]
    df = pd.DataFrame(data, columns=columns)
    # ret_pp is the last column
    X_seq, y_seq = make_tensor(df)
    assert X_seq.shape[1] == 48, "SEQ_LEN should be 48"
    assert X_seq.shape[0] == y_seq.shape[0], "Number of samples should match"
    assert X_seq.shape[2] == n_features, "Feature dimension should match input"
