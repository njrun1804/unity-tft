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
    n_rows = 80  # Ensure enough rows for SEQ_LEN=48 after shifting and alignment
    data = np.random.randn(n_rows, 4)
    columns = ["4. close", "5. volume", "feat_1", "ret_pp"]
    df = pd.DataFrame(data, columns=columns)
    # ret_pp is the last column
    X_seq, y_seq = make_tensor(df)
    SEQ_LEN = 48
    # Only check the window and feature dimensions
    assert X_seq.shape[1] == SEQ_LEN, f"Sequence length should be {SEQ_LEN}, got {X_seq.shape[1]}"
    assert X_seq.shape[2] == 4, f"Feature dimension should be 4, got {X_seq.shape[2]}"
