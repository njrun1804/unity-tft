import pandas as pd, pytest, numpy as np, pathlib, os

@pytest.fixture
def dummy_preds(tmp_path):
    df = pd.DataFrame({
        "symbol":  ["U"] * 3,
        "P10":     [30.1, 30.2, 30.4],
        "P50":     [31.0, 31.1, 31.2],
        "P90":     [33.0, 33.2, 33.3],
        "certainty":[0.82, 0.76, 0.79]
    })
    p = tmp_path/"predictions.parquet"
    df.to_parquet(p, index=False)
    return p

def test_certainty_range(dummy_preds):
    df = pd.read_parquet(dummy_preds)
    assert df["certainty"].between(0, 1).all()

def test_ingest_merge(dummy_preds, tmp_path):
    # call your ingest CLI with dummy file + minimal watch/options parquet
    cache = tmp_path/"cache.parquet"
    rc = os.system(
        f"python scripts/deep_research_ingest.py "
        f"--option_file tests/data/opt_dummy.parquet "
        f"--watch_file  tests/data/watch_dummy.parquet "
        f"--pred_dir    {dummy_preds.parent} "
        f"--cache_file  {cache}"
    )
    assert rc == 0
    df = pd.read_parquet(cache)
    assert "certainty" in df.columns
