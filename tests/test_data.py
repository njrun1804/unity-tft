import os, pytest
from unity_gpu.data import fetch_intraday

@pytest.mark.skipif(
    "AV_KEY" not in os.environ,
    reason="Set the Alpha Vantage key in $AV_KEY to run this test",
)
def test_fetch_intraday(tmp_path):
    """Pull one intraday file and make sure it’s non-empty and cached."""
    df = fetch_intraday("U", os.environ["AV_KEY"], cache_dir=tmp_path)
    assert not df.empty, "DataFrame came back empty"
    assert "4. close" in df.columns  # Alpha Vantage naming
