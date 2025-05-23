import pandas as pd
from src.utils.preprocess import preprocess_df

def test_ticker_column_present():
    # Load a sample preprocessed parquet (train or val)
    df = pd.read_parquet('data/pre_train.parquet')
    df = preprocess_df(df)
    assert 'ticker' in df.columns, "Ticker column missing after preprocessing!"
    assert df['ticker'].notnull().all(), "Null values in ticker column!"

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
