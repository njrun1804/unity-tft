import torch
import pandas as pd
from torch_geometric.data import Data

def build_news_edges(news_df: pd.DataFrame, ticker2idx: dict) -> Data:
    """
    Build a PyG Data object from a DataFrame of news co-mentions.
    Args:
        news_df: DataFrame with columns ['date', 'ticker_a', 'ticker_b', 'sentiment', ...]
        ticker2idx: dict mapping ticker to node index
    Returns:
        PyG Data object with edge_index and edge_attr
    """
    # Filter for valid tickers
    news_df = news_df[news_df['ticker_a'].isin(ticker2idx) & news_df['ticker_b'].isin(ticker2idx)]
    # Build edge_index
    src = news_df['ticker_a'].map(ticker2idx).values
    dst = news_df['ticker_b'].map(ticker2idx).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # Edge attributes (e.g., sentiment)
    edge_attr = torch.tensor(news_df['sentiment'].values, dtype=torch.float).unsqueeze(1)
    # Optionally add more edge features here
    return Data(edge_index=edge_index, edge_attr=edge_attr)
