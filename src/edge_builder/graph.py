"""Build heterograph with edge_index of (source_id, target_id) and edge_attr = decay_weight, sentiment_mean, sentiment_vol, event_onehot… Ensure deterministic ordering via sorted unique tickers."""
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def build_hetero_graph(edge_df: pd.DataFrame) -> HeteroData:
    """
    Build a torch-geometric HeteroData graph from edge_df with event one-hot features.
    """
    # Collect all unique tickers for deterministic node ordering
    all_tickers = np.unique(edge_df[["ticker_a", "ticker_b"]].values)
    label_encoder = LabelEncoder()
    label_encoder.fit(all_tickers)
    senders = edge_df[["ticker_a", "ticker_b"]].apply(label_encoder.transform)
    edge_index = torch.tensor(senders.values.T, dtype=torch.long)
    # Find all event one-hot columns
    event_cols = [c for c in edge_df.columns if c.startswith("event_")]
    edge_attr_cols = ["sentiment_mean", "sentiment_std", "decay_weight_sum"] + event_cols
    edge_attr = torch.tensor(edge_df[edge_attr_cols].values, dtype=torch.float32)
    g = HeteroData()
    g["stock"].x = torch.eye(len(label_encoder.classes_))
    g["stock", "news", "stock"].edge_index = edge_index
    g["stock", "news", "stock"].edge_attr  = edge_attr
    g["stock"].ticker = label_encoder.classes_
    return g
