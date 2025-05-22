from pathlib import Path
import duckdb, pandas as pd, torch
from torch_geometric.data import Data

def build_edges(date: str, universe: list[str]) -> Data:
    """
    One directed edge per (supplier → customer) pair active on `date`.
    Expects a parquet file `data/alt/supply_chain/{date}.parquet`
    with columns: supplier, customer, rel_strength (0-1).
    """
    pq = Path(f"data/alt/supply_chain/{date}.parquet")
    if not pq.exists():
        raise FileNotFoundError(pq)

    df = duckdb.read_parquet(pq).to_df()
    df = df[df["supplier"].isin(universe) & df["customer"].isin(universe)]

    # map tickers → contiguous node ids
    id_map = {t: i for i, t in enumerate(universe)}
    src = torch.tensor(df["supplier"].map(id_map).values, dtype=torch.long)
    dst = torch.tensor(df["customer"].map(id_map).values, dtype=torch.long)
    edge_index = torch.stack([src, dst])          # shape [2, E]
    edge_attr  = torch.tensor(df["rel_strength"].values, dtype=torch.float).unsqueeze(1)

    return Data(edge_index=edge_index, edge_attr=edge_attr)
