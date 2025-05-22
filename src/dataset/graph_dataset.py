import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from src.graphs.edge_builders import news, supply_chain
from src.utils.data_utils import load_price_features  # Adjust if your loader is elsewhere

_EDGE_FNS = [news.build_news_edges, supply_chain.build_edges]

class GraphDataset(Dataset):
    """
    Yields a dict:
      {
        'graph' : torch_geometric.data.Data (or Batch),
        'tft_x' : price feature tensor for TFT encoder,
        'y'     : target vector (next-period return, etc.)
      }
    """
    def __init__(self, dates: list[str], universe: list[str]):
        self.dates, self.universe = dates, universe

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        # ----- build/merge graphs -----
        data_objs = [fn(date, self.universe) for fn in _EDGE_FNS]
        graph = Batch.from_data_list(data_objs)
        # --- Unity feature + label ---
        from src.utils.data_utils import load_price_features
        tft_x, y = load_price_features(date, ["U"])  # shape (1, T, F)
        # Mask: True for Unity node (assumes graph has .ticker attribute per node)
        # If not, just mask index 0 (Unity is first node)
        target_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        if hasattr(graph, 'ticker'):
            target_mask = (graph.ticker == "U")
        else:
            target_mask[0] = True
        return {"graph": graph, "tft_x": tft_x, "y": y, "target_mask": target_mask}
