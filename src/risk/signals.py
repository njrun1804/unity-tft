"""Project edge attributes through a risk-propagation layer (e.g., 1-hop Laplacian diffusion or GAT). Output per-ticker risk_score, certainty."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import HeteroData

def propagate_risk_gat(data: HeteroData, in_channels: int = 4, out_channels: int = 1):
    """
    Project edge attributes through a GAT layer to get per-ticker risk scores.
    """
    x, edge_index, edge_attr = data["stock"].x, data["stock", "news", "stock"].edge_index, data["stock", "news", "stock"].edge_attr
    gat = GATConv(in_channels, out_channels, heads=2, concat=False)
    x = gat(x, edge_index, edge_attr)
    risk_score = F.relu(x).squeeze()
    certainty = torch.sigmoid(risk_score)
    return risk_score, certainty

def propagate_risk_laplacian(data: HeteroData):
    """
    Simple Laplacian diffusion: risk = sum of edge_attr for each node.
    """
    edge_index = data["stock", "news", "stock"].edge_index
    edge_attr = data["stock", "news", "stock"].edge_attr
    num_nodes = data["stock"].x.size(0)
    risk_score = torch.zeros(num_nodes)
    for i in range(edge_index.size(1)):
        tgt = edge_index[1, i]
        risk_score[tgt] += edge_attr[i, 0]  # Use first attr (e.g., sent_mean)
    certainty = torch.sigmoid(risk_score)
    return risk_score, certainty
