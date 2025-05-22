import torch
import torch.nn as nn

class WeightedBlend(nn.Module):
    """
    Simple learnable convex combination of model outputs.
    Feed it (batch, n_models) → (batch, 1)
    """
    def __init__(self, n_models: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_models) / n_models)

    def forward(self, preds):          # preds shape (B, n_models)
        weights = torch.softmax(self.w, dim=0)
        return (preds * weights).sum(dim=1, keepdim=True)
