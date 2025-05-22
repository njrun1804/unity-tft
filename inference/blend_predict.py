import lightgbm as lgb
import torch
import numpy as np
import pandas as pd

# load trained blender
blender = lgb.Booster(model_file="lightgbm_blend.txt")

def blended_predict(tft_pred: torch.Tensor, gnn_pred: torch.Tensor):
    X = pd.DataFrame({
        "pred_tft": tft_pred.cpu().numpy().ravel(),
        "pred_gnn": gnn_pred.cpu().numpy().ravel()
    })
    preds = blender.predict(X)
    return torch.tensor(preds, dtype=torch.float, device=tft_pred.device).unsqueeze(1)
