from pathlib import Path
import torch
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer

def ensemble_predict(
    checkpoints_dir: Path,
    dataset,
    dataloader,
    top_k: int = 5,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
):
    """
    Load the top-k checkpoints produced by ModelCheckpoint and
    return the mean-pooled prediction tensor.
    """
    ckpts = sorted(checkpoints_dir.glob("epoch*=ckpt"), key=lambda p: p.stat().st_mtime)[:top_k]
    preds = []
    for ckpt in ckpts:
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location=device)
        model.eval()
        with torch.no_grad():
            batch_pred = model.predict(dataloader, mode="prediction", return_y=False)
        preds.append(batch_pred.cpu())
        del model  # free memory
    return torch.stack(preds).mean(dim=0)

def save_to_csv(pred_tensor, output_path: Path, id_col="time_idx"):
    df = pd.DataFrame(pred_tensor.numpy())
    df.to_csv(output_path, index=False)
