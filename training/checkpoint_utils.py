import os, tempfile, torch
from ray.tune import Checkpoint

def save_ckpt(model) -> Checkpoint:
    ckpt_dir = tempfile.mkdtemp()
    model_path = os.path.join(ckpt_dir, "model.pt")
    if hasattr(model, "save"):
        model.save(model_path)
    else:
        torch.save(model.state_dict(), model_path)
    return Checkpoint.from_directory(ckpt_dir)
