import torch, os, pandas as pd
from pathlib import Path
from .model import SimpleLSTM
from .data import fetch_intraday, make_tensor

# Always resolve model path relative to project root
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "lstm-v0.ckpt"
TICKER     = "U"
THRESH     = 0.0            # >0 ⇒ bullish

def load_latest_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleLSTM.load_from_checkpoint(str(MODEL_PATH))
    model.to(device)
    model.eval()
    return model

def latest_score():
    key  = os.environ["AV_KEY"]
    df   = fetch_intraday(TICKER, key)     # uses cache
    X,_  = make_tensor(df)
    x    = X[-1:].to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    with torch.no_grad():
        pred = load_latest_model()(x).item()
    return pred

if __name__ == "__main__":
    score = latest_score()
    action = "RUN wheel today" if score > THRESH else "PASS"
    print(f"Score = {score:+.4f}  →  {action}")
