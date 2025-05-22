import torch, os
from unity_gpu.data import fetch_intraday, engineer_features, make_tensor
from unity_gpu.model import SimpleLSTM

torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)
torch.set_float32_matmul_precision('medium')  # Apple M-series: fast matmul

N_FEATURES = 4  # keep in sync with training
BATCH = 1024
NUM_W = 8

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    df = fetch_intraday("U", os.environ.get("AV_KEY", "demo"))
    feats = engineer_features(df)
    # Ensure feature shape matches training: [N, N_FEATURES]
    X = torch.tensor(feats.values, dtype=torch.float32)
    y = torch.tensor(feats["ret_pp"].shift(-1).dropna().values, dtype=torch.float32)
    X = X[:-1]  # align after shift
    SEQ_LEN = 48
    X_seq, y_seq = [], []
    for i in range(len(X) - SEQ_LEN):
        X_seq.append(X[i : i + SEQ_LEN])
        y_seq.append(y[i + SEQ_LEN])
    X_eval = torch.stack(X_seq).to(device)
    y_eval = torch.tensor(y_seq).to(device)

    # Defensive checkpoint loading
    ckpt = torch.load("models/lstm-v0.ckpt", weights_only=True, map_location=device)
    model = SimpleLSTM.load_from_checkpoint("models/lstm-v0.ckpt", map_location=device)
    model = torch.compile(model) if hasattr(torch, 'compile') else model
    model = model.to(device)
    model.eval()
    # Ensure data and model are on the same device
    X_eval = X_eval.to(device)
    y_eval = y_eval.to(device)
    with torch.no_grad():
        pred = model(X_eval[-500:]).squeeze()
    # Diagnostics: mean/std of predictions and targets
    print(f"Pred mean: {pred.mean().item():.3f}, std: {pred.std().item():.3f}")
    print(f"Target mean: {y_eval[-500:].mean().item():.3f}, std: {y_eval[-500:].std().item():.3f}")
    # Baseline: always predict zero
    baseline_hit = (torch.zeros_like(y_eval[-500:]) == torch.sign(y_eval[-500:])).float().mean().item()
    print(f"Baseline (predict 0) sign hit-rate: {baseline_hit:.3f}")
    # Baseline: previous return sign
    prev_sign = torch.sign(y_eval[-501:-1])
    curr_sign = torch.sign(y_eval[-500:])
    prev_hit = (prev_sign == curr_sign).float().mean().item()
    print(f"Baseline (prev sign) hit-rate: {prev_hit:.3f}")
    sign_hit = (torch.sign(pred) == torch.sign(y_eval[-500:])).float().mean().item()
    corr = torch.corrcoef(torch.stack((pred, y_eval[-500:])))[0, 1].item()
    print(f"Sign hit-rate: {sign_hit:.3f}   Corr: {corr:+.3f}")

if __name__ == "__main__":
    main()
