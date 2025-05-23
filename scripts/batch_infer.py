#!/usr/bin/env python3
"""
batch_infer.py
─────────────────────────────────────────────────────────
Usage:
python batch_infer.py \
    --checkpoints data/ckpts/ \
    --dataset     data/live/2025-05-23.parquet \
    --out         data/2025-05-23/predictions.parquet
"""

import argparse, joblib, pandas as pd, torch
from pathlib import Path
from train_lstm import (
    ensemble_predict,                      # ← your existing helper
    compute_certainty                      # ← the calibrated wrapper
)

CKPT_GLOB = "epoch*=ckpt"

def main(args):
    ckpts_dir = Path(args.checkpoints)
    ckpts = sorted(ckpts_dir.glob(CKPT_GLOB))[: args.top_k]

    # 1) point & quantile forecasts  ──────────────────────────
    preds_dict, ckpt_preds, quant_df = ensemble_predict(
        checkpoints     = ckpts,
        dataset_path    = args.dataset,
        device          = args.device,
        quantiles       = args.quantiles
    )
    pred_df = pd.DataFrame(preds_dict)

    # 2) certainty  ────────────────────────────────────────────
    iso = joblib.load(args.calibrator)
    certainty = compute_certainty(
        quant_df   = quant_df,
        ckpt_preds = ckpt_preds,
        iso_model  = iso,
        y_val      = None,                # not needed at inference
        y_val_hat  = None
    )
    pred_df["certainty"] = certainty

    # 3) housekeeping & save  ─────────────────────────────────
    pred_df.insert(0, "symbol", args.symbol)    # simple 1-symbol case
    pred_df.to_parquet(args.out, index=False)
    print(f"✅  wrote {args.out.relative_to(Path.cwd())}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--dataset",     required=True)
    p.add_argument("--out",         required=True)
    p.add_argument("--symbol",      required=True)
    p.add_argument("--calibrator",  default="models/certainty_calibrator.pkl")
    p.add_argument("--device",      default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--top_k",       type=int, default=5)
    p.add_argument("--quantiles",   nargs="+", type=float, default=[0.1, 0.5, 0.9])
    main(p.parse_args())
