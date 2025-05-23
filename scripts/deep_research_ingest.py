import pandas as pd
from pathlib import Path

def _read_any(path: Path, columns=None) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype=str, usecols=columns)
    elif path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    else:
        raise ValueError(f"Unsupported file type: {path}")

# Example usage in CLI:
# df_opt  = _read_any(args.option_file)
# df_watch= _read_any(args.watch_file)
# pred_files = list(Path(args.pred_dir).glob("*predictions*.parquet"))
# df_pred = pd.concat([_read_any(f) for f in pred_files], ignore_index=True).drop_duplicates(subset=["symbol", "horizon"])
# df_full.to_parquet(cache_path, index=False)
