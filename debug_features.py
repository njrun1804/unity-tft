import pandas as pd, json, pathlib, pprint

# 1️⃣ canonical list (nine features the checkpoint expects)
FEATURES = json.load(open("artifacts/feature_list.json"))

# 2️⃣ grab the most-recent option-chain parquet
latest = max(pathlib.Path("data/feature_store/tos_option_chain").rglob("*.parquet"))
print(f"Attempting to read Parquet file: {latest}")
df = pd.read_parquet(latest)
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
if df.empty:
    print("DataFrame is empty! No rows to check.")
    exit()

present  = set(df.columns)
missing  = [c for c in FEATURES if c not in present]
extra    = sorted(present - set(FEATURES))

print(f"\nFILE → {latest}")
print("columns in file:")
pprint.pp(sorted(df.columns.tolist()))
print("\nmissing features:", missing)
print("extra numeric-like columns that snuck in:", extra)

# quick dtype glance
print("\n--- dtypes ---")
print(df[FEATURES].dtypes, end="\n\n")
