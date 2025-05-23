# Midday Workflow (Unity)

1. Export TOS CSVs (watchlist and option chain) to your computer.
2. Run:

   python src/midday_ingest.py \
      --watchlist ~/Downloads/U_watchlist.csv \
      --options   ~/Downloads/U_chain.csv \
      --out       data/factors/u_midday.json

3. Paste the JSON + prompt into ChatGPT.

That’s it. All other automation, news, and multi-ticker code is archived in `old/`.
