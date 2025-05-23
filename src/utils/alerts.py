def scenario_alerts(samples, rules: dict):
    """
    rules: { "below_strike": {"level": 18, "prob": 0.25},
             "above_takeprofit": {"level": 30, "prob": 0.15} }
    Returns dict of {rule_name: True/False}
    """
    last_prices = samples[:, -1]
    out = {}
    for name, r in rules.items():
        if "below" in name:
            p = (last_prices < r["level"]).mean()
        else:
            p = (last_prices > r["level"]).mean()
        out[name] = p >= r["prob"]
    return out
