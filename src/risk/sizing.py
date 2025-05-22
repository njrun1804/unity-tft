"""Given a cert_series ∈ [0,1], implement stepwise sizing: 0-0.2 ⇒ 0.5× base, 0.2-0.5 ⇒ 1×, 0.5-0.8 ⇒ 1.5×, >0.8 ⇒ 2×.  Enforce β-net-Δ guardrails from config."""
def sizing_ladder(cert_series, base_size, config):
    """
    Map certainty to position size using config-driven ladder and enforce max_single_name_weight.
    """
    sizes = []
    for cert in cert_series:
        if cert < 0.2:
            scale = 0.5
        elif cert < 0.5:
            scale = 1.0
        elif cert < 0.8:
            scale = 1.5
        else:
            scale = 2.0
        size = min(base_size * scale, config.get("max_single_name_weight", 1.0))
        sizes.append(size)
    return sizes

def compute_target_deltas(cert_series, exposures, config):
    """
    Compute target deltas for a portfolio given certainties and current exposures.
    Enforces portfolio_delta_target from config.
    """
    target = config.get("portfolio_delta_target", 0.25)
    deltas = []
    for cert, exp in zip(cert_series, exposures):
        # Use sizing ladder to get size, then scale to target delta
        size = sizing_ladder([cert], 1.0, config)[0]
        delta = min(size, target - exp)
        deltas.append(delta)
    return deltas

def enforce_risk_off(deltas, risk_scores, config):
    """
    Enforce risk-off guardrails: if risk_score exceeds threshold, reduce or zero out delta.
    Returns adjusted deltas and a mask of risk-off positions.
    """
    threshold = config.get("risk_off_thresholds", {}).get("risk_score_p95", 0.8)
    adjusted = []
    mask = []
    for delta, risk in zip(deltas, risk_scores):
        if risk > threshold:
            adjusted.append(0.0)
            mask.append(True)
        else:
            adjusted.append(delta)
            mask.append(False)
    return adjusted, mask
