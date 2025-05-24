import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_positions_config(config_path=None):
    """
    Load the positions.json config file.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "positions.json"
    with open(config_path, "r") as f:
        return json.load(f)


def get_current_positions(config):
    """
    Extract current positions from the config file.
    """
    portfolio = config.get("prompt_config", {}).get("portfolio", {})
    equity_positions = portfolio.get("equity", {}).get("positions", {})
    options_positions = portfolio.get("options", {}).get("positions", {})
    
    return {
        "equity": equity_positions,
        "options": options_positions,
        "cash": portfolio.get("cash", 0)
    }


def calculate_portfolio_metrics(current_positions, predictions_df, stock_price=None):
    """
    Calculate comprehensive portfolio metrics including Greeks, Sharpe ratios, and risk metrics.
    """
    portfolio_delta = 0.0
    portfolio_gamma = 0.0
    portfolio_theta = 0.0
    portfolio_vega = 0.0
    portfolio_value = 0.0
    total_contracts = 0
    stock_exposure = 0.0
    
    # Calculate from current options positions
    for pos_id, position in current_positions["options"].items():
        contracts = position.get("contracts", 0)
        delta = position.get("delta", 0)
        gamma = position.get("gamma", 0)
        theta = position.get("theta", 0)
        vega = position.get("vega", 0)
        market_value = position.get("market_value", 0)
        
        # Apply Greeks for portfolio exposure (100 shares per contract)
        portfolio_delta += contracts * delta * 100
        portfolio_gamma += contracts * gamma * 100
        portfolio_theta += contracts * theta * 100
        portfolio_vega += contracts * vega * 100
        portfolio_value += market_value
        total_contracts += abs(contracts)
    
    # Add equity positions
    for symbol, position in current_positions["equity"].items():
        shares = position.get("shares", 0)
        price = position.get("current_price", position.get("cost_basis", stock_price or 0))
        
        # Each stock share has delta of 1.0
        portfolio_delta += shares
        stock_exposure += shares * price
        portfolio_value += shares * price
    
    # Calculate additional risk metrics
    total_exposure = portfolio_value + current_positions["cash"]
    leverage = portfolio_value / max(total_exposure, 1) if total_exposure > 0 else 0
    
    # Estimate portfolio volatility based on Greeks exposure
    portfolio_vol = abs(portfolio_gamma) * 0.1 + abs(portfolio_vega) * 0.001  # Greeks-based volatility estimate
    
    return {
        "portfolio_delta": portfolio_delta,
        "portfolio_gamma": portfolio_gamma,
        "portfolio_theta": portfolio_theta,
        "portfolio_vega": portfolio_vega,
        "portfolio_value": portfolio_value,
        "stock_exposure": stock_exposure,
        "total_contracts": total_contracts,
        "leverage": leverage,
        "estimated_volatility": portfolio_vol,
        "cash": current_positions["cash"],
        "total_capital": total_exposure
    }


def calculate_sharpe_metrics(predictions_df, portfolio_metrics, risk_free_rate=0.05):
    """
    Calculate Sharpe ratio and risk-adjusted return metrics for position sizing.
    """
    if predictions_df.empty:
        return {"sharpe_ratio": 0, "expected_return": 0, "portfolio_vol": 0.1}
    
    # Extract expected returns from predictions
    expected_returns = predictions_df.get('prediction', pd.Series([0]))
    confidences = predictions_df.get('confidence', pd.Series([0.5]))
    
    # Calculate risk-adjusted expected return
    risk_adjusted_returns = expected_returns * confidences
    expected_return = risk_adjusted_returns.mean() if len(risk_adjusted_returns) > 0 else 0
    
    # Estimate portfolio volatility
    portfolio_vol = max(portfolio_metrics.get("estimated_volatility", 0.1), 0.05)
    
    # Calculate Sharpe ratio
    excess_return = expected_return - risk_free_rate
    sharpe_ratio = excess_return / portfolio_vol if portfolio_vol > 0 else 0
    
    return {
        "sharpe_ratio": sharpe_ratio,
        "expected_return": expected_return,
        "portfolio_vol": portfolio_vol,
        "excess_return": excess_return
    }


def calculate_optimal_position_size(
    prediction: float, 
    confidence: float,
    delta: float, 
    portfolio_metrics: Dict, 
    risk_params: Dict,
    current_stock_price: float = 20.0
) -> Tuple[int, str]:
    """
    Calculate optimal position size using Kelly Criterion with portfolio constraints.
    
    Returns: (contracts, reasoning)
    """
    max_position_pct = risk_params.get("portfolio_limits", {}).get("max_position_pct", 0.15)
    kelly_enabled = risk_params.get("kelly_criterion", {}).get("enabled", True)
    kelly_max_fraction = risk_params.get("kelly_criterion", {}).get("max_fraction", 0.25)
    kelly_safety = risk_params.get("kelly_criterion", {}).get("safety_factor", 0.5)
    
    available_capital = portfolio_metrics["cash"]
    
    # Basic position sizing based on confidence and prediction strength
    base_size = 1
    
    if kelly_enabled and confidence > 0.6:
        # Kelly Criterion calculation
        win_prob = confidence
        avg_win = abs(prediction) if prediction > 0 else 0.5
        avg_loss = 0.5  # Estimated average loss
        
        # Kelly fraction = (win_prob * avg_win - loss_prob * avg_loss) / avg_win
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction * kelly_safety, kelly_max_fraction))
        
        # Convert to position size
        option_cost = current_stock_price * abs(delta) * 100  # Rough option cost estimate
        max_capital_risk = available_capital * kelly_fraction
        kelly_size = int(max_capital_risk / max(option_cost, 100))  # Minimum $100 per contract
        
        base_size = max(1, min(kelly_size, 5))  # Cap at 5 contracts
        reasoning = f"Kelly-optimized size: {kelly_fraction:.1%} allocation"
    else:
        reasoning = "Conservative sizing: Kelly disabled or low confidence"
    
    # Apply portfolio constraints
    position_value = base_size * current_stock_price * abs(delta) * 100
    max_position_value = available_capital * max_position_pct
    
    if position_value > max_position_value:
        constrained_size = int(max_position_value / (current_stock_price * abs(delta) * 100))
        base_size = max(1, constrained_size)
        reasoning += f" | Capped by {max_position_pct:.0%} limit"
    
    return base_size, reasoning


def recommend_positions(predictions, certainty, config=None, stock_price=20.75):
    """
    Enhanced position recommendations with comprehensive portfolio delta management 
    and Sharpe ratio optimization for both options and stock positions.
    
    Features:
    1. Combined portfolio delta management (stocks + options)
    2. Sharpe ratio optimization for position sizing
    3. Kelly Criterion for optimal capital allocation
    4. Risk-adjusted stock position recommendations
    5. Greeks-based portfolio balancing
    """
    if config is None:
        config = load_positions_config()
    
    # Get configuration parameters
    prompt_config = config.get("prompt_config", {})
    meta = prompt_config.get("meta", {})
    risk_mgmt = config.get("risk_management", {})
    
    # Portfolio targets from config
    delta_target_range = meta.get("portfolio_delta_target", [0.6, 0.8])
    target_delta_lo, target_delta_hi = delta_target_range
    
    # Greeks limits from risk management
    greeks_limits = risk_mgmt.get("greeks_limits", {})
    portfolio_delta_limits = greeks_limits.get("portfolio_delta", {"min": 0.6, "max": 0.8, "target": 0.7})
    portfolio_gamma_limits = greeks_limits.get("portfolio_gamma", {"min": -0.1, "max": 0.1})
    portfolio_vega_limits = greeks_limits.get("portfolio_vega", {"max": 1000})
    portfolio_theta_limits = greeks_limits.get("portfolio_theta", {"min": -500})
    
    # Risk limits
    max_positions = risk_mgmt.get("portfolio_limits", {}).get("max_positions", 10)
    max_position_pct = risk_mgmt.get("portfolio_limits", {}).get("max_position_pct", 0.15)
    
    # Get current positions and calculate comprehensive metrics
    current_positions = get_current_positions(config)
    portfolio_metrics = calculate_portfolio_metrics(current_positions, predictions, stock_price)
    
    # Calculate Sharpe metrics for portfolio optimization
    sharpe_metrics = calculate_sharpe_metrics(predictions, portfolio_metrics)
    
    # Convert predictions to DataFrame if it isn't already
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)
    
    # Filter for meaningful options
    meaningful_options = predictions[
        (predictions['delta'].abs() > 0.01) |  # Has meaningful delta OR
        (predictions['bid'] > 0.01) |          # Has actual bid OR  
        (predictions['ask'] > 0.01) |          # Has actual ask OR
        (predictions['oi'] > 10)               # Has open interest
    ].copy()
    
    print(f"Filtered from {len(predictions)} to {len(meaningful_options)} meaningful options")
    
    recommendations = {}
    current_delta = portfolio_metrics["portfolio_delta"]
    target_delta = portfolio_delta_limits.get("target", 0.7)
    delta_gap = target_delta - current_delta
    
    # === OPTIONS POSITION RECOMMENDATIONS FIRST ===
    # Generate options recommendations first, then calculate stock needs based on total portfolio delta
    if len(meaningful_options) == 0:
        message = "No tradeable options found - all have zero delta/bid/ask"
        if "STOCK_U" not in recommendations:
            print("No meaningful options available for new positions")
        else:
            print(f"No meaningful options, but recommending stock position: {recommendations['STOCK_U']['action']} {recommendations['STOCK_U']['shares']} shares")
    else:
        # Enhanced options recommendations with Sharpe optimization
        if portfolio_metrics["total_contracts"] == 0:
            # New positions: Focus on high Sharpe ratio opportunities
            high_conf_threshold = 0.75
            
            # Handle confidence data
            if isinstance(certainty, (int, float)):
                meaningful_options['confidence'] = meaningful_options.get('confidence', certainty)
            
            # Calculate risk-adjusted scores incorporating Sharpe metrics
            meaningful_options['sharpe_score'] = (
                meaningful_options['confidence'] * 0.4 +
                meaningful_options['prediction'].clip(0, 2) * 0.3 +
                meaningful_options['delta'].abs() * 2.0 * 0.3  # Prefer meaningful deltas
            ) * (1 + sharpe_metrics["sharpe_ratio"])  # Boost by portfolio Sharpe
            
            # Filter and sort by enhanced scoring
            high_conf_predictions = meaningful_options[
                meaningful_options['confidence'] >= high_conf_threshold
            ].copy()
            
            if len(high_conf_predictions) == 0:
                high_conf_predictions = meaningful_options[
                    meaningful_options['confidence'] >= 0.65
                ].copy()
            
            high_conf_predictions = high_conf_predictions.sort_values('sharpe_score', ascending=False)
            
            # Intelligent position selection based on portfolio needs
            max_new_positions = min(3, max_positions)
            selected_positions = []
            
            for idx, row in high_conf_predictions.iterrows():
                if len(selected_positions) >= max_new_positions:
                    break
                    
                option_delta = row.get('delta', 0)
                prediction = row.get('prediction', 0)
                confidence = row.get('confidence', 0.5)
                
                # Portfolio-aware position sizing
                optimal_size, size_reasoning = calculate_optimal_position_size(
                    prediction, confidence, option_delta, portfolio_metrics, risk_mgmt, stock_price
                )
                
                # Determine action based on prediction, delta gap, and Greeks limits
                if prediction > 1.2 and confidence > 0.75:
                    if delta_gap > 0 and option_delta > 0:  # Need more delta, option provides it
                        action = "BUY"
                        position_reasoning = f"Strong bullish + delta alignment (gap: {delta_gap:.0f})"
                    elif delta_gap < 0 and option_delta < 0:  # Need less delta, put option
                        action = "BUY" 
                        position_reasoning = f"Bearish hedge + delta alignment (gap: {delta_gap:.0f})"
                    else:
                        action = "BUY"
                        position_reasoning = f"Strong signal (pred: {prediction:.2f}, conf: {confidence:.1%})"
                elif prediction < 0.8 and confidence > 0.7:
                    action = "SELL"  # Consider short strategies
                    position_reasoning = f"Bearish signal - short strategy (pred: {prediction:.2f})"
                else:
                    continue  # Skip borderline signals
                
                position_id = f"{row.get('ticker', 'U')}_{row.get('strike', idx):.1f}"
                
                recommendations[position_id] = {
                    "action": action,
                    "size": optimal_size,
                    "certainty": round(confidence, 3),
                    "predicted_move": round(prediction, 3),
                    "strike": float(row.get("strike", 0)),
                    "ticker": row.get("ticker", "U"),
                    "delta": round(option_delta, 4),
                    "gamma": round(row.get("gamma", 0), 4),
                    "theta": round(row.get("theta", 0), 4),
                    "vega": round(row.get("vega", 0), 4),
                    "bid": round(row.get("bid", 0), 3),
                    "ask": round(row.get("ask", 0), 3),
                    "iv": round(row.get("iv", 0), 3),
                    "oi": int(row.get("oi", 0)),
                    "sharpe_score": round(row.get("sharpe_score", 0), 3),
                    "reasoning": f"{position_reasoning} | {size_reasoning}",
                    "portfolio_delta_impact": optimal_size * option_delta * 100,
                    "estimated_cost": optimal_size * stock_price * abs(option_delta) * 100
                }
                
                selected_positions.append(position_id)
        
        else:
            # Manage existing positions with enhanced portfolio balancing
            print(f"Managing {portfolio_metrics['total_contracts']} existing positions with portfolio delta management")
            
            for pos_id, position in current_positions["options"].items():
                strike = position.get("strike")
                ticker = position.get("ticker", "U")
                
                matching_rows = meaningful_options[
                    (meaningful_options["ticker"] == ticker) & 
                    (meaningful_options["strike"] == strike)
                ]
                
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    confidence = row.get('confidence', 0.5)
                    prediction = row.get('prediction', 0)
                    current_contracts = position.get("contracts", 0)
                    current_delta = position.get("delta", 0)
                    
                    # Portfolio-aware position management
                    position_delta_contribution = current_contracts * current_delta * 100
                    
                    # Enhanced logic considering portfolio Greeks and Sharpe
                    if prediction > 1.5 and confidence > 0.8:
                        if current_contracts > 0 and delta_gap > 0:
                            action = "INCREASE"
                            size = min(abs(current_contracts) + 1, 3)
                        elif current_contracts < 0:  # Close shorts on strong bullish
                            action = "CLOSE"
                            size = 0
                        else:
                            action = "HOLD"
                            size = current_contracts
                    elif prediction < 0.5 and confidence > 0.75:
                        if current_contracts > 0 and delta_gap < 0:
                            action = "DECREASE"
                            size = max(current_contracts - 1, 0)
                        elif current_contracts < 0 and delta_gap < 0:
                            action = "INCREASE"
                            size = min(abs(current_contracts) + 1, 2)
                        else:
                            action = "CLOSE"
                            size = 0
                    elif confidence < 0.6:
                        action = "CLOSE"
                        size = 0
                    else:
                        action = "HOLD" 
                        size = current_contracts
                    
                    recommendations[pos_id] = {
                        "action": action,
                        "size": size,
                        "current_size": current_contracts,
                        "certainty": round(confidence, 3),
                        "predicted_move": round(prediction, 3),
                        "strike": float(strike),
                        "ticker": ticker,
                        "delta": round(row.get("delta", 0), 4),
                        "current_delta": round(current_delta, 4),
                        "portfolio_delta_contribution": round(position_delta_contribution, 1),
                        "reasoning": f"Portfolio-managed: {action.lower()} (pred: {prediction:.2f}, portfolio δ gap: {delta_gap:.0f})"
                    }
                else:
                    recommendations[pos_id] = {
                        "action": "CLOSE",
                        "size": 0,
                        "current_size": position.get("contracts", 0),
                        "reasoning": "Position not found in current market data - recommend close"
                    }
    
    # === STOCK POSITION RECOMMENDATIONS ===
    # Calculate theoretical portfolio delta after applying recommended options positions
    theoretical_portfolio_delta = current_delta
    
    # Add delta impact from recommended options positions
    for pos_id, rec in recommendations.items():
        if not pos_id.startswith("_") and "portfolio_delta_impact" in rec:
            theoretical_portfolio_delta += rec["portfolio_delta_impact"]
    
    # Calculate delta gap based on theoretical portfolio after options recommendations
    theoretical_delta_gap = target_delta - theoretical_portfolio_delta
    current_stock_shares = sum(pos.get("shares", 0) for pos in current_positions["equity"].values())
    
    # Debug output for stock recommendation logic
    print(f"DEBUG: Stock recommendation analysis:")
    print(f"  Current stock shares: {current_stock_shares}")
    print(f"  Original portfolio delta: {current_delta}")
    print(f"  Theoretical portfolio delta (after options): {theoretical_portfolio_delta}")
    print(f"  Target delta: {target_delta}")
    print(f"  Theoretical delta gap: {theoretical_delta_gap}")
    print(f"  Abs theoretical delta gap: {abs(theoretical_delta_gap)}")
    print(f"  Should trigger stock rec (>50): {abs(theoretical_delta_gap) > 50}")
    
    # Recommend stock position adjustments based on theoretical portfolio delta target
    if abs(theoretical_delta_gap) > 50:  # Significant delta gap after options recommendations
        if theoretical_delta_gap > 0:  # Need more delta
            stock_action = "BUY"
            stock_shares = min(int(theoretical_delta_gap * 0.3), 100)  # Use 30% of gap, cap at 100 shares
            stock_reasoning = f"Increase delta exposure: theoretical portfolio δ={theoretical_portfolio_delta:.0f}, target={target_delta:.0f}"
            print(f"  → BUY recommendation: {stock_shares} shares")
        else:  # Need less delta  
            # If we have existing shares, sell some of them
            if current_stock_shares > 0:
                stock_action = "SELL"
                stock_shares = min(int(abs(theoretical_delta_gap) * 0.3), current_stock_shares, 100)
                stock_reasoning = f"Reduce delta exposure: theoretical portfolio δ={theoretical_portfolio_delta:.0f}, target={target_delta:.0f}"
                print(f"  → SELL recommendation: {stock_shares} shares (from existing {current_stock_shares})")
            else:
                # No existing shares to sell, but we could short sell if allowed
                # For now, we'll skip stock recommendation when we can't sell
                stock_action = "SKIP"
                stock_shares = 0
                stock_reasoning = f"Need to reduce delta but no existing shares to sell (theoretical δ={theoretical_portfolio_delta:.0f}, target={target_delta:.0f})"
                print(f"  → SKIP stock recommendation: need to reduce delta but no existing shares")
        
        if stock_shares > 0 and stock_action != "SKIP":
            print(f"  → Adding STOCK_U recommendation to output")
            recommendations["STOCK_U"] = {
                "action": stock_action,
                "shares": stock_shares,
                "type": "STOCK",
                "current_price": stock_price,
                "estimated_cost": stock_shares * stock_price,
                "delta_impact": stock_shares if stock_action == "BUY" else -stock_shares,
                "reasoning": stock_reasoning,
                "sharpe_contribution": sharpe_metrics["sharpe_ratio"] * 0.5  # Conservative Sharpe for stocks
            }
        else:
            print(f"  → No stock recommendation: stock_shares = {stock_shares}, action = {stock_action}")
    else:
        print(f"  → No stock recommendation: theoretical delta gap too small ({abs(theoretical_delta_gap):.1f} <= 50)")
    
    # Add portfolio summary
    recommendations["_portfolio_summary"] = {
        "current_delta": round(portfolio_metrics["portfolio_delta"], 2),
        "target_delta_range": delta_target_range,
        "current_positions": portfolio_metrics["total_contracts"],
        "available_cash": portfolio_metrics["cash"],
        "portfolio_value": round(portfolio_metrics["portfolio_value"], 2),
        "meaningful_options_available": len(meaningful_options),
        "total_options_scanned": len(predictions),
        "recommendations_generated": len([r for r in recommendations.keys() if not r.startswith("_")])
    }
    
    return recommendations


def save_recommendations(recommendations, out_path):
    """Save recommendations to JSON file."""
    with open(out_path, "w") as f:
        json.dump(recommendations, f, indent=2)
