"""
Greek calculations for options pricing.

This module provides functions to calculate the Greeks (delta, gamma, theta, vega, rho)
for options using the Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm
from typing import Union


def black_scholes_theta(
    spot_price: Union[float, np.ndarray],
    strike_price: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    option_type: str = 'call'
) -> Union[float, np.ndarray]:
    """
    Calculate theta (time decay) for options using Black-Scholes model.
    
    Theta measures the rate of decline in the value of an option due to the passage of time.
    Result is converted to daily theta by dividing by 365.
    
    Args:
        spot_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (annualized)
        volatility: Volatility of the underlying asset (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Theta value (daily time decay)
    """
    # Convert inputs to numpy arrays for vectorization
    S = np.asarray(spot_price)
    K = np.asarray(strike_price)
    T = np.asarray(time_to_expiry)
    r = np.asarray(risk_free_rate)
    sigma = np.asarray(volatility)
    
    # Handle edge case where time to expiry is zero or negative
    if np.any(T <= 0):
        return np.zeros_like(S) if hasattr(S, '__len__') else 0.0
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Theta calculation (time decay)
    # For calls: -S * norm.pdf(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)
    # For puts: -S * norm.pdf(d1) * sigma / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2)
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:  # put
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    return theta


def black_scholes_delta(
    spot_price: Union[float, np.ndarray],
    strike_price: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    option_type: str = 'call'
) -> Union[float, np.ndarray]:
    """
    Calculate delta for options using Black-Scholes model.
    
    Args:
        spot_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (annualized)
        volatility: Volatility of the underlying asset (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Delta value
    """
    S = np.asarray(spot_price)
    K = np.asarray(strike_price)
    T = np.asarray(time_to_expiry)
    r = np.asarray(risk_free_rate)
    sigma = np.asarray(volatility)
    
    if np.any(T <= 0):
        return np.ones_like(S) if option_type.lower() == 'call' else np.zeros_like(S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type.lower() == 'call':
        return norm.cdf(d1)
    elif option_type.lower() == 'put':
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_gamma(
    spot_price: Union[float, np.ndarray],
    strike_price: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate gamma for options using Black-Scholes model.
    Gamma is the same for calls and puts.
    
    Args:
        spot_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (annualized)
        volatility: Volatility of the underlying asset (annualized)
    
    Returns:
        Gamma value
    """
    S = np.asarray(spot_price)
    K = np.asarray(strike_price)
    T = np.asarray(time_to_expiry)
    r = np.asarray(risk_free_rate)
    sigma = np.asarray(volatility)
    
    if np.any(T <= 0):
        return np.zeros_like(S) if hasattr(S, '__len__') else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def black_scholes_vega(
    spot_price: Union[float, np.ndarray],
    strike_price: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate vega for options using Black-Scholes model.
    Vega is the same for calls and puts.
    
    Args:
        spot_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (annualized)
        volatility: Volatility of the underlying asset (annualized)
    
    Returns:
        Vega value (per 1% change in volatility)
    """
    S = np.asarray(spot_price)
    K = np.asarray(strike_price)
    T = np.asarray(time_to_expiry)
    r = np.asarray(risk_free_rate)
    sigma = np.asarray(volatility)
    
    if np.any(T <= 0):
        return np.zeros_like(S) if hasattr(S, '__len__') else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Return vega per 1% change (divide by 100)
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def black_scholes_rho(
    spot_price: Union[float, np.ndarray],
    strike_price: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    option_type: str = 'call'
) -> Union[float, np.ndarray]:
    """
    Calculate rho for options using Black-Scholes model.
    
    Args:
        spot_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (annualized)
        volatility: Volatility of the underlying asset (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Rho value (per 1% change in interest rate)
    """
    S = np.asarray(spot_price)
    K = np.asarray(strike_price)
    T = np.asarray(time_to_expiry)
    r = np.asarray(risk_free_rate)
    sigma = np.asarray(volatility)
    
    if np.any(T <= 0):
        return np.zeros_like(S) if hasattr(S, '__len__') else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Return rho per 1% change (divide by 100)
    return rho / 100


# Note: Theta calculation implemented above in black_scholes_theta function
# Uses the Black-Scholes formula with daily conversion factor (÷365)

def calculate_option_greeks(
    spot_price: Union[float, np.ndarray],
    strike_price: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    option_type: str = 'call'
) -> dict:
    """
    Calculate all Greeks for an option.
    
    Args:
        spot_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (annualized)
        volatility: Volatility of the underlying asset (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Dictionary containing all Greeks
    """
    return {
        'delta': black_scholes_delta(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type),
        'gamma': black_scholes_gamma(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility),
        'theta': black_scholes_theta(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type),
        'vega': black_scholes_vega(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility),
        'rho': black_scholes_rho(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type)
    }
