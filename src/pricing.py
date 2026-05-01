# pricing.py
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad

# Constants
S = 86.71
r = 0.0369
q = 0.0453
HESTON_PARAMS = {
    'kappa': 4.5704,
    'theta': 0.0223,
    'sigma': 0.9920,
    'rho':  -0.1925,
    'v0':    0.0118
}


def d_calculator(S, K, T, r, q, sigma):
    """ This function creates the d_1 and d_2 values needed for Black-Scholes.
    Args:
        S = Spot price
        K = Strike
        T = Time to maturity (in years)
        r = risk-free rate
        q = annualized dividend yield
        sigma = volatility
    Returns:
        d_1 and d_2 which are components used in BS.
    """
    # Need to check for edge cases, important for calibration later
    if sigma <= 0 or T <= 0:
        return np.nan, np.nan
    
    d_1 = (np.log(S/K) + (r - q + (sigma**2 / 2))*T) / ((sigma * np.sqrt(T)))
    d_2 = d_1 - sigma * np.sqrt(T)

    return d_1, d_2


def bs_call(S, K, T, r, q, sigma):
    """ This function prices a call option using the BS formula.
    Args:
        S = Spot price
        K = Strike
        T = Time to maturity (in years)
        r = risk-free rate
        q = annualized dividend yield
        sigma = volatility
    Returns:
        Call price.
    """
    # Use d_calculator() to create d_1 and d_2
    d_1, d_2 = d_calculator(S, K, T, r, q, sigma)

    # Price call using BS formula
    c = (S * np.exp(-q * T) * norm.cdf(d_1)) - (K * np.exp(-r*T) * norm.cdf(d_2))

    return c


def bs_put(S, K, T, r, q, sigma):
    """ This function prices a put option using the BS formula.
    Args:
        S = Spot price
        K = Strike
        T = Time to maturity (in years)
        r = risk-free rate
        q = annualized dividend yield
        sigma = volatility
    Returns:
        Put price.
    """
    # Use d_calculator() to create d_1 and d_2
    d_1, d_2 = d_calculator(S, K, T, r, q, sigma)

    # Price put using BS formula
    p = (K * np.exp(-r * T) * norm.cdf(-d_2)) - (S * np.exp(-q * T) * norm.cdf(-d_1))

    return p


def bs_iv(market_mid, S, K, T, r, q, flag):
    """
    Inverts BS to find IV given a market mid-price. Uses Brent's method of root finding.
    flag: 'Call' for call, 'Put' for put
    """
    if flag == 'Call':
        pricer = bs_call
    else:
        pricer = bs_put

    # Simple lambda function representing f(sigma) = Price - market_mid
    objective = lambda sigma: pricer(S, K, T, r, q, sigma) - market_mid
    
    try:
        iv = brentq(objective, 1e-4, 5.0)  # search vol between 0.01% and 500%
    except ValueError:
        iv = np.nan  # root not bracketed (bad quote, skip it)
    
    return iv


def heston_cf(u, S, T, r, q, kappa, theta, sigma, rho, v0):
    """
    Heston (1993) characteristic function of ln(S_T) under risk-neutral measure.
    Args:
        u     = integration variable (complex-valued)
        S     = spot price
        T     = time to maturity (in years)
        r     = risk-free rate
        q     = dividend yield
        kappa = mean reversion speed of variance
        theta = long-run variance (long-run vol = sqrt(theta))
        sigma = vol of vol
        rho   = spot-vol correlation (expected negative for TLT)
        v0    = initial variance (current vol = sqrt(v0))
    Returns:
        Complex value of the characteristic function at u
    """
    i = 1j

    # d: analogous to the BS d1/d2 denominator, but complex-valued
    # Captures how fast the variance process mean-reverts under complex rotation
    d = np.sqrt((kappa - rho * sigma * i * u)**2 + sigma**2 * (u**2 + i * u))

    # g: ratio used to simplify the time-dependent terms below
    g = (kappa - rho * sigma * i * u - d) / (kappa - rho * sigma * i * u + d)

    exp_dT = np.exp(-d * T)

    # C: contribution from the drift and long-run variance (kappa, theta)
    C = (r - q) * i * u * T + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * i * u - d) * T
        - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )

    # D: contribution from the initial variance v0
    D = ((kappa - rho * sigma * i * u - d) / sigma**2) * (
        (1 - exp_dT) / (1 - g * exp_dT)
    )

    return np.exp(C + D * v0 + i * u * np.log(S))


def heston_call(S, K, T, r, q, kappa, theta, sigma, rho, v0):
    """
    Prices a European call using the Heston (1993) model via 
    characteristic function inversion.
    Args:
        S     = spot price
        K     = strike price
        T     = time to maturity (in years)
        r     = risk-free rate
        q     = dividend yield
        kappa = mean reversion speed of variance
        theta = long-run variance
        sigma = vol of vol
        rho   = spot-vol correlation
        v0    = initial variance
    Returns:
        Call price
    """
    if T <= 0:
        return max(S * np.exp(-q * T) - K, 0.0)

    # Forward price: used to normalize the P1 integrand
    F = S * np.exp((r - q) * T)

    # P2: risk-neutral probability that S_T > K
    # Uses standard CF: phi(u)
    integrand_P2 = lambda u: np.real(
        np.exp(-1j * u * np.log(K))
        * heston_cf(u, S, T, r, q, kappa, theta, sigma, rho, v0)
        / (1j * u)
    )

    # P1: stock-measure probability that S_T > K
    # Uses CF shifted by -i: phi(u - i), normalized by forward F
    # The shift by -i changes the measure from risk-neutral to stock measure
    integrand_P1 = lambda u: np.real(
        np.exp(-1j * u * np.log(K))
        * heston_cf(u - 1j, S, T, r, q, kappa, theta, sigma, rho, v0)
        / (1j * u * F)
    )

    P1 = 0.5 + (1 / np.pi) * quad(integrand_P1, 1e-9, 100, limit=200, epsabs=1e-6, epsrel=1e-6)[0]
    P2 = 0.5 + (1 / np.pi) * quad(integrand_P2, 1e-9, 100, limit=200, epsabs=1e-6, epsrel=1e-6)[0]

    return S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2


def heston_put(S, K, T, r, q, kappa, theta, sigma, rho, v0):
    """
    Prices a European put using the Heston (1993) model.
    Derived from heston_call via put-call parity — no additional
    integration needed.

    Args: (same as heston_call)
    Returns:
        Put price
    """
    if T <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    c = heston_call(S, K, T, r, q, kappa, theta, sigma, rho, v0)

    return c - S * np.exp(-q * T) + K * np.exp(-r * T)


def heston_vega(S, K, T, r, q, kappa, theta, sigma, rho, v0, eps=1e-4):
    """
    Returns the option's Vega using the heston call pricer and a finite difference scheme (central difference).
    Can be used for both puts and calls due to put-call parity
    """
    price_up   = heston_call(S, K, T, r, q, kappa, theta, sigma, rho, v0 + eps)
    price_down = heston_call(S, K, T, r, q, kappa, theta, sigma, rho, v0 - eps)

    return (price_up - price_down) / (2 * eps)

def bs_delta(S, K, t, r, q, sigma, flag):
    """
    BS delta for hedging. Uses Merton continuous dividend extension.
    flag: 'Call' or 'Put'
    """
    d1, _ = d_calculator(S, K, t, r, q, sigma)
    if np.isnan(d1):
        return np.nan
    if flag == 'Call':
        return np.exp(-q * t) * norm.cdf(d1)
    else:
        return -np.exp(-q * t) * norm.cdf(-d1)

def heston_iv(S, K, T, r, q, kappa, theta, sigma, rho, v0, flag):
    """
    Computes Heston implied vol by pricing with calibrated Heston params,
    then inverting through BS to express in IV terms.
    """
    if flag == 'Call':
        heston_price = heston_call(S, K, T, r, q, kappa, theta, sigma, rho, v0)
    else:
        heston_price = heston_put(S, K, T, r, q, kappa, theta, sigma, rho, v0)

    return bs_iv(heston_price, S, K, T, r, q, flag)