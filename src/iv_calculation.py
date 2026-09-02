"""
Implied volatility by inverting Black-76.

Prices are quoted against the forward, not spot, so the pricer takes F and the
forward comes from put-call parity (see forward.py). The original version of
this file used the spot form with no dividend yield, which puts the model's
effective forward above the real one and biases call IV down and put IV up:
manufactured skew.

    C = e^(-rT) [F N(d1) - K N(d2)]
    P = e^(-rT) [K N(-d2) - F N(-d1)]
    d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from forward import (DEFAULT_RATE, add_forward_columns, forward_curve,
                     infer_snapshot, year_fraction)

# Parity on the SPY snapshot implies a carry of 2.45%; with SPY's trailing yield
# near 1.15% that puts r at roughly 3.6%. The forward itself is read from the
# market and does not depend on this, but the discount factor does.
RISK_FREE_RATE = DEFAULT_RATE

# What the first version of this project assumed, kept only so the "before"
# case in the README and the before/after figure reproduce the actual before
# rather than a flattering approximation of it.
LEGACY_RATE = 0.05

IV_LOWER_BOUND = 1e-6
IV_UPPER_BOUND = 10.0
MIN_TIME_TO_EXPIRY = 1e-6

# Newton is abandoned below this vega. Deep out of the money, vega collapses and
# the Newton step divides by nearly nothing.
VEGA_FLOOR = 1e-8
MAX_NEWTON_ITER = 12

# Relative, not absolute. An absolute price tolerance is meaningless for an
# option worth 1e-12 of the forward: the seed clamp alone would satisfy it and
# Newton would report the clamp as a solved vol.
NEWTON_REL_TOL = 1e-10


def _d1_d2(F, K, T, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return d1, d1 - sigma * sqrt_T


def black76_price(F: float, K: float, T: float, r: float, sigma: float,
                  option_type: str) -> float:
    """Black-76 price of a European option struck against forward *F*.

    Parameters
    ----------
    F : float
        Forward price for the option's expiry.
    K, T, r, sigma : float
        Strike, time to expiry in years, continuously-compounded rate, and
        annualised volatility.
    option_type : str
        ``'call'`` or ``'put'``.

    Returns
    -------
    float
        Theoretical price, or ``np.nan`` if the inputs are degenerate.
    """
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return np.nan

    d1, d2 = _d1_d2(F, K, T, sigma)
    discount = np.exp(-r * T)

    if option_type == "call":
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def black76_vega(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Sensitivity of price to volatility, e^(-rT) F phi(d1) sqrt(T).

    Same for calls and puts. Largest at the money and collapsing in both wings,
    which is why the Newton path below needs a floor.
    """
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return np.nan
    d1, _ = _d1_d2(F, K, T, sigma)
    return np.exp(-r * T) * F * norm.pdf(d1) * np.sqrt(T)


def black76_delta(F: float, K: float, T: float, r: float, sigma: float,
                  option_type: str) -> float:
    """Forward delta: the derivative with respect to F, not to spot.

    Reported as ``delta_forward`` downstream. The distinction matters for any
    claim about a "25-delta" strike, since spot delta differs by e^(rT).
    """
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return np.nan
    d1, _ = _d1_d2(F, K, T, sigma)
    discount = np.exp(-r * T)
    if option_type == "call":
        return discount * norm.cdf(d1)
    elif option_type == "put":
        return -discount * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str, q: float = 0.0) -> float:
    """Spot-argument Black-Scholes, as a wrapper on the forward form.

    Kept because it is the formula the README quotes and the one the textbook
    check in validation.py tests against. With q=0 it reduces exactly to
    S N(d1) - K e^(-rT) N(d2).
    """
    if T <= 0 or S <= 0:
        return np.nan
    return black76_price(S * np.exp((r - q) * T), K, T, r, sigma, option_type)


def implied_volatility(market_price: float, F: float, K: float, T: float,
                       r: float, option_type: str, method: str = "auto"):
    """Invert Black-76 for volatility.

    Newton with analytic vega is the fast path, Brent the fallback. Brent cannot
    diverge: Black-76 is monotonic in sigma, so a sign change over the bracket
    guarantees a root. Newton is faster but misbehaves where vega is near zero,
    which is exactly the deep wings this surface cares about, so it is used only
    where it is safe and its result is always inside the Brent bracket.

    Parameters
    ----------
    market_price : float
        Observed price, here the bid-ask mid.
    F : float
        Forward for the expiry, from put-call parity.
    K, T, r : float
        Strike, time to expiry in years, discount rate.
    option_type : str
        ``'call'`` or ``'put'``.
    method : str
        ``'auto'`` for Newton-then-Brent, ``'brent'`` to force the bracketed
        solver, ``'newton'`` to force the fast path with no fallback.

    Returns
    -------
    tuple
        ``(iv, how)`` where *how* is ``'newton'``, ``'brent'`` or ``''`` when
        no finite IV exists. IV is a decimal, so 0.20 is 20%.
    """
    if not np.isfinite(market_price) or market_price <= 0:
        return (np.nan, "")
    if not np.isfinite(F) or F <= 0 or K <= 0 or T < MIN_TIME_TO_EXPIRY:
        return (np.nan, "")

    discount = np.exp(-r * T)
    if option_type == "call":
        intrinsic = discount * max(F - K, 0.0)
        ceiling = discount * F
    else:
        intrinsic = discount * max(K - F, 0.0)
        ceiling = discount * K

    # Below intrinsic or above the undiscounted payoff cap there is no sigma
    # that reproduces the price.
    if market_price <= intrinsic or market_price >= ceiling:
        return (np.nan, "")

    def objective(sigma):
        return black76_price(F, K, T, r, sigma, option_type) - market_price

    if method in ("auto", "newton"):
        sigma = _newton_iv(market_price, F, K, T, r, option_type)
        if sigma is not None:
            return (sigma, "newton")
        if method == "newton":
            return (np.nan, "")

    try:
        if objective(IV_LOWER_BOUND) * objective(IV_UPPER_BOUND) > 0:
            return (np.nan, "")
        iv = brentq(objective, IV_LOWER_BOUND, IV_UPPER_BOUND,
                    xtol=1e-10, maxiter=500)
        return (iv, "brent")
    except (ValueError, RuntimeError):
        return (np.nan, "")


def _newton_seed(market_price, F, K, T):
    """Starting volatility for Newton.

    Brenner-Subrahmanyam, sqrt(2 pi / T) * price / F, is an at-the-money
    approximation. Out of the money the price is small, so on its own it seeds
    at a couple of vol points when the answer is 27, and vega at that sigma is
    zero to machine precision: Newton bails before it starts.

    So floor it at the volatility that makes total variance of order 2|k|,
    which is where sigma sqrt(T) is comparable to the log-moneyness and d1 sits
    near zero. That is the region where vega is largest, which is the region
    Newton needs to be started in.
    """
    atm = np.sqrt(2.0 * np.pi / T) * market_price / F
    wing = np.sqrt(2.0 * abs(np.log(K / F)) / T) if K != F else 0.0
    return min(max(atm, wing, 0.05), 5.0)


def _newton_iv(market_price, F, K, T, r, option_type):
    """Newton on vega from a moneyness-aware seed, or None if it is unsafe."""
    sigma = _newton_seed(market_price, F, K, T)

    for _ in range(MAX_NEWTON_ITER):
        # Vega first. Where it has collapsed the price carries no information
        # about sigma at all, and the step below would divide by nearly nothing.
        vega = black76_vega(F, K, T, r, sigma)
        if not np.isfinite(vega) or vega < VEGA_FLOOR:
            return None
        diff = black76_price(F, K, T, r, sigma, option_type) - market_price
        if not np.isfinite(diff):
            return None
        if abs(diff) <= NEWTON_REL_TOL * market_price:
            return sigma
        sigma -= diff / vega
        if not np.isfinite(sigma) or sigma <= 1e-4 or sigma >= 5.0:
            return None

    return None


def calculate_iv_surface(df: pd.DataFrame, r: float = RISK_FREE_RATE,
                         method: str = "auto") -> pd.DataFrame:
    """Add implied volatility, forward delta and vega to an options frame.

    Needs a ``forward`` column. If it is absent the forward curve is built here
    from put-call parity, which requires both legs to still be present, so it
    is better done in data_collection before any liquidity filter.
    """
    df = df.copy()
    if "mid_price" not in df.columns:
        df["mid_price"] = (df["bid"] + df["ask"]) / 2

    snapshot = infer_snapshot(df)

    if "forward" not in df.columns:
        spot = float(df["underlying_price"].iloc[0])
        curve = forward_curve(df, spot, r=r, snapshot_ts=snapshot)
        df = add_forward_columns(df, curve)

    if "T" not in df.columns or df["T"].isna().all():
        df["T"] = year_fraction(df["expiration"], snapshot)

    ivs, hows, deltas, vegas = [], [], [], []
    for row in df.itertuples(index=False):
        iv, how = implied_volatility(
            market_price=row.mid_price,
            F=row.forward,
            K=row.strike,
            T=row.T,
            r=r,
            option_type=row.option_type,
            method=method,
        )
        ivs.append(iv)
        hows.append(how)
        if np.isfinite(iv):
            deltas.append(black76_delta(row.forward, row.strike, row.T, r, iv,
                                        row.option_type))
            vegas.append(black76_vega(row.forward, row.strike, row.T, r, iv))
        else:
            deltas.append(np.nan)
            vegas.append(np.nan)

    df["implied_volatility_bs"] = ivs
    df["iv_method"] = hows
    df["delta_forward"] = deltas
    df["vega"] = vegas

    solved = df["implied_volatility_bs"].notna().sum()
    total = len(df)
    fell_back = (df["iv_method"] == "brent").sum()
    print(f"IV solved for {solved}/{total} contracts ({100 * solved / total:.1f} %)")
    print(f"  Newton: {(df['iv_method'] == 'newton').sum()}   "
          f"Brent fallback: {fell_back}")
    return df


def main():
    """Load the most recent cleaned CSV, solve for IV, and save the result."""
    import glob
    import os

    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    csv_files = [f for f in glob.glob(os.path.join(processed_dir, "*.csv"))
                 if "_with_iv" not in f]

    if not csv_files:
        print(f"No CSV files found in {processed_dir}. Run data_collection.py first.")
        return

    # Newest by modification time, not by name. Sorting by name would put
    # spy_options_reference.csv after every timestamped fetch and quietly
    # process the reference instead of the data just downloaded.
    latest = max(csv_files, key=os.path.getmtime)
    print(f"Loading {latest}")
    df = pd.read_csv(latest)

    df_iv = calculate_iv_surface(df)

    print("\n--- IV summary (OTM only) ---")
    otm = df_iv[df_iv["is_otm"]] if "is_otm" in df_iv.columns else df_iv
    for opt_type in ["call", "put"]:
        subset = otm.loc[otm["option_type"] == opt_type, "implied_volatility_bs"].dropna()
        if not subset.empty:
            print(f"{opt_type.capitalize()}s  n: {len(subset)}  "
                  f"mean: {subset.mean():.2%}  median: {subset.median():.2%}")

    out_path = latest.replace(".csv", "_with_iv.csv")
    df_iv.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
