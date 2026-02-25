"""
Black-Scholes implied volatility calculation.

Implements the Black-Scholes pricing formula for European calls and puts,
then uses Brent's root-finding method to invert the formula and recover
implied volatility from market mid-prices.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

RISK_FREE_RATE = 0.05       # annualised, approximate
IV_LOWER_BOUND = 1e-6       # smallest sigma passed to brentq
IV_UPPER_BOUND = 10.0       # 1000 % vol — large enough for any real option
MIN_TIME_TO_EXPIRY = 1e-6   # avoid division by zero (fractions of a year)


# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------

def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    """Return the Black-Scholes price of a European option.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Continuously-compounded risk-free rate.
    sigma : float
        Annualised volatility (> 0).
    option_type : str
        ``'call'`` or ``'put'``.

    Returns
    -------
    float
        Theoretical option price, or ``np.nan`` if inputs are invalid.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


# ---------------------------------------------------------------------------
# IV solver
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
) -> float:
    """Solve for implied volatility using Brent's root-finding method.

    Finds the volatility ``sigma`` such that ``black_scholes_price(..., sigma)``
    equals ``market_price``.

    Parameters
    ----------
    market_price : float
        Observed option price (typically the mid of bid and ask).
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Continuously-compounded risk-free rate.
    option_type : str
        ``'call'`` or ``'put'``.

    Returns
    -------
    float
        Implied volatility as a decimal (e.g. 0.20 = 20 %), or ``np.nan``
        if the solver fails or inputs are invalid.
    """
    # Basic sanity checks
    if market_price <= 0 or T < MIN_TIME_TO_EXPIRY or S <= 0 or K <= 0:
        return np.nan

    # Intrinsic value bounds — market price must exceed intrinsic value for a
    # finite IV to exist
    discount = np.exp(-r * T)
    if option_type == "call":
        intrinsic = max(S - K * discount, 0.0)
    else:
        intrinsic = max(K * discount - S, 0.0)

    if market_price <= intrinsic:
        return np.nan

    def objective(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    try:
        # brentq requires the objective to change sign over [a, b]
        low = objective(IV_LOWER_BOUND)
        high = objective(IV_UPPER_BOUND)

        if low * high > 0:
            # No sign change — market price is outside the BS range for this sigma interval
            return np.nan

        iv = brentq(objective, IV_LOWER_BOUND, IV_UPPER_BOUND, xtol=1e-8, maxiter=500)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


# ---------------------------------------------------------------------------
# DataFrame-level helper
# ---------------------------------------------------------------------------

def calculate_iv_surface(df: pd.DataFrame, r: float = RISK_FREE_RATE) -> pd.DataFrame:
    """Add an ``implied_volatility_bs`` column to an options DataFrame.

    Expects the DataFrame to have at minimum these columns:
    ``strike``, ``option_type``, ``bid``, ``ask``,
    ``underlying_price``, and either ``days_to_expiration`` or ``expiration``.

    Parameters
    ----------
    df : pd.DataFrame
        Options data (output of ``data_collection.clean_options_data``).
    r : float, optional
        Risk-free rate. Defaults to :data:`RISK_FREE_RATE`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an additional ``implied_volatility_bs`` column
        containing IVs as decimals (e.g. 0.20 = 20 %).
    """
    df = df.copy()
    df["mid_price"] = (df["bid"] + df["ask"]) / 2

    # Compute time to expiration in years
    if "days_to_expiration" in df.columns:
        df["T"] = df["days_to_expiration"] / 365.0
    else:
        today = pd.Timestamp.now().normalize()
        df["T"] = (pd.to_datetime(df["expiration"]) - today).dt.days / 365.0

    ivs = []
    for row in df.itertuples(index=False):
        iv = implied_volatility(
            market_price=row.mid_price,
            S=row.underlying_price,
            K=row.strike,
            T=row.T,
            r=r,
            option_type=row.option_type,
        )
        ivs.append(iv)

    df["implied_volatility_bs"] = ivs
    df = df.drop(columns=["T"])

    solved = df["implied_volatility_bs"].notna().sum()
    total = len(df)
    print(f"IV solved for {solved}/{total} contracts ({100 * solved / total:.1f} %)")
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Load the most recent cleaned CSV, compute IVs, and save results."""
    import glob
    import os

    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    csv_files = sorted(glob.glob(os.path.join(processed_dir, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in {processed_dir}. Run data_collection.py first.")
        return

    latest = csv_files[-1]
    print(f"Loading {latest}")
    df = pd.read_csv(latest)

    df_iv = calculate_iv_surface(df)

    # Summary statistics
    print("\n--- IV Summary ---")
    for opt_type in ["call", "put"]:
        subset = df_iv.loc[df_iv["option_type"] == opt_type, "implied_volatility_bs"].dropna()
        if not subset.empty:
            print(f"{opt_type.capitalize()}s — mean: {subset.mean():.2%}  "
                  f"median: {subset.median():.2%}  "
                  f"std: {subset.std():.2%}")

    # Save alongside the source file
    out_path = latest.replace(".csv", "_with_iv.csv")
    df_iv.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
