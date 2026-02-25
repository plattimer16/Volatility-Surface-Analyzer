"""
Download and clean SPY options chain data from Yahoo Finance.

Fetches all available expirations, filters out illiquid contracts, adds
derived columns (mid-price, moneyness, DTE), and saves the result to CSV
in data/processed/.

Pipeline
--------
1. fetch_options_chain  — pull raw calls + puts for every listed expiry
2. clean_options_data   — filter by spread / volume, compute helper columns
3. save_to_csv          — write timestamped CSV to data/processed/
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration — change these to fetch a different ticker or relax filters
# ---------------------------------------------------------------------------

TICKER = "SPY"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Contracts with a bid-ask spread wider than this are too illiquid to trust;
# the mid-price would be far from the true fair value.
MAX_BID_ASK_SPREAD = 0.50  # dollars

# Very low-volume contracts often have stale or crossed quotes.
MIN_VOLUME = 10


def fetch_options_chain(ticker: str) -> pd.DataFrame:
    """Fetch the full options chain for all available expirations.

    Iterates over every listed expiry, downloads calls and puts, tags each row
    with the expiration date, option type, and the current spot price, then
    concatenates everything into a single DataFrame.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (e.g. ``'SPY'``).

    Returns
    -------
    pd.DataFrame
        Raw options data — one row per contract, with yfinance default columns
        plus ``expiration``, ``option_type``, and ``underlying_price``.
    """
    stock = yf.Ticker(ticker)

    # yfinance returns a tuple of 'YYYY-MM-DD' strings, one per listed expiry
    expirations = stock.options

    # Prefer regularMarketPrice; fall back to fast_info for after-hours quotes
    underlying_price = (
        stock.info.get("regularMarketPrice") or stock.fast_info["lastPrice"]
    )

    print(f"Underlying price: ${underlying_price:.2f}")
    print(f"Found {len(expirations)} expiration dates")

    frames = []
    for exp in expirations:
        chain = stock.option_chain(exp)  # returns (calls, puts) namedtuple

        for option_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            # Attach metadata columns so rows stay self-contained after concat
            df["expiration"] = exp
            df["option_type"] = option_type
            df["underlying_price"] = underlying_price
            frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    print(f"Total raw contracts: {len(raw)}")
    return raw


def clean_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean raw options data, adding derived columns.

    Derived columns added
    ---------------------
    ``mid_price``          : (bid + ask) / 2 — used as the market price for IV solving
    ``bid_ask_spread``     : ask − bid — proxy for liquidity/transaction cost
    ``days_to_expiration`` : calendar days until expiry from today
    ``moneyness``          : K / S — normalised measure of how far from ATM a strike is

    Filters applied
    ---------------
    - Bid-ask spread ≤ MAX_BID_ASK_SPREAD  (remove illiquid contracts)
    - Volume ≥ MIN_VOLUME                  (remove stale/orphaned quotes)
    - Bid > 0 and ask > 0                  (remove crossed / zero quotes)
    - days_to_expiration > 0               (remove already-expired contracts)

    Parameters
    ----------
    df : pd.DataFrame
        Raw output of :func:`fetch_options_chain`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a curated set of columns, sorted by
        (DTE, option_type, strike).
    """
    df = df.copy()

    # --- Derived price columns ---
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["bid_ask_spread"] = df["ask"] - df["bid"]

    # --- Time to expiration ---
    df["expiration_dt"] = pd.to_datetime(df["expiration"])
    # Normalise to midnight so partial-day differences don't bleed between dates
    df["days_to_expiration"] = (
        df["expiration_dt"] - pd.Timestamp.now().normalize()
    ).dt.days

    # --- Normalised strike (K/S) ---
    # Moneyness = 1.0 means exactly ATM, < 1.0 is OTM for a call, > 1.0 is ITM
    df["moneyness"] = df["strike"] / df["underlying_price"]

    # --- Liquidity and validity filters ---
    mask = (
        (df["bid_ask_spread"] <= MAX_BID_ASK_SPREAD)  # wide spreads → unreliable mid
        & (df["volume"] >= MIN_VOLUME)                 # thin volume → stale quotes
        & (df["bid"] > 0)                              # crossed / zero quotes
        & (df["ask"] > 0)
        & (df["days_to_expiration"] > 0)               # skip already-expired rows
    )
    cleaned = df.loc[mask].copy()

    # Keep only the columns we'll actually use downstream
    cols = [
        "underlying_price",
        "expiration",
        "days_to_expiration",
        "strike",
        "moneyness",
        "option_type",
        "bid",
        "ask",
        "mid_price",
        "bid_ask_spread",
        "volume",
        "openInterest",
        "impliedVolatility",   # Yahoo's own IV estimate — useful as a sanity check
    ]
    cleaned = (
        cleaned[cols]
        .sort_values(["days_to_expiration", "option_type", "strike"])
        .reset_index(drop=True)
    )

    print(f"Contracts after cleaning: {len(cleaned)}")
    return cleaned


def save_to_csv(df: pd.DataFrame, ticker: str) -> str:
    """Save the cleaned DataFrame to a timestamped CSV file.

    The timestamp in the filename lets you accumulate multiple daily snapshots
    in data/processed/ and always know which file is most recent.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned options data.
    ticker : str
        Ticker symbol used as the filename prefix.

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker.lower()}_options_{timestamp}.csv"
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    return path


def main():
    """Orchestrate the full download → clean → save pipeline."""
    print(f"Fetching options data for {TICKER}...")
    raw = fetch_options_chain(TICKER)
    cleaned = clean_options_data(raw)

    if cleaned.empty:
        print("No contracts passed the filters. Try relaxing thresholds.")
        return

    # Quick summary so you can sanity-check the download at a glance
    print("\n--- Summary ---")
    print(f"Expirations: {cleaned['expiration'].nunique()}")
    print(f"Strikes:     {cleaned['strike'].nunique()}")
    print(f"Calls:       {(cleaned['option_type'] == 'call').sum()}")
    print(f"Puts:        {(cleaned['option_type'] == 'put').sum()}")
    print(
        f"DTE range:   {cleaned['days_to_expiration'].min()} – "
        f"{cleaned['days_to_expiration'].max()} days"
    )

    save_to_csv(cleaned, TICKER)


if __name__ == "__main__":
    main()
