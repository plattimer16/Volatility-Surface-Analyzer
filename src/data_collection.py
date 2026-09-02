"""
Download and clean the SPY options chain from Yahoo Finance.

The raw chain is written to data/raw/ before anything is filtered, so a filter
sweep is a pure function of a saved file rather than something that needs a
fresh fetch into a different market. That is what makes the robustness
comparison in visualization.py mean anything.

    python src/data_collection.py                     fetch, save raw, clean
    python src/data_collection.py --from-raw <path>    re-clean a saved chain
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from forward import (DEFAULT_RATE, add_forward_columns, forward_curve,
                     infer_snapshot)

TICKER = "SPY"
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_SRC_DIR, "..", "data", "raw")
OUTPUT_DIR = os.path.join(_SRC_DIR, "..", "data", "processed")

# Relative, not absolute. A $0.50 cap keeps a penny option quoted 0.01/0.06,
# which is 600% wide, and throws away an ATM contract quoted 21.85/21.95 the
# moment the absolute spread clears fifty cents.
MAX_REL_SPREAD = 0.10

# One tick, and only one tick. A contract quoted 0.05/0.06 cannot be tighter
# than this and should not be punished for it; a contract quoted 0.01/0.06 is
# five ticks wide and is exactly what the relative test is here to remove.
MIN_ABS_SPREAD = 0.01

# Options worth less than a dime carry no usable volatility information: vega
# has collapsed, so the mid rounds to the same price across a wide range of
# sigma. On SPY this only removes the sub-2-delta tail.
MIN_MID_PRICE = 0.10

# Open interest, not volume. Volume is a flow and reads near zero early in the
# session; open interest is a stock and does not care what time it is.
MIN_OPEN_INTEREST = 100

MIN_DTE, MAX_DTE = 7, 365

OUTPUT_COLUMNS = [
    "snapshot_ts",
    "underlying_price",
    "expiration",
    "days_to_expiration",
    "T",
    "strike",
    "moneyness",
    "forward",
    "log_moneyness",
    "is_otm",
    "option_type",
    "bid",
    "ask",
    "mid_price",
    "bid_ask_spread",
    "rel_spread",
    "volume",
    "openInterest",
    "impliedVolatility",
]


def fetch_options_chain(ticker: str) -> pd.DataFrame:
    """Pull raw calls and puts for every listed expiry.

    Returns one row per contract with the yfinance columns plus
    ``expiration``, ``option_type``, ``underlying_price`` and ``snapshot_ts``.
    Nothing is filtered here.
    """
    import yfinance as yf

    stock = yf.Ticker(ticker)
    expirations = stock.options

    underlying_price = (
        stock.info.get("regularMarketPrice") or stock.fast_info["lastPrice"]
    )
    snapshot_ts = pd.Timestamp.now(tz="America/New_York")

    print(f"Underlying price: ${underlying_price:.2f}")
    print(f"Snapshot: {snapshot_ts:%Y-%m-%d %H:%M:%S %Z}")
    print(f"Found {len(expirations)} expiration dates")

    frames = []
    for exp in expirations:
        chain = stock.option_chain(exp)
        for option_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            df["expiration"] = exp
            df["option_type"] = option_type
            df["underlying_price"] = underlying_price
            df["snapshot_ts"] = snapshot_ts
            frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    print(f"Total raw contracts: {len(raw)}")
    return raw


def save_raw(df: pd.DataFrame, ticker: str) -> str:
    """Write the unfiltered chain to data/raw/."""
    os.makedirs(RAW_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RAW_DIR, f"{ticker.lower()}_chain_{timestamp}.csv")
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} raw rows to {path}")
    return path


def clean_options_data(
    df: pd.DataFrame,
    *,
    max_rel_spread: float = MAX_REL_SPREAD,
    min_abs_spread: float = MIN_ABS_SPREAD,
    min_mid_price: float = MIN_MID_PRICE,
    min_oi: int = MIN_OPEN_INTEREST,
    dte_range: tuple = (MIN_DTE, MAX_DTE),
    r: float = DEFAULT_RATE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter the raw chain and attach forward-relative columns.

    The thresholds are arguments rather than constants so the filter-robustness
    comparison can sweep them without editing this file.

    The forward curve is built from the two-sided-quote frame *before* the
    liquidity mask is applied. Put-call parity needs the near-ATM call and put
    together and a spread filter will happily delete one leg of that pair; on
    the Feb 2026 snapshot the old absolute-spread filter destroyed the ATM pair
    at 6 of 31 expiries. Computed upstream the forward is a property of the
    market, computed downstream it becomes a property of the filter settings.
    """
    df = df.copy()

    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["bid_ask_spread"] = df["ask"] - df["bid"]
    df["rel_spread"] = np.where(
        df["mid_price"] > 0, df["bid_ask_spread"] / df["mid_price"], np.inf
    )

    df["expiration_dt"] = pd.to_datetime(df["expiration"])
    snapshot = infer_snapshot(df)
    if snapshot is None:
        snapshot = pd.Timestamp.now()
    df["snapshot_ts"] = snapshot
    ref = pd.Timestamp(snapshot).tz_localize(None) if pd.Timestamp(snapshot).tz else pd.Timestamp(snapshot)
    df["days_to_expiration"] = (df["expiration_dt"] - ref.normalize()).dt.days

    df["moneyness"] = df["strike"] / df["underlying_price"]

    # Two-sided live quotes are required before anything else: parity needs a
    # real price on both legs, and a mid built off a zero bid is not a price.
    quoted = df.loc[(df["bid"] > 0) & (df["ask"] > 0)].copy()

    spot = float(df["underlying_price"].iloc[0])
    curve = forward_curve(quoted, spot, r=r, snapshot_ts=snapshot)

    lo, hi = dte_range
    mask = (
        ((quoted["rel_spread"] <= max_rel_spread)
         | (quoted["bid_ask_spread"] <= min_abs_spread))
        & (quoted["mid_price"] >= min_mid_price)
        & (quoted["openInterest"] >= min_oi)
        & (quoted["days_to_expiration"] >= lo)
        & (quoted["days_to_expiration"] <= hi)
    )
    cleaned = quoted.loc[mask].copy()
    cleaned = add_forward_columns(cleaned, curve)

    cols = [c for c in OUTPUT_COLUMNS if c in cleaned.columns]
    cleaned = (
        cleaned[cols]
        .sort_values(["days_to_expiration", "option_type", "strike"])
        .reset_index(drop=True)
    )

    if verbose:
        print(f"Two-sided quotes: {len(quoted)} of {len(df)}")
        print(f"Contracts after cleaning: {len(cleaned)}")
        n_parity = int((curve["source"] == "parity").sum())
        print(f"Forward curve: {n_parity}/{len(curve)} expiries from parity, "
              f"carry r-q = {curve.attrs['fitted_carry']:.4f} "
              f"(implied q = {r - curve.attrs['fitted_carry']:.4f})")

    cleaned.attrs["forward_curve"] = curve
    return cleaned


def save_to_csv(df: pd.DataFrame, ticker: str) -> str:
    """Write the cleaned frame to a timestamped CSV in data/processed/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{ticker.lower()}_options_{timestamp}.csv")
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    return path


def main():
    from_raw = None
    if "--from-raw" in sys.argv:
        from_raw = sys.argv[sys.argv.index("--from-raw") + 1]

    if from_raw:
        print(f"Re-cleaning saved chain {from_raw}")
        raw = pd.read_csv(from_raw)
    else:
        print(f"Fetching options data for {TICKER}...")
        raw = fetch_options_chain(TICKER)
        save_raw(raw, TICKER)

    cleaned = clean_options_data(raw)

    if cleaned.empty:
        print("No contracts passed the filters. Try relaxing thresholds.")
        return

    print("\n--- Summary ---")
    print(f"Expirations: {cleaned['expiration'].nunique()}")
    print(f"Strikes:     {cleaned['strike'].nunique()}")
    print(f"Calls:       {(cleaned['option_type'] == 'call').sum()}")
    print(f"Puts:        {(cleaned['option_type'] == 'put').sum()}")
    print(f"OTM:         {cleaned['is_otm'].sum()}")
    print(
        f"DTE range:   {cleaned['days_to_expiration'].min()} - "
        f"{cleaned['days_to_expiration'].max()} days"
    )

    save_to_csv(cleaned, TICKER)


if __name__ == "__main__":
    main()
