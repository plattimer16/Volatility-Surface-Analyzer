"""
Forward prices backed out of put-call parity.

The forward, not spot, is what an option is really struck against. Rather than
guess a dividend yield for SPY, we read the forward straight off the market:
for a European pair at the same strike and expiry,

    C - P = e^(-rT) (F - K)   =>   F = K + e^(rT) (C - P)

evaluated at the strike where |C - P| is smallest, which is the strike nearest
the forward and therefore the one where both legs are cheapest and tightest.

Only F is taken from the market. Backing r out of parity as well is tempting
and does not work: fitting C - P = D(F - K) across strikes gives a well
determined F but a wildly unstable D at short maturities, because F - K spans
too small a range to pin the slope. r stays an assumption; F does not.
"""

import numpy as np
import pandas as pd

# Parity carry on the Feb 2026 SPY snapshot is 2.45%. With SPY's trailing yield
# near 1.15% that implies r ~ 3.6%, not the 5% this project started with.
DEFAULT_RATE = 0.036

# TODO: this treats the dividend as a continuous yield, which it is not. SPY
# pays four discrete dividends a year, so the forward really steps down on each
# ex-date rather than decaying smoothly. Reading F off parity per expiry sidesteps
# most of that, but the fitted-carry fallback still assumes the smooth version.

DAYS_PER_YEAR = 365.0

# US equity options expire at the 16:00 ET close.
EXPIRY_HOUR = 16
MARKET_TZ = "America/New_York"

# An expiry whose parity forward sits this far (in logs) off the fitted carry
# curve is treated as a bad quote rather than a real observation. The test has
# to be on the forward level, not on the carry: carry = ln(F/S)/T divides by a
# tiny T at the front of the curve, so a one-cent quote error there shows up as
# a carry error of several percent even when the forward is perfectly good.
LOG_TOLERANCE = 0.005

# Below this maturity the per-expiry carry is not a meaningful number, for the
# same reason. Reported as NaN rather than as noise.
MIN_T_FOR_CARRY = 1.0 / 12.0


def year_fraction(expiration, snapshot_ts=None) -> np.ndarray:
    """Time to expiry in years, measured from an actual timestamp.

    Counting calendar days from midnight gives a contract expiring tomorrow
    T = 1/365 no matter what time of day the snapshot was taken. Measuring from
    the snapshot timestamp to the 16:00 ET close keeps T continuous in real
    time, which matters most exactly where it is smallest.

    Calendar-day basis (365), not business-day: the convention that matches how
    implied vols are quoted.

    TODO: a business-day count would fit the front of the curve better, where
    a weekend is a real fraction of the remaining life.
    """
    expiry = pd.to_datetime(pd.Series(expiration).values)
    expiry = pd.Series(expiry).dt.tz_localize(None) + pd.Timedelta(hours=EXPIRY_HOUR)

    if snapshot_ts is None:
        snapshot_ts = pd.Timestamp.now()
    snapshot = pd.Timestamp(snapshot_ts)

    try:
        expiry = expiry.dt.tz_localize(MARKET_TZ)
        if snapshot.tz is None:
            snapshot = snapshot.tz_localize(MARKET_TZ)
        else:
            snapshot = snapshot.tz_convert(MARKET_TZ)
    except (TypeError, ValueError):
        # No zoneinfo database available; fall back to naive arithmetic.
        if snapshot.tz is not None:
            snapshot = snapshot.tz_localize(None)

    delta = (expiry - snapshot).dt.total_seconds()
    return np.asarray(delta) / (DAYS_PER_YEAR * 86400.0)


def infer_snapshot(df: pd.DataFrame):
    """Best available snapshot timestamp for a contract frame.

    A live fetch stamps ``snapshot_ts``. Files written before that column
    existed do not have it, and falling back to "now" for a historical file
    silently produces negative times to expiry and a chain of nonsense
    downstream, so reconstruct the date from expiration minus days_to_expiration
    instead. Returns None only when there is nothing to go on.
    """
    if "snapshot_ts" in df.columns and df["snapshot_ts"].notna().any():
        return pd.Timestamp(df["snapshot_ts"].dropna().iloc[0])
    if {"expiration", "days_to_expiration"}.issubset(df.columns):
        recovered = (pd.to_datetime(df["expiration"])
                     - pd.to_timedelta(df["days_to_expiration"], unit="D")).min()
        if pd.notna(recovered):
            print(f"  NOTE: no snapshot_ts column; inferred {recovered:%Y-%m-%d} "
                  "from expiration minus days_to_expiration.")
            return pd.Timestamp(recovered)
    return None


def parity_forward(pairs: pd.DataFrame, T: float, r: float) -> tuple:
    """Back the forward out of put-call parity for one expiry.

    Parameters
    ----------
    pairs : pd.DataFrame
        One row per strike quoted on both sides, with columns ``strike``,
        ``call_mid``, ``put_mid``.
    T : float
        Time to expiry in years.
    r : float
        Continuously-compounded discount rate.

    Returns
    -------
    tuple
        ``(forward, strike_used, n_pairs)``, or ``(nan, nan, 0)`` if *pairs*
        is empty. The strike chosen minimises ``|C - P|``, so the forward is
        read where the two legs are closest to equal and both are near ATM.
    """
    if pairs is None or len(pairs) == 0:
        return (np.nan, np.nan, 0)

    diff = pairs["call_mid"] - pairs["put_mid"]
    i = diff.abs().values.argmin()
    strike = float(pairs["strike"].values[i])
    forward = strike + np.exp(r * T) * float(diff.values[i])
    return (forward, strike, len(pairs))


def _pairs_by_expiry(df: pd.DataFrame):
    """Yield (expiration, pairs frame) for every expiry with two-sided strikes."""
    price = "mid_price" if "mid_price" in df.columns else None
    work = df.copy()
    if price is None:
        work["mid_price"] = (work["bid"] + work["ask"]) / 2

    for expiration, group in work.groupby("expiration", sort=True):
        wide = group.pivot_table(
            index="strike", columns="option_type", values="mid_price", aggfunc="first"
        )
        if "call" not in wide.columns or "put" not in wide.columns:
            yield expiration, pd.DataFrame(columns=["strike", "call_mid", "put_mid"])
            continue
        wide = wide.dropna(subset=["call", "put"]).reset_index()
        wide = wide.rename(columns={"call": "call_mid", "put": "put_mid"})
        yield expiration, wide[["strike", "call_mid", "put_mid"]]


def fit_carry(curve: pd.DataFrame) -> float:
    """Through-origin least squares of ln(F/S) on T over parity-sourced rows.

    The forward curve is close to a one-parameter object: a single carry rate
    r - q reproduces every expiry to within a few tenths of a percent in logs.
    That fitted carry is what fills in expiries with no usable pair.
    """
    ok = curve[(curve["source"] == "parity") & curve["log_fs"].notna()]
    if len(ok) == 0:
        return 0.0
    T = ok["T"].to_numpy(dtype=float)
    y = ok["log_fs"].to_numpy(dtype=float)
    denom = float((T * T).sum())
    if denom <= 0:
        return 0.0
    return float((T * y).sum() / denom)


def forward_curve(
    df: pd.DataFrame,
    spot: float,
    r: float = DEFAULT_RATE,
    snapshot_ts=None,
) -> pd.DataFrame:
    """Build the per-expiry forward curve from put-call parity.

    Must be called *before* any liquidity filter. Parity needs the near-ATM
    call and put together, and a spread filter will happily delete one leg of
    that pair. Computed upstream, the forward is a property of the market
    snapshot; computed downstream it becomes a property of the filter
    settings, which would make the filter-robustness comparison circular.

    Returns one row per expiration with columns ``expiration``, ``T``,
    ``forward``, ``strike_used``, ``n_pairs``, ``carry``, ``implied_q`` and
    ``source`` (``'parity'`` or ``'fitted'``).
    """
    expirations, Ts, forwards, strikes, counts = [], [], [], [], []

    for expiration, pairs in _pairs_by_expiry(df):
        T = float(year_fraction([expiration], snapshot_ts)[0])
        F, K_used, n = parity_forward(pairs, T, r)
        expirations.append(expiration)
        Ts.append(T)
        forwards.append(F)
        strikes.append(K_used)
        counts.append(n)

    curve = pd.DataFrame(
        {
            "expiration": expirations,
            "T": Ts,
            "forward": forwards,
            "strike_used": strikes,
            "n_pairs": counts,
        }
    )
    curve = curve[curve["T"] > 0].reset_index(drop=True)
    curve["source"] = np.where(curve["forward"].notna(), "parity", "fitted")
    curve["log_fs"] = np.log(curve["forward"] / spot)

    # Two passes: fit the carry on the expiries parity handled cleanly, then
    # use it to replace the ones it did not.
    carry = fit_carry(curve)
    residual = (curve["log_fs"] - carry * curve["T"]).abs()
    bad = curve["forward"].isna() | (residual > LOG_TOLERANCE)
    if bad.any():
        curve.loc[bad, "forward"] = spot * np.exp(carry * curve.loc[bad, "T"])
        curve.loc[bad, "source"] = "fitted"
        curve.loc[bad, "strike_used"] = np.nan
        curve["log_fs"] = np.log(curve["forward"] / spot)
        carry = fit_carry(curve)

    # Only quote a per-expiry carry where T is long enough for it to mean
    # something; the fitted carry in curve.attrs is the number to trust.
    curve["carry"] = np.where(
        curve["T"] >= MIN_T_FOR_CARRY, curve["log_fs"] / curve["T"], np.nan
    )
    curve["implied_q"] = r - curve["carry"]
    curve.attrs["fitted_carry"] = carry
    curve.attrs["rate"] = r
    return curve.drop(columns=["log_fs"])


def add_forward_columns(df: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    """Attach forward, T, log-moneyness and the OTM flag to a contract frame.

    ``is_otm`` uses ``K <= F`` for puts and ``K > F`` for calls, so the strike
    straddling the forward belongs to exactly one side and no (expiry, strike)
    can end up with two implied vols.

    The flag is carried, not applied. Dropping the ITM leg here would destroy
    the same-strike pairs the put-call parity check needs, and with them the
    only evidence that the surface is right.
    """
    # Drop any stale derived columns first. Without this, re-running the
    # pipeline over its own output collides on merge (pandas silently makes
    # forward_x and forward_y) and everything downstream looks for a column
    # that no longer exists.
    stale = [c for c in ("T", "forward", "log_moneyness", "is_otm")
             if c in df.columns]
    out = df.drop(columns=stale).merge(
        curve[["expiration", "T", "forward"]], on="expiration", how="left"
    )
    out["log_moneyness"] = np.log(out["strike"] / out["forward"])
    is_put = out["option_type"] == "put"
    out["is_otm"] = np.where(
        is_put, out["strike"] <= out["forward"], out["strike"] > out["forward"]
    )
    return out


def main():
    """Print the forward curve for the most recent processed CSV."""
    import glob
    import os

    processed = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    files = [f for f in sorted(glob.glob(os.path.join(processed, "*.csv")))
             if "_with_iv" not in f]
    if not files:
        print(f"No CSV files found in {processed}. Run data_collection.py first.")
        return

    df = pd.read_csv(files[-1])
    spot = float(df["underlying_price"].iloc[0])
    snapshot = df["snapshot_ts"].iloc[0] if "snapshot_ts" in df.columns else None

    curve = forward_curve(df, spot, snapshot_ts=snapshot)
    print(f"{os.path.basename(files[-1])}  spot={spot:.2f}")
    print(curve.to_string(index=False))
    print(f"\nfitted carry r-q = {curve.attrs['fitted_carry']:.4f}")
    print(f"implied q at r={curve.attrs['rate']:.3f}: "
          f"{curve.attrs['rate'] - curve.attrs['fitted_carry']:.4f}")


if __name__ == "__main__":
    main()
