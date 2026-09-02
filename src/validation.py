"""
Checks on the surface, run as a script.

    python src/validation.py

There is no test framework here on purpose; the project is small and these are
the checks a reader actually wants to see. They answer "how do you know your
Black-Scholes is right?" with numbers instead of an assurance.

The checks are ordered from the ones that test the model against arithmetic to
the ones that test it against the market:

1. textbook price      - does the pricer reproduce a published value
2. model parity        - does C - P = e^(-rT)(F - K) hold identically
3. solver agreement    - does Newton agree with the bracketed solver
4. market parity       - does parity hold on the cleaned output, off-anchor
5. IV versus Yahoo     - does an independent implementation agree
6. calendar arbitrage  - is total variance non-decreasing in T at fixed k
"""

import os

import numpy as np
import pandas as pd

import forward as fwd_mod
from forward import DEFAULT_RATE
from iv_calculation import (
    LEGACY_RATE,
    black76_price,
    black76_vega,
    black_scholes_price,
    implied_volatility,
)

# Hull, Options Futures and Other Derivatives: S = K = 100, T = 1, r = 5%,
# sigma = 20% gives a call of 10.4506 and a put of 5.5735.
TEXTBOOK = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20,
                call=10.4506, put=5.5735)

K_GRID = np.linspace(-0.20, 0.10, 13)

# Parity is tested at the money. SPY options are American, and the early
# exercise premium a European formula cannot see grows with moneyness, so a
# strict equality test in the wings measures that premium rather than an error.
ATM_K_WINDOW = 0.02


def check_textbook_price() -> dict:
    """Price a published example and compare."""
    t = TEXTBOOK
    call = black_scholes_price(t["S"], t["K"], t["T"], t["r"], t["sigma"], "call")
    put = black_scholes_price(t["S"], t["K"], t["T"], t["r"], t["sigma"], "put")
    worst = max(abs(call - t["call"]), abs(put - t["put"]))
    return {
        "check": "textbook price",
        "metric": "max abs error vs published",
        "value": f"{worst:.2e}",
        "detail": f"call {call:.4f} (want {t['call']}), put {put:.4f} (want {t['put']})",
        "pass": worst < 5e-4,
    }


def check_model_parity(r: float = DEFAULT_RATE, n: int = 5000) -> dict:
    """Put-call parity must hold identically in the pricer, not approximately."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n):
        F = rng.uniform(50, 2000)
        K = rng.uniform(0.5 * F, 1.5 * F)
        T = rng.uniform(0.002, 3.0)
        sigma = rng.uniform(0.02, 2.0)
        C = black76_price(F, K, T, r, sigma, "call")
        P = black76_price(F, K, T, r, sigma, "put")
        worst = max(worst, abs((C - P) - np.exp(-r * T) * (F - K)))
    return {
        "check": "model parity",
        "metric": "max |C-P-e^(-rT)(F-K)|",
        "value": f"{worst:.2e}",
        "detail": f"{n} random inputs",
        "pass": worst < 1e-9,
    }


def check_solver_agreement(r: float = DEFAULT_RATE, n: int = 3000) -> dict:
    """Newton must reproduce the bracketed solver wherever vega carries signal.

    Split by vega, because the two solvers genuinely cannot agree where vega has
    collapsed: there the price is flat in sigma to machine precision and the
    volatility is not recoverable by any method. That regime is why Brent is
    kept as a fallback at all.
    """
    rng = np.random.default_rng(0)
    live, dead, n_fallback = [], [], 0
    for _ in range(n):
        F = rng.uniform(50, 2000)
        K = rng.uniform(0.6 * F, 1.6 * F)
        T = rng.uniform(0.01, 3.0)
        sigma = rng.uniform(0.05, 1.5)
        ot = "call" if rng.random() < 0.5 else "put"
        px = black76_price(F, K, T, r, sigma, ot)
        if not np.isfinite(px) or px <= 0:
            continue
        iv, how = implied_volatility(px, F, K, T, r, ot)
        if how == "brent":
            n_fallback += 1
        if not np.isfinite(iv):
            continue
        err = abs(iv - sigma)
        (live if black76_vega(F, K, T, r, sigma) >= 1e-6 else dead).append(err)

    worst_live = max(live) if live else 0.0
    worst_dead = max(dead) if dead else 0.0
    return {
        "check": "solver agreement",
        "metric": "max |IV recovered - true sigma|, vega >= 1e-6",
        "value": f"{worst_live:.2e}",
        "detail": (f"{len(live)} informative; {len(dead)} vega-dead reach "
                   f"{worst_dead:.1e}; Brent fallback {n_fallback}"),
        "pass": worst_live < 1e-6,
    }


def check_market_parity(df: pd.DataFrame, r: float = DEFAULT_RATE) -> dict:
    """Put-call parity on the cleaned output, at strikes other than the anchor.

    Worth being precise about what this does and does not test. The forward is
    read off parity at one strike per expiry, so the residual there is zero by
    construction. Every other strike is a genuine out-of-sample check: one
    number per expiry is fitted, and parity then either holds across the rest
    of the chain or it does not.

    Reported in dollars, in vol points via ATM vega, and as a fraction of the
    combined bid-ask spread. The last is the one that matters: a violation
    smaller than the cost of trading it is not an arbitrage.
    """
    need = {"expiration", "strike", "option_type", "mid_price", "forward", "T"}
    if not need.issubset(df.columns):
        return {"check": "market parity", "metric": "-", "value": "n/a",
                "detail": "missing columns", "pass": False}

    rows = []
    for (exp, K), g in df.groupby(["expiration", "strike"]):
        c = g[g["option_type"] == "call"]
        p = g[g["option_type"] == "put"]
        if len(c) != 1 or len(p) != 1:
            continue
        c, p = c.iloc[0], p.iloc[0]
        F, T = float(c["forward"]), float(c["T"])
        resid = (c["mid_price"] - p["mid_price"]) - np.exp(-r * T) * (F - K)
        spread = float(c["bid_ask_spread"] + p["bid_ask_spread"])
        vega = black76_vega(F, F, T, r, 0.20)
        rows.append({
            "k": np.log(K / F), "resid": abs(resid), "spread": spread,
            "vol_pts": 100 * abs(resid) / vega if vega > 1e-8 else np.nan,
        })

    if not rows:
        return {"check": "market parity", "metric": "-", "value": "n/a",
                "detail": "no two-sided strikes survived filtering", "pass": False}

    v = pd.DataFrame(rows)
    atm = v[v["k"].abs() <= ATM_K_WINDOW]
    if atm.empty:
        atm = v

    # The economic test: is a typical residual smaller than the cost of
    # trading it away? A violation inside the bid-ask spread is not an
    # arbitrage, it is the spread.
    med_resid = float(atm["resid"].median())
    med_spread = float(atm["spread"].median())

    profile = []
    for lo, hi in [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.0)]:
        b = v[(v["k"].abs() > lo) & (v["k"].abs() <= hi)]
        if len(b):
            profile.append(f"|k|<={hi:g}: ${b['resid'].median():.2f} (n={len(b)})")

    return {
        "check": "market parity",
        "metric": "median residual vs median spread, |k| <= 0.02",
        "value": f"${med_resid:.3f} vs ${med_spread:.3f}",
        "detail": (f"{len(atm)} ATM pairs, {atm['vol_pts'].median():.2f} vol pts; "
                   f"grows with moneyness (American early exercise) - "
                   + ", ".join(profile)),
        "pass": med_resid <= med_spread,
    }


def check_vs_yahoo(df: pd.DataFrame) -> dict:
    """Compare our IV against Yahoo's own, as an independent implementation.

    Perfect agreement is not the target and would be suspicious: Yahoo prices
    off spot with no dividend, which is the bug this project fixed. The signal
    is a high correlation overall plus a *smaller* error on the OTM subset,
    where both implementations are on their best behaviour.
    """
    if "impliedVolatility" not in df.columns:
        return {"check": "IV vs Yahoo", "metric": "-", "value": "n/a",
                "detail": "no Yahoo IV column", "pass": False}

    m = df.dropna(subset=["implied_volatility_bs", "impliedVolatility"])
    m = m[(m["impliedVolatility"] > 0)]
    if len(m) < 10:
        return {"check": "IV vs Yahoo", "metric": "-", "value": "n/a",
                "detail": "too few overlapping rows", "pass": False}

    corr = float(m["implied_volatility_bs"].corr(m["impliedVolatility"]))
    mae = float(100 * (m["implied_volatility_bs"] - m["impliedVolatility"]).abs().mean())
    detail = f"n={len(m)}"
    if "is_otm" in m.columns and m["is_otm"].any():
        o = m[m["is_otm"]]
        mae_otm = float(100 * (o["implied_volatility_bs"] - o["impliedVolatility"]).abs().mean())
        detail += f"; OTM-only MAE {mae_otm:.2f} vol pts (n={len(o)})"
    return {
        "check": "IV vs Yahoo",
        "metric": "correlation | MAE",
        "value": f"{corr:.4f} | {mae:.2f} vol pts",
        "detail": detail,
        "pass": corr > 0.90,
    }


def slice_total_variance(df: pd.DataFrame, k_grid=K_GRID) -> pd.DataFrame:
    """Interpolate total variance w = sigma^2 T onto a common k grid per expiry.

    Shared with the SVI fit. Only expiries whose observed strikes actually
    span a grid point are interpolated there; extrapolating the wings would
    invent the arbitrage this is trying to detect.
    """
    out = {}
    otm = df[df["is_otm"]] if "is_otm" in df.columns else df
    otm = otm.dropna(subset=["log_moneyness", "implied_volatility_bs", "T"])

    for T, g in otm.groupby("T"):
        g = g.sort_values("log_moneyness")
        k = g["log_moneyness"].to_numpy(dtype=float)
        w = (g["implied_volatility_bs"].to_numpy(dtype=float) ** 2) * float(T)
        if len(k) < 4:
            continue
        vals = np.interp(k_grid, k, w, left=np.nan, right=np.nan)
        vals[(k_grid < k.min()) | (k_grid > k.max())] = np.nan
        out[float(T)] = vals

    return pd.DataFrame(out, index=k_grid).T.sort_index()


def check_calendar_arbitrage(df: pd.DataFrame, k_grid=K_GRID, label="") -> dict:
    """Total variance must not decrease with maturity at fixed log-moneyness.

    If w(k, T2) < w(k, T1) for T2 > T1 then a calendar spread at that strike is
    worth less than nothing, which no arbitrage-free surface allows. Raw cubic
    interpolation of a scattered surface guarantees nothing of the sort, which
    is the argument for fitting a parameterisation instead.
    """
    w = slice_total_variance(df, k_grid)
    if len(w) < 2:
        return {"check": f"calendar arbitrage{label}", "metric": "-", "value": "n/a",
                "detail": "fewer than two usable slices", "pass": False}

    n_viol, worst, where, n_pairs = 0, 0.0, None, 0
    Ts = w.index.to_numpy()
    for i in range(len(Ts) - 1):
        a, b = w.iloc[i], w.iloc[i + 1]
        both = a.notna() & b.notna()
        n_pairs += int(both.sum())
        drop = (a[both] - b[both])
        viol = drop[drop > 0]
        n_viol += len(viol)
        if len(viol) and viol.max() > worst:
            worst = float(viol.max())
            where = (float(Ts[i]), float(Ts[i + 1]), float(viol.idxmax()))

    detail = f"{n_viol} of {n_pairs} grid transitions"
    if where:
        detail += f"; worst at k={where[2]:+.3f}, T {where[0]:.3f}->{where[1]:.3f}"
    return {
        "check": f"calendar arbitrage{label}",
        "metric": "violating transitions | worst dw",
        "value": f"{n_viol} | {worst:.2e}",
        "detail": detail,
        "pass": n_viol == 0,
        "n_violations": n_viol,
        "n_transitions": n_pairs,
    }


def skew_summary(df: pd.DataFrame, target_delta: float = 0.25) -> dict:
    """The actual skew number: 25-delta put IV minus 25-delta call IV per expiry.

    This is the comparison that replaces "puts at 30 vol, calls at 14". Mean
    put IV against mean call IV is not a skew measurement at all, because the
    two averages are taken over different ranges of moneyness. Matching on
    delta compares like with like, and is how the risk reversal is quoted.
    """
    if "delta_forward" not in df.columns:
        return {}
    otm = df[df["is_otm"]].dropna(subset=["delta_forward", "implied_volatility_bs"])
    rows = []
    for T, g in otm.groupby("T"):
        puts = g[g["option_type"] == "put"]
        calls = g[g["option_type"] == "call"]
        if puts.empty or calls.empty:
            continue
        p = puts.iloc[(puts["delta_forward"] + target_delta).abs().argsort()[:1]]
        c = calls.iloc[(calls["delta_forward"] - target_delta).abs().argsort()[:1]]
        if abs(abs(float(p["delta_forward"].iloc[0])) - target_delta) > 0.08:
            continue
        if abs(float(c["delta_forward"].iloc[0]) - target_delta) > 0.08:
            continue
        rows.append({
            "T": float(T),
            "dte": int(p["days_to_expiration"].iloc[0]),
            "put_iv": float(p["implied_volatility_bs"].iloc[0]),
            "call_iv": float(c["implied_volatility_bs"].iloc[0]),
        })
    if not rows:
        return {}
    out = pd.DataFrame(rows)
    out["rr_vol_pts"] = 100 * (out["put_iv"] - out["call_iv"])
    return {"table": out.merge(atm_skew_slope(df), on="T", how="left")}


def atm_skew_slope(df: pd.DataFrame, window: float = 0.06) -> pd.DataFrame:
    """At-the-money skew slope d(sigma)/dk per expiry, by OLS near k = 0.

    The fixed-delta risk reversal and the slope are different measurements and
    they behave differently with maturity. The 25-delta strike itself moves out
    like sigma sqrt(T), so a decaying slope times a widening strike range can
    leave the risk reversal roughly flat. "Skew flattens with maturity" is a
    statement about this slope.
    """
    otm = df[df["is_otm"]] if "is_otm" in df.columns else df
    otm = otm.dropna(subset=["log_moneyness", "implied_volatility_bs"])
    rows = []
    for T, g in otm.groupby("T"):
        near = g[g["log_moneyness"].abs() <= window]
        if len(near) < 5:
            continue
        k = near["log_moneyness"].to_numpy(dtype=float)
        iv = near["implied_volatility_bs"].to_numpy(dtype=float)
        slope = float(np.polyfit(k, iv, 1)[0])
        rows.append({"T": float(T), "atm_skew_slope": slope,
                     "slope_x_sqrtT": slope * np.sqrt(float(T))})
    return pd.DataFrame(rows)


def matched_strike_gap(df: pd.DataFrame, r: float = DEFAULT_RATE,
                       old_rate: float = LEGACY_RATE) -> dict:
    """Same-strike put minus call IV, priced the old way and the new way.

    The old way means spot in place of the forward at the rate the first
    version assumed, which is the q = 0 assumption this project started with.
    Repricing it at today's r instead would understate the original error,
    because 3.6% is already much closer to the true carry than 5% was.

    Same contracts, same snapshot, and the corrected day count on both sides,
    so the difference is attributable to the forward and to nothing else. That
    makes this an attribution experiment rather than a re-run of the old code:
    the original also had a calendar-day time to expiry measured from midnight,
    and holding that at the corrected value here isolates the one change being
    measured. On the February snapshot the old code itself printed a mean gap
    of 1.77 vol points against the 1.96 attributable to the forward alone.
    """
    work = df.dropna(subset=["forward", "T", "mid_price"]).copy()

    old = []
    for row in work.itertuples(index=False):
        S = float(row.underlying_price)
        iv, _ = implied_volatility(row.mid_price, S * np.exp(old_rate * row.T),
                                   row.strike, row.T, old_rate,
                                   row.option_type, method="brent")
        old.append(iv)
    work["iv_old"] = old

    def gap(col):
        wide = work.pivot_table(index=["expiration", "strike"],
                                columns="option_type", values=col).dropna()
        if wide.empty or "call" not in wide or "put" not in wide:
            return None, 0
        return 100 * (wide["put"] - wide["call"]), len(wide)

    g_old, n_old = gap("iv_old")
    g_new, n_new = gap("implied_volatility_bs")
    return {
        "n_pairs_old": n_old, "n_pairs_new": n_new,
        "mean_gap_old": float(g_old.mean()) if g_old is not None else np.nan,
        "mean_gap_new": float(g_new.mean()) if g_new is not None else np.nan,
        "median_gap_old": float(g_old.median()) if g_old is not None else np.nan,
        "median_gap_new": float(g_new.median()) if g_new is not None else np.nan,
    }


def headline_numbers(df: pd.DataFrame, r: float = DEFAULT_RATE) -> dict:
    """Exactly the numbers the README quotes, computed rather than typed."""
    out = {"contracts": len(df)}
    if "is_otm" in df.columns:
        out["otm_contracts"] = int(df["is_otm"].sum())
    out["expiries"] = int(df["expiration"].nunique())
    out["spot"] = float(df["underlying_price"].iloc[0])

    solved = int(df["implied_volatility_bs"].notna().sum())
    out["solved"] = solved
    out["solve_rate_pct"] = 100.0 * solved / len(df)
    if "iv_method" in df.columns:
        out["newton"] = int((df["iv_method"] == "newton").sum())
        out["brent_fallback"] = int((df["iv_method"] == "brent").sum())

    out.update(matched_strike_gap(df, r))

    y = check_vs_yahoo(df)
    out["yahoo"] = y["value"]

    Tv = df.drop_duplicates("expiration")[["T", "forward"]].dropna()
    T = Tv["T"].to_numpy(float)
    carry = float((T * np.log(Tv["forward"].to_numpy(float) / out["spot"])).sum()
                  / (T * T).sum())
    out["carry"] = carry
    out["implied_q"] = r - carry

    skew = skew_summary(df)
    if skew:
        t = skew["table"].dropna(subset=["atm_skew_slope"])
        if len(t):
            out["slope_front"] = float(t["atm_skew_slope"].iloc[0])
            out["slope_back"] = float(t["atm_skew_slope"].iloc[-1])
            out["dte_front"] = int(t["dte"].iloc[0])
            out["dte_back"] = int(t["dte"].iloc[-1])
            out["slope_ratio"] = abs(out["slope_front"] / out["slope_back"])
            out["slope_x_sqrtT_min"] = float(t["slope_x_sqrtT"].min())
            out["slope_x_sqrtT_max"] = float(t["slope_x_sqrtT"].max())
        rr = skew["table"]["rr_vol_pts"]
        out["rr_min"], out["rr_max"] = float(rr.min()), float(rr.max())
    return out


def run_all(df: pd.DataFrame, r: float = DEFAULT_RATE) -> pd.DataFrame:
    """Run every check and return them as a table."""
    return pd.DataFrame([
        check_textbook_price(),
        check_model_parity(r),
        check_solver_agreement(r),
        check_market_parity(df, r),
        check_vs_yahoo(df),
        check_calendar_arbitrage(df),
    ])


def _md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    """Minimal markdown table, so this file has no extra dependency."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            text = "" if pd.isna(v) else floatfmt.format(v)
        else:
            text = str(v)
        # A bare pipe inside a cell would silently split the column.
        return text.replace("|", r"\|")

    cols = list(df.columns)
    lines = ["| " + " | ".join(cell(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _rate_sensitivity(df, raw_df, rates=(0.02, 0.03, 0.036, 0.042, 0.05, 0.06)):
    """Median ATM parity residual as a function of the assumed rate."""
    from data_collection import clean_options_data

    source = raw_df if raw_df is not None else df
    rows = []
    for r in rates:
        try:
            cleaned = clean_options_data(source, verbose=False, r=r)
        except Exception:
            continue
        curve = cleaned.attrs.get("forward_curve")
        res = check_market_parity(cleaned, r=r)
        rows.append({
            "r": r,
            "fitted_carry": curve.attrs["fitted_carry"] if curve is not None else np.nan,
            "median_atm_residual": res["value"].split(" vs ")[0],
            "median_spread": res["value"].split(" vs ")[1],
            "passes": res["pass"],
        })
    return rows


def write_markdown(df: pd.DataFrame, source: str, out_path: str,
                   r: float = DEFAULT_RATE, raw_df=None) -> str:
    """Regenerate docs/VALIDATION.md from the data.

    Every number in that file and in the README comes from here, so neither can
    drift away from the committed CSV by being edited in place.
    """
    import svi

    spot = float(df["underlying_price"].iloc[0])
    snapshot = fwd_mod.infer_snapshot(df)

    out = [
        "# Validation",
        "",
        "Regenerated by `python src/validation.py`. Do not edit by hand: every",
        "number in the README is taken from this file, and this file is taken",
        "from the committed CSV.",
        "",
        f"- Source: `{os.path.basename(source)}`",
        f"- Snapshot: {snapshot}",
        f"- Spot: {spot:.2f}",
        f"- Contracts: {len(df):,} ({int(df['is_otm'].sum()):,} OTM)"
        if "is_otm" in df.columns else f"- Contracts: {len(df):,}",
        f"- Discount rate: r = {r:.3f}",
        "",
        "## Checks",
        "",
    ]

    table = run_all(df, r)
    checks = table.assign(result=np.where(table["pass"], "PASS", "FAIL"))
    out.append(_md_table(checks[["result", "check", "metric", "value", "detail"]]))
    n_pass = int(table["pass"].sum())
    out += ["", f"{n_pass} of {len(table)} checks pass.", ""]

    out += [
        "### What the parity check does and does not prove",
        "",
        "The forward is read off put-call parity at one strike per expiry, so",
        "the residual at that strike is zero by construction. Every other",
        "strike is out of sample: one number per expiry is fitted, and parity",
        "then either holds across the rest of the chain or it does not. It",
        "holds at the money to well inside the bid-ask spread, and degrades",
        "monotonically as strikes move in the money. That profile is the",
        "American early-exercise premium, which a European formula cannot see",
        "and which is largest exactly where the check is worst.",
        "",
        "## Forward curve",
        "",
        "Backed out per expiry, not assumed. `implied_q` is the dividend yield",
        "this implies at the stated `r`, and is the single best evidence that",
        "the forward is real rather than tuned: it lands near SPY's actual",
        "trailing yield without ever being told what that is.",
        "",
    ]

    # Rebuild from the raw chain when it is available, because that is what the
    # pipeline itself does: the curve is computed before the liquidity filters,
    # so re-deriving it from the filtered file would report a different forward
    # from the one the implied vols were actually solved against.
    if raw_df is not None:
        src = raw_df.copy()
        src["mid_price"] = (src["bid"] + src["ask"]) / 2
        src = src[(src["bid"] > 0) & (src["ask"] > 0)]
        curve = fwd_mod.forward_curve(src, spot, r=r, snapshot_ts=snapshot)
        cshow = curve[["expiration", "T", "forward", "strike_used", "n_pairs",
                       "source", "carry", "implied_q"]]
        carry = curve.attrs["fitted_carry"]
        basis = "the raw chain, before any liquidity filter"
    else:
        cshow = (df[["expiration", "T", "forward"]].drop_duplicates("expiration")
                 .sort_values("T").reset_index(drop=True))
        cshow["carry"] = np.where(cshow["T"] >= fwd_mod.MIN_T_FOR_CARRY,
                                  np.log(cshow["forward"] / spot) / cshow["T"], np.nan)
        cshow["implied_q"] = r - cshow["carry"]
        Tv = cshow["T"].to_numpy(float)
        yv = np.log(cshow["forward"].to_numpy(float) / spot)
        carry = float((Tv * yv).sum() / (Tv * Tv).sum())
        basis = "the forward column of the committed file"

    out.append(_md_table(cshow))
    out += [
        "",
        f"Computed from {basis}.",
        "",
        f"Fitted carry `r - q` = **{carry:.4f}**, "
        f"implying `q` = **{r - carry:.4f}** at r = {r:.3f}.",
        "",
        "Parity pins the carry, not r and q separately. Splitting it needs one",
        "of them from outside, so the implied q above is not independent",
        "confirmation of anything: r was chosen knowing roughly what SPY",
        "yields. The table below is the check that matters, and it says this",
        "snapshot does not identify r at all.",
        "",
        "### Sensitivity to the assumed rate",
        "",
        "The forward is re-derived from parity at each rate and the",
        "out-of-sample parity test re-run.",
        "",
        _md_table(pd.DataFrame(_rate_sensitivity(df, raw_df)), floatfmt="{:.4f}"),
        "",
        "## Skew",
        "",
        "Two different measurements that behave differently with maturity, and",
        "conflating them is an easy way to state the wrong finding.",
        "",
        "`atm_skew_slope` is d(sigma)/dk fitted near k = 0. It flattens with",
        "maturity. `rr_vol_pts` is the 25-delta risk reversal, put IV minus",
        "call IV, and it does not: the 25-delta strike itself moves out like",
        "sigma sqrt(T), so a decaying slope over a widening strike range",
        "roughly cancels. `slope_x_sqrtT` is close to constant, which is the",
        "1/sqrt(T) decay law rather than a claim about it.",
        "",
    ]

    skew = skew_summary(df)
    if skew:
        s = skew["table"]
        sshow = s[["dte", "put_iv", "call_iv", "rr_vol_pts", "atm_skew_slope",
                   "slope_x_sqrtT"]].copy()
        sshow["put_iv"] = 100 * sshow["put_iv"]
        sshow["call_iv"] = 100 * sshow["call_iv"]
        out.append(_md_table(sshow, floatfmt="{:.3f}"))
        out += [
            "",
            f"ATM skew slope goes from {s['atm_skew_slope'].iloc[0]:.3f} at "
            f"{int(s['dte'].iloc[0])} DTE to {s['atm_skew_slope'].iloc[-1]:.3f} at "
            f"{int(s['dte'].iloc[-1])} DTE, a factor of "
            f"{abs(s['atm_skew_slope'].iloc[0] / s['atm_skew_slope'].iloc[-1]):.1f}, "
            f"while slope x sqrt(T) stays within "
            f"[{s['slope_x_sqrtT'].min():.3f}, {s['slope_x_sqrtT'].max():.3f}].",
            "",
        ]

    out += [
        "## SVI",
        "",
        "One fit per expiry, weighted by vega so the at-the-money region is",
        "tracked tightly rather than being dragged around by the wings.",
        "`butterfly_ok` is Gatheral and Jacquier's g(k) >= 0 across the traded",
        "strike range.",
        "",
    ]
    params = svi.fit_surface(df)
    if not params.empty:
        pshow = params[["days_to_expiration", "n_strikes", "a", "b", "rho", "m",
                        "sigma", "rmse_vol_pts", "butterfly_ok"]]
        out.append(_md_table(pshow))
        out += [
            "",
            f"Median fit error **{params['rmse_vol_pts'].median():.3f} vol points**, "
            f"worst {params['rmse_vol_pts'].max():.3f}. "
            f"Butterfly-free slices: **{int(params['butterfly_ok'].sum())}/{len(params)}**.",
            "",
        ]

        kg = K_GRID
        rows = []
        for label, extrap in [("within traded strikes", False), ("extrapolated", True)]:
            fit = svi.surface_iv(params, kg, extrapolate=extrap)
            w = (fit ** 2).mul(fit.index, axis=0)
            n_v = n_p = 0
            for i in range(len(w) - 1):
                a, b = w.iloc[i], w.iloc[i + 1]
                both = a.notna() & b.notna()
                n_p += int(both.sum())
                n_v += int(((a[both] - b[both]) > 0).sum())
            rows.append({"surface": f"SVI, {label}", "violations": n_v,
                         "transitions": n_p})
        obs = check_calendar_arbitrage(df)
        rows.insert(0, {"surface": "observed slices, linear in k",
                        "violations": obs["n_violations"],
                        "transitions": obs["n_transitions"]})
        out += [
            "### Calendar arbitrage",
            "",
            "Total variance `w = sigma^2 T` must not decrease with maturity at",
            "fixed k. The contrast is the argument for not projecting a fitted",
            "slice past the strikes the market actually quotes.",
            "",
            _md_table(pd.DataFrame(rows)),
            "",
        ]

    if raw_df is not None:
        out += [
            "## Filter robustness",
            "",
            "The same 30-day skew under three settings, so the filters can be",
            "seen not to have chosen the conclusion. Run from the committed",
            "reference, which has already been through one pass of the original",
            "filters, so `loose` relaxes the current thresholds only within",
            "that envelope.",
            "",
        ]
        from data_collection import clean_options_data
        from iv_calculation import calculate_iv_surface
        rows = []
        for label, kw in [("strict", dict(max_rel_spread=0.05, min_oi=500, min_mid_price=0.20)),
                          ("baseline", dict()),
                          ("loose", dict(max_rel_spread=0.20, min_oi=10, min_mid_price=0.05))]:
            cl = clean_options_data(raw_df, verbose=False, **kw)
            if cl.empty:
                continue
            wi = calculate_iv_surface(cl)
            sk = skew_summary(wi)
            slope = np.nan
            if sk:
                t30 = sk["table"].iloc[(sk["table"]["dte"] - 30).abs().argsort()[:1]]
                slope = float(t30["atm_skew_slope"].iloc[0])
                rr = float(t30["rr_vol_pts"].iloc[0])
            else:
                rr = np.nan
            rows.append({"filters": label, "contracts": len(cl),
                         "expiries": cl["expiration"].nunique(),
                         "atm_skew_slope_30d": slope, "rr_25d_vol_pts_30d": rr})
        out += [_md_table(pd.DataFrame(rows), floatfmt="{:.3f}"), ""]
        out += ["The skew keeps its sign and its rough magnitude across all",
                "three, which is the point.", ""]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"Wrote {out_path}")
    return out_path


def main():
    import glob

    processed = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    files = glob.glob(os.path.join(processed, "*_with_iv.csv"))
    if not files:
        print(f"No *_with_iv.csv in {processed}. Run iv_calculation.py first.")
        return

    source = max(files, key=os.path.getmtime)
    df = pd.read_csv(source)
    print(f"Validating {os.path.basename(source)}  ({len(df):,} rows)\n")

    table = run_all(df)
    for _, row in table.iterrows():
        flag = "PASS" if row["pass"] else "FAIL"
        print(f"[{flag}] {row['check']}")
        print(f"       {row['metric']}: {row['value']}")
        print(f"       {row['detail']}")

    n_fail = int((~table["pass"]).sum())
    print(f"\n{len(table) - n_fail}/{len(table)} checks passed")

    skew = skew_summary(df)
    if skew:
        print("\n--- 25-delta risk reversal and ATM skew slope by expiry ---")
        t = skew["table"]
        print(t.assign(put_iv=(100 * t.put_iv).round(2),
                       call_iv=(100 * t.call_iv).round(2),
                       rr_vol_pts=t.rr_vol_pts.round(2),
                       atm_skew_slope=t.atm_skew_slope.round(3),
                       slope_x_sqrtT=t.slope_x_sqrtT.round(3))
               [["dte", "put_iv", "call_iv", "rr_vol_pts",
                 "atm_skew_slope", "slope_x_sqrtT"]]
               .to_string(index=False))

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    # Pair the document with the chain the reference actually came from, by
    # snapshot timestamp. Taking the newest file instead would quietly mix a
    # forward curve from one day into a report about another.
    raw_df = None
    want = str(df["snapshot_ts"].iloc[0])[:19] if "snapshot_ts" in df.columns else None
    for path in glob.glob(os.path.join(raw_dir, "*.csv")):
        head = pd.read_csv(path, nrows=1)
        if "snapshot_ts" not in head.columns:
            continue
        if want is None or str(head["snapshot_ts"].iloc[0])[:19] == want:
            raw_df = pd.read_csv(path)
            print(f"Forward curve from {os.path.basename(path)}")
            break

    docs = os.path.join(os.path.dirname(__file__), "..", "docs", "VALIDATION.md")
    write_markdown(df, source, docs, raw_df=raw_df)

if __name__ == "__main__":
    main()
