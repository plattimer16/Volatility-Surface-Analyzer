"""
Raw SVI fit, one expiry slice at a time.

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]

where w = sigma_BS^2 T is total variance and k = ln(K/F). Five parameters per
slice: a sets the level, b the wing slope, rho the asymmetry, m the horizontal
shift, sigma the curvature at the bottom.

The point is not a prettier surface. Interpolating scattered implied vols with
a cubic spline guarantees nothing about arbitrage: the fitted surface can imply
a negative risk-neutral density, which is a butterfly you could in principle
trade against the interpolation. SVI is a shape that can be checked, so it is
checked here (Gatheral and Jacquier's g(k) >= 0).

Scope: this fits each maturity independently and tests the butterfly condition
per slice. It does not enforce the cross-maturity condition, so the result is
not a globally arbitrage-free surface. The calendar check in validation.py
measures what is left.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

MIN_STRIKES_PER_SLICE = 8


class SVIParams(NamedTuple):
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_dict(self):
        return self._asdict()


def svi_total_variance(k, a, b, rho, m, sigma):
    """Total variance w(k) for raw SVI parameters."""
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_derivatives(k, p: SVIParams):
    """Return w, dw/dk and d2w/dk2 at *k*."""
    k = np.asarray(k, dtype=float)
    x = k - p.m
    root = np.sqrt(x ** 2 + p.sigma ** 2)
    w = p.a + p.b * (p.rho * x + root)
    dw = p.b * (p.rho + x / root)
    d2w = p.b * p.sigma ** 2 / root ** 3
    return w, dw, d2w


def butterfly_g(k, p: SVIParams):
    """Gatheral-Jacquier g(k). The risk-neutral density is non-negative iff g >= 0."""
    w, dw, d2w = svi_derivatives(k, p)
    return ((1.0 - k * dw / (2.0 * w)) ** 2
            - (dw ** 2 / 4.0) * (1.0 / w + 0.25)
            + d2w / 2.0)


def butterfly_ok(p: SVIParams, k_grid=None) -> bool:
    """Whether the slice is free of butterfly arbitrage across *k_grid*."""
    if k_grid is None:
        k_grid = np.linspace(-0.5, 0.3, 161)
    g = butterfly_g(k_grid, p)
    return bool(np.all(np.isfinite(g)) and np.all(g >= 0))


def _seed(k, w) -> SVIParams:
    """Moment-style starting point: level and shift from the minimum, wings from
    the observed slopes on either side of it."""
    i = int(np.argmin(w))
    m0 = float(k[i])
    a0 = max(float(w[i]), 1e-8)

    left, right = k < m0, k > m0
    sl = np.polyfit(k[left], w[left], 1)[0] if left.sum() >= 2 else -0.1
    sr = np.polyfit(k[right], w[right], 1)[0] if right.sum() >= 2 else 0.1

    b0 = max((abs(sl) + abs(sr)) / 2.0, 1e-4)
    rho0 = float(np.clip((sr + sl) / (abs(sr) + abs(sl) + 1e-12), -0.9, 0.9))
    span = float(k.max() - k.min()) or 0.1
    return SVIParams(a0, b0, rho0, m0, max(0.1 * span, 1e-3))


def fit_slice(k, w, weights=None, n_starts: int = 4) -> SVIParams:
    """Least-squares fit of one SVI slice.

    Residuals are weighted by vega when weights are supplied, so the fit tracks
    the at-the-money region tightly instead of being dragged around by the
    noisy wings where a cent of quote error is worth several vol points.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if weights is None:
        weights = np.ones_like(w)
    weights = np.asarray(weights, dtype=float)
    weights = weights / (weights.mean() or 1.0)

    def residual(theta):
        a, b, rho, m, sig = theta
        model = svi_total_variance(k, a, b, rho, m, sig)
        res = weights * (model - w)
        # Keep the minimum of the parabola non-negative: a + b sigma sqrt(1-rho^2)
        floor = a + b * sig * np.sqrt(max(1.0 - rho ** 2, 0.0))
        penalty = 1e3 * min(floor, 0.0)
        return np.concatenate([res, [penalty]])

    # Bounds matter more than they look. Left loose, the fit runs off along a
    # flat ridge where b -> large, a -> negative and m walks outside the traded
    # strikes: the shape still tracks the data but the parameters stop meaning
    # anything and the slice extrapolates absurdly. Tying m to the observed
    # range is the constraint that kills it, since m is where the variance
    # minimum sits and that has to be somewhere the market actually quotes.
    k_lo, k_hi = float(k.min()), float(k.max())
    lo = [-0.5, 0.0, -0.999, k_lo - 0.10, 1e-3]
    hi = [5.0, 2.0, 0.999, k_hi + 0.10, 2.0]

    base = _seed(k, w)
    best, best_cost = None, np.inf
    rng = np.random.default_rng(0)

    for attempt in range(n_starts):
        if attempt == 0:
            start = np.array(base, dtype=float)
        else:
            start = np.array([
                base.a * rng.uniform(0.5, 1.5),
                base.b * rng.uniform(0.3, 3.0),
                float(np.clip(base.rho + rng.uniform(-0.5, 0.5), -0.95, 0.95)),
                base.m + rng.uniform(-0.05, 0.05),
                base.sigma * rng.uniform(0.3, 3.0),
            ])
        start = np.clip(start, [x + 1e-9 for x in lo], [x - 1e-9 for x in hi])
        try:
            out = least_squares(residual, start, bounds=(lo, hi), max_nfev=4000)
        except (ValueError, RuntimeError):
            continue
        if out.cost < best_cost:
            best_cost, best = out.cost, out.x

    if best is None:
        return base
    return SVIParams(*[float(x) for x in best])


def fit_surface(df: pd.DataFrame, min_strikes: int = MIN_STRIKES_PER_SLICE) -> pd.DataFrame:
    """Fit every expiry with enough OTM strikes.

    Returns one row per expiry with the parameters, the fit error in vol points
    and whether the slice passes the butterfly test.
    """
    otm = df[df["is_otm"]] if "is_otm" in df.columns else df
    otm = otm.dropna(subset=["log_moneyness", "implied_volatility_bs", "T"])

    rows = []
    for T, g in otm.groupby("T"):
        g = g.sort_values("log_moneyness")
        if len(g) < min_strikes:
            continue
        k = g["log_moneyness"].to_numpy(dtype=float)
        iv = g["implied_volatility_bs"].to_numpy(dtype=float)
        w = iv ** 2 * float(T)
        vega = g["vega"].to_numpy(dtype=float) if "vega" in g.columns else None
        if vega is not None and np.isfinite(vega).all() and vega.sum() > 0:
            weights = vega
        else:
            weights = None

        p = fit_slice(k, w, weights)
        fitted_iv = np.sqrt(np.maximum(svi_total_variance(k, *p), 0.0) / float(T))
        rmse = float(100 * np.sqrt(np.mean((fitted_iv - iv) ** 2)))

        rows.append({
            "T": float(T),
            "days_to_expiration": int(g["days_to_expiration"].iloc[0]),
            "n_strikes": len(g),
            **p.as_dict(),
            "k_min": float(k.min()),
            "k_max": float(k.max()),
            "rmse_vol_pts": rmse,
            "butterfly_ok": butterfly_ok(p, np.linspace(k.min(), k.max(), 121)),
        })

    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def surface_iv(params: pd.DataFrame, k_grid, T_values=None,
               extrapolate: bool = False) -> pd.DataFrame:
    """Evaluate the fitted slices on a common k grid, as implied vol.

    Returns a frame indexed by T with one column per k, which is the shape the
    calendar-arbitrage check and the surface plot both want.

    Outside a slice's own traded strike range the value is NaN unless
    *extrapolate* is set. A fit is only evidence where there were quotes;
    projecting it into strikes the market never quoted invents the very
    arbitrage the check is looking for.
    """
    k_grid = np.asarray(k_grid, dtype=float)
    out = {}
    for _, row in params.iterrows():
        T = float(row["T"])
        if T_values is not None and T not in T_values:
            continue
        p = SVIParams(row["a"], row["b"], row["rho"], row["m"], row["sigma"])
        iv = np.sqrt(np.maximum(svi_total_variance(k_grid, *p), 0.0) / T)
        if not extrapolate and "k_min" in row:
            iv = np.where((k_grid >= row["k_min"]) & (k_grid <= row["k_max"]),
                          iv, np.nan)
        out[T] = iv
    return pd.DataFrame(out, index=k_grid).T.sort_index()


def main():
    import glob
    import os

    processed = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    files = glob.glob(os.path.join(processed, "*_with_iv.csv"))
    if not files:
        print(f"No *_with_iv.csv in {processed}. Run iv_calculation.py first.")
        return

    latest = max(files, key=os.path.getmtime)
    df = pd.read_csv(latest)
    params = fit_surface(df)
    if params.empty:
        print("No expiry had enough OTM strikes to fit.")
        return

    print(f"Fitted {len(params)} slices from {os.path.basename(latest)}\n")
    show = params[["days_to_expiration", "n_strikes", "a", "b", "rho", "m",
                   "sigma", "rmse_vol_pts", "butterfly_ok"]]
    print(show.round(4).to_string(index=False))
    print(f"\nmedian fit error: {params['rmse_vol_pts'].median():.3f} vol pts")
    print(f"butterfly-free slices: {int(params['butterfly_ok'].sum())}/{len(params)}")


if __name__ == "__main__":
    main()
