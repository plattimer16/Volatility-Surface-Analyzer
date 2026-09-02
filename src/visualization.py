"""
Plots of the volatility surface.

Everything is drawn against log-moneyness k = ln(K/F) rather than K/S. K/S is
not comparable across maturities, because the width of the distribution scales
with sqrt(T): the same K/S is a very different number of standard deviations at
one week and at one year. k is comparable, and k/sqrt(T) more so.

    python src/visualization.py

Figures land in data/figures/.
"""

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3-D projection
from scipy.interpolate import griddata

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_SRC_DIR, "..", "data", "figures")

STYLE = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}

# Asymmetric on purpose: the put wing is where the content is.
K_MIN, K_MAX = -0.30, 0.20

ATM_K_BAND = 0.02

CALL_COLOR = "#2196F3"
PUT_COLOR = "#E53935"


def _resolve_iv_column(df):
    """Our own IV if present, else Yahoo's."""
    for col in ("implied_volatility_bs", "impliedVolatility"):
        if col in df.columns:
            return col
    raise KeyError(
        "DataFrame has no IV column. Expected 'implied_volatility_bs' or "
        "'impliedVolatility'. Run iv_calculation.py first."
    )


def _resolve_k(df):
    """Return a log-moneyness series, deriving or approximating it if needed.

    Older CSVs in this repo predate the forward, so fall back to ln(K/S) and say
    so rather than refusing to plot them.
    """
    if "log_moneyness" in df.columns:
        return df["log_moneyness"]
    if "forward" in df.columns:
        return np.log(df["strike"] / df["forward"])
    print("  NOTE: no forward in this file; using ln(K/S), which is not "
          "comparable across maturities.")
    return np.log(df["strike"] / df["underlying_price"])


def _resolve_otm(df):
    """OTM mask, falling back to spot when there is no forward."""
    if "is_otm" in df.columns:
        return df["is_otm"].astype(bool)
    ref = df["forward"] if "forward" in df.columns else df["underlying_price"]
    is_put = df["option_type"] == "put"
    return np.where(is_put, df["strike"] <= ref, df["strike"] > ref)


def _prepare(df, iv_col=None, otm_only=True, k_min=K_MIN, k_max=K_MAX):
    """Attach k, drop unusable rows, optionally keep only the OTM side."""
    iv_col = iv_col or _resolve_iv_column(df)
    out = df.copy()
    out["k"] = _resolve_k(out)
    out["_otm"] = _resolve_otm(out)

    mask = (
        out[iv_col].notna()
        & (out[iv_col] > 0)
        & out["k"].notna()
        & (out["k"] >= k_min)
        & (out["k"] <= k_max)
    )
    if otm_only:
        mask &= out["_otm"]
    return out.loc[mask].copy(), iv_col


def _pct_formatter(decimals=0):
    """Percent tick labels. Needs a decimal place on a narrow axis, or the
    ticks round to the same number twice."""
    return mticker.FuncFormatter(lambda x, _: f"{x:.{decimals}%}")


def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    # A little padding: with bbox_inches alone the 3-D axis labels clip.
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    print(f"  Saved -> {path}")
    return path


def plot_iv_surface(df, ticker="SPY", iv_col=None):
    """3-D implied volatility surface over log-moneyness and days to expiry.

    Built from the OTM side only, so there is exactly one volatility per strike
    by construction. The earlier version of this plot used calls alone to avoid
    "duplicate grid points"; that was treating the symptom. Overlaying an ITM
    put and an OTM call at the same strike gave two different vols for the same
    contract, and the fix is to choose the liquid side rather than to drop half
    the chain.

    Note the interpolation guarantees nothing about arbitrage. See svi.py.
    """
    data, iv_col = _prepare(df, iv_col, otm_only=True)
    if data.empty:
        raise ValueError("No valid OTM data for the IV surface plot.")

    x_vals = np.linspace(data["k"].min(), data["k"].max(), 60)
    y_vals = np.linspace(data["days_to_expiration"].min(),
                         data["days_to_expiration"].max(), 60)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)

    grid_z = griddata(
        points=data[["k", "days_to_expiration"]].values,
        values=data[iv_col].values,
        xi=(grid_x, grid_y),
        method="cubic",
    )

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap="RdYlGn_r",
                              alpha=0.85, linewidth=0, antialiased=True)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
        cbar.ax.yaxis.set_major_formatter(_pct_formatter())

        ax.set_xlabel("Log-moneyness  k = ln(K/F)", labelpad=12)
        ax.set_ylabel("Days to expiration", labelpad=10)
        ax.set_zlabel("Implied volatility", labelpad=8)
        ax.zaxis.set_major_formatter(_pct_formatter())
        ax.set_title(f"{ticker} - Implied volatility surface (OTM options)", pad=12)
        ax.view_init(elev=28, azim=-55)

    return fig


def plot_volatility_smile(df, target_dte=30, ticker="SPY", iv_col=None):
    """One continuous OTM smile for the expiry nearest *target_dte*.

    Puts to the left of the forward, calls to the right, joined as a single
    curve. The old version overlaid all calls against all puts across the whole
    strike range, which drew two curves where there is only one volatility per
    strike: a picture of the bug rather than of the market.
    """
    data, iv_col = _prepare(df, iv_col, otm_only=True)
    dtes = data["days_to_expiration"].unique()
    chosen = dtes[np.argmin(np.abs(dtes - target_dte))]
    exp_data = data[data["days_to_expiration"] == chosen].sort_values("k")

    print(f"  Smile: DTE={chosen} (requested {target_dte})")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(exp_data["k"], exp_data[iv_col], color="#37474F",
                linewidth=1.6, zorder=2, label="OTM smile")

        for label, sub, color, marker in [
            ("Puts (K < F)", exp_data[exp_data["option_type"] == "put"], PUT_COLOR, "s"),
            ("Calls (K > F)", exp_data[exp_data["option_type"] == "call"], CALL_COLOR, "o"),
        ]:
            ax.scatter(sub["k"], sub[iv_col], s=18, color=color, marker=marker,
                       zorder=3, label=label)

        ax.axvline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="Forward (k=0)")
        ax.set_xlabel("Log-moneyness  k = ln(K/F)")
        ax.set_ylabel("Implied volatility")
        ax.yaxis.set_major_formatter(_pct_formatter())
        ax.set_title(f"{ticker} - Volatility smile, OTM only  (DTE = {chosen})")
        ax.legend()

    return fig


def plot_before_after_smile(df, target_dte=30, ticker="SPY"):
    """The bug and its fix, side by side on one expiry.

    Left: every call and every put priced off spot with no dividend yield and at
    the 5% rate the first version assumed, which is what this project originally
    produced. Right: one volatility per strike, taken from the OTM side, priced
    off the parity forward.

    Both panels are the same snapshot and the same contracts, so the difference
    between them is the pricing and nothing else.
    """
    from iv_calculation import LEGACY_RATE, implied_volatility

    work = df.copy()
    work["k"] = _resolve_k(work)
    work["_otm"] = _resolve_otm(work)
    dtes = work["days_to_expiration"].unique()
    chosen = dtes[np.argmin(np.abs(dtes - target_dte))]
    exp_data = work[work["days_to_expiration"] == chosen].copy()

    # Reprice the same contracts the old way: spot in place of the forward,
    # which is exactly the q = 0 assumption.
    old_iv = []
    for row in exp_data.itertuples(index=False):
        S = float(row.underlying_price)
        iv, _ = implied_volatility(row.mid_price, S * np.exp(LEGACY_RATE * row.T),
                                   row.strike, row.T, LEGACY_RATE,
                                   row.option_type, method="brent")
        old_iv.append(iv)
    exp_data["iv_old"] = old_iv

    window = exp_data[(exp_data["k"] >= K_MIN) & (exp_data["k"] <= K_MAX)]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

        ax = axes[0]
        for label, ot, color, marker in [("Puts", "put", PUT_COLOR, "s"),
                                         ("Calls", "call", CALL_COLOR, "o")]:
            sub = window[window["option_type"] == ot].sort_values("k")
            ax.plot(sub["k"], sub["iv_old"], color=color, marker=marker,
                    markersize=4, linewidth=1.6, label=label)
        ax.set_title("Before: spot, no dividend yield, all strikes")
        ax.set_ylabel("Implied volatility")

        ax = axes[1]
        otm = window[window["_otm"]].sort_values("k")
        ax.plot(otm["k"], otm["implied_volatility_bs"], color="#37474F",
                linewidth=1.6, zorder=2)
        for label, ot, color, marker in [("Puts (K < F)", "put", PUT_COLOR, "s"),
                                         ("Calls (K > F)", "call", CALL_COLOR, "o")]:
            sub = otm[otm["option_type"] == ot]
            ax.scatter(sub["k"], sub["implied_volatility_bs"], s=18, color=color,
                       marker=marker, zorder=3, label=label)
        ax.set_title("After: parity forward, OTM only")

        for ax in axes:
            ax.axvline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_xlabel("Log-moneyness  k = ln(K/F)")
            ax.yaxis.set_major_formatter(_pct_formatter())
            ax.legend()

        fig.suptitle(f"{ticker} - the put/call volatility gap was mostly ours "
                     f"(DTE = {chosen})", y=1.02)

    return fig


def plot_delta_smile(df, max_dtes=5, ticker="SPY", iv_col=None):
    """Smile against forward delta, which is how the market quotes it.

    A desk trades the 25-delta risk reversal and the 25-delta butterfly, not a
    strike. Delta is also the axis on which the two wings are directly
    comparable, since it already accounts for maturity.
    """
    data, iv_col = _prepare(df, iv_col, otm_only=True)
    if "delta_forward" not in data.columns:
        raise ValueError("No delta_forward column; run iv_calculation.py.")

    data = data.dropna(subset=["delta_forward"])

    # Put both legs on one axis: the call delta of the same strike. Exactly,
    # delta_call - delta_put = e^(-rT), so a -25 delta put is a 75 delta call
    # at that strike, not a 25 delta one.
    from iv_calculation import RISK_FREE_RATE
    disc = np.exp(-RISK_FREE_RATE * data["T"])
    data["delta_axis"] = np.where(data["option_type"] == "put",
                                  data["delta_forward"] + disc,
                                  data["delta_forward"])
    dtes = sorted(data["days_to_expiration"].unique())[:max_dtes]
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(dtes) - 1, 1)) for i in range(len(dtes))]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for dte, color in zip(dtes, colors):
            sub = data[data["days_to_expiration"] == dte].sort_values("delta_axis")
            ax.plot(sub["delta_axis"], sub[iv_col], color=color, linewidth=1.8,
                    label=f"{dte} DTE")

        for x, lab in [(0.75, "25d put"), (0.50, "ATM"), (0.25, "25d call")]:
            ax.axvline(x, color="grey", linestyle=":", linewidth=0.9, alpha=0.8)
            ax.annotate(lab, (x, 0.97), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=8, color="#555")

        # Reversed, so strike still increases to the right as it does on the
        # log-moneyness plots: a high call delta is a low strike.
        ax.invert_xaxis()
        ax.set_xlabel("Call delta of the strike  (puts shown as $\\Delta_c=\\Delta_p+e^{-rT}$)")
        ax.set_ylabel("Implied volatility")
        ax.yaxis.set_major_formatter(_pct_formatter())
        ax.set_title(f"{ticker} - Smile in delta space")
        ax.legend(title="Expiration", loc="upper left")

    return fig


def plot_put_skew(df, ticker="SPY", iv_col=None, max_dtes=5):
    """OTM put skew across expirations, in log-moneyness.

    The front expiry is steepest. Plotted in k rather than K/S so the slopes
    can honestly be compared to each other.
    """
    data, iv_col = _prepare(df, iv_col, otm_only=True)
    puts = data[data["option_type"] == "put"]
    dtes = sorted(puts["days_to_expiration"].unique())[:max_dtes]

    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(dtes) - 1, 1)) for i in range(len(dtes))]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for dte, color in zip(dtes, colors):
            sub = puts[puts["days_to_expiration"] == dte].sort_values("k")
            ax.plot(sub["k"], sub[iv_col], color=color, linewidth=1.8,
                    label=f"{dte} DTE")

        ax.axvline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Log-moneyness  k = ln(K/F)")
        ax.set_ylabel("Implied volatility")
        ax.yaxis.set_major_formatter(_pct_formatter())
        ax.set_title(f"{ticker} - OTM put skew by expiration")
        ax.legend(title="Expiration", loc="upper right")

    return fig


def plot_atm_term_structure(df, ticker="SPY", iv_col=None):
    """ATM volatility and ATM total variance against maturity.

    Total variance is the second panel because it is what has to be
    non-decreasing in T for the surface to be free of calendar arbitrage; the
    volatility panel alone cannot show that.
    """
    data, iv_col = _prepare(df, iv_col, otm_only=True)
    atm = data[data["k"].abs() <= ATM_K_BAND]

    term = (
        atm.groupby("days_to_expiration")
        .agg(atm_iv=(iv_col, "median"), T=("T", "first"))
        .reset_index()
        .sort_values("days_to_expiration")
    )
    term["total_var"] = term["atm_iv"] ** 2 * term["T"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

        axes[0].plot(term["days_to_expiration"], term["atm_iv"], color="#37474F",
                     marker="o", markersize=5, linewidth=1.8)
        axes[0].set_ylabel("ATM implied volatility (median)")
        axes[0].yaxis.set_major_formatter(_pct_formatter(1))
        axes[0].set_title("ATM volatility")

        axes[1].plot(term["days_to_expiration"], term["total_var"], color="#00695C",
                     marker="s", markersize=5, linewidth=1.8)
        axes[1].set_ylabel(r"ATM total variance  $w=\sigma^2 T$")
        axes[1].set_title("ATM total variance (must not decrease)")

        for ax in axes:
            ax.set_xlabel("Days to expiration")

        fig.suptitle(f"{ticker} - ATM term structure", y=1.02)

    return fig


def plot_skew_robustness(raw_df, ticker="SPY", target_dte=30, settings=None):
    """The same skew under three filter settings.

    Pre-empts the obvious objection that the filters chose the conclusion.

    Ideally this runs on the unfiltered chain, since a sweep should be able to
    loosen as well as tighten. The committed reference has already been through
    one pass of the original filters (absolute spread <= $0.50, volume >= 10),
    so "loose" here relaxes the current filters only within that envelope
    rather than reaching contracts the first pass discarded. It still shows
    whether the current thresholds are what produce the shape, which is the
    question being asked.
    """
    from data_collection import clean_options_data
    from iv_calculation import calculate_iv_surface

    if settings is None:
        settings = [
            ("strict", dict(max_rel_spread=0.05, min_oi=500, min_mid_price=0.20)),
            ("baseline", dict()),
            ("loose", dict(max_rel_spread=0.20, min_oi=10, min_mid_price=0.05)),
        ]

    curves = {}
    for label, kwargs in settings:
        cleaned = clean_options_data(raw_df, verbose=False, **kwargs)
        if cleaned.empty:
            print(f"  {label}: no rows survived")
            continue
        with_iv = calculate_iv_surface(cleaned)
        data, iv_col = _prepare(with_iv, otm_only=True)
        if data.empty:
            continue
        dtes = data["days_to_expiration"].unique()
        chosen = dtes[np.argmin(np.abs(dtes - target_dte))]
        sub = data[data["days_to_expiration"] == chosen].sort_values("k")
        curves[label] = (sub["k"].to_numpy(), sub[iv_col].to_numpy(), int(chosen))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.4, 5))

        # Drawn widest first, so three coincident curves read as concentric
        # bands instead of as one line hiding two others. Them agreeing is the
        # result, and the figure has to show that there are three of them.
        styling = [("#EF9A9A", 5.0, 1.0), ("#37474F", 2.4, 1.0), ("#1565C0", 1.0, 1.0)]
        base = curves.get("baseline")

        for (label, (k, iv, dte)), (color, lw, alpha) in zip(curves.items(), styling):
            note = f"{label}  (n={len(k)}, DTE={dte})"
            if base is not None and label != "baseline" and len(base[0]) > 1:
                dev = 100 * np.abs(np.interp(k, base[0], base[1]) - iv).max()
                note += f", max {dev:.2f} vol pt from baseline"
            ax.plot(k, iv, color=color, linewidth=lw, alpha=alpha, label=note,
                    solid_capstyle="round")

        ax.axvline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Log-moneyness  k = ln(K/F)")
        ax.set_ylabel("Implied volatility")
        ax.yaxis.set_major_formatter(_pct_formatter())
        ax.set_title(f"{ticker} - the skew is not an artifact of the filters")
        ax.legend(title="Filter setting", loc="lower left", fontsize=8)

    return fig


def plot_all(df, ticker="SPY", save=True, show=False, raw_df=None):
    """Generate the standard figures. Returns a name -> Figure mapping."""
    plots = {
        "iv_surface": (plot_iv_surface, f"{ticker.lower()}_iv_surface.png"),
        "volatility_smile": (plot_volatility_smile, f"{ticker.lower()}_volatility_smile.png"),
        "put_skew": (plot_put_skew, f"{ticker.lower()}_put_skew.png"),
        "atm_term_structure": (plot_atm_term_structure, f"{ticker.lower()}_atm_term_structure.png"),
        "before_after": (plot_before_after_smile, f"{ticker.lower()}_before_after_smile.png"),
        "delta_smile": (plot_delta_smile, f"{ticker.lower()}_delta_smile.png"),
    }

    figures = {}
    for name, (func, filename) in plots.items():
        print(f"Generating: {name} ...")
        try:
            fig = func(df, ticker=ticker)
            if save:
                _save(fig, filename)
            if show:
                plt.show()
            else:
                plt.close(fig)
            figures[name] = fig
        except Exception as exc:
            print(f"  WARNING - {name} failed: {exc}")

    if raw_df is not None:
        print("Generating: skew_robustness ...")
        try:
            fig = plot_skew_robustness(raw_df, ticker=ticker)
            if save:
                _save(fig, f"{ticker.lower()}_skew_robustness.png")
            plt.close(fig)
            figures["skew_robustness"] = fig
        except Exception as exc:
            print(f"  WARNING - skew_robustness failed: {exc}")

    return figures


def main():
    processed_dir = os.path.join(_SRC_DIR, "..", "data", "processed")
    raw_dir = os.path.join(_SRC_DIR, "..", "data", "raw")

    iv_files = glob.glob(os.path.join(processed_dir, "*_with_iv.csv"))
    raw_files = [f for f in glob.glob(os.path.join(processed_dir, "*.csv"))
                 if "_with_iv" not in f]

    if iv_files:
        source = max(iv_files, key=os.path.getmtime)
    elif raw_files:
        source = max(raw_files, key=os.path.getmtime)
        print("NOTE: no _with_iv.csv found; using the Yahoo impliedVolatility "
              "column. Run iv_calculation.py for our own.")
    else:
        print(f"No CSV files found in {processed_dir}. Run data_collection.py first.")
        return

    print(f"Loading {source}")
    df = pd.read_csv(source)
    print(f"Loaded {len(df):,} rows.")

    # Prefer an unfiltered chain for the filter sweep; fall back to the
    # committed reference, which has been filtered once already.
    chains = glob.glob(os.path.join(raw_dir, "*.csv"))
    sweep_src = None
    if chains:
        # The least filtered chain available, by row count. A sweep has to be
        # able to loosen as well as tighten, so more rows is strictly better
        # here regardless of which snapshot they came from.
        sweep_src = max(chains, key=lambda f: sum(1 for _ in open(f)))
    elif raw_files:
        # Prefer a file the pipeline has not already filtered: one without a
        # forward column is an input, one with is a stage-1 output, and
        # sweeping filters over an output only re-narrows what survived.
        inputs = [f for f in raw_files
                  if "forward" not in pd.read_csv(f, nrows=0).columns]
        sweep_src = (max(inputs, key=os.path.getmtime) if inputs
                     else max(raw_files, key=os.path.getmtime))
        print(f"NOTE: no unfiltered chain in data/raw/; sweeping filters over "
              f"{os.path.basename(sweep_src)}, which has already been "
              f"filtered once.")
    raw_df = pd.read_csv(sweep_src) if sweep_src else None

    ticker = os.path.basename(source).split("_")[0].upper()
    plot_all(df, ticker=ticker, save=True, show=False, raw_df=raw_df)
    print("\nDone. Check data/figures/ for outputs.")


if __name__ == "__main__":
    main()
