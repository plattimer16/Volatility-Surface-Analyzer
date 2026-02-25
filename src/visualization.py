"""
Volatility surface visualization.

Generates four publication-quality plots from a cleaned options DataFrame:

1. ``plot_iv_surface``      — 3-D implied-volatility surface (moneyness × DTE)
2. ``plot_volatility_smile``— 2-D smile for a chosen expiration (~30 DTE)
3. ``plot_put_skew``        — Moneyness vs IV for puts (volatility skew)
4. ``plot_atm_term_structure`` — ATM implied volatility across expirations

All plots are saved to ``data/figures/`` and displayed if a screen is available.

Usage (CLI)::

    python src/visualization.py

or in a notebook / script::

    from src.visualization import plot_all
    df = pd.read_csv("data/processed/spy_options_with_iv.csv")
    plot_all(df, ticker="SPY")
"""

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for 3-D projection
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# Paths and output directory
# ---------------------------------------------------------------------------

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_SRC_DIR, "..", "data", "figures")

# ---------------------------------------------------------------------------
# Style defaults — tweak here to change the look of all plots at once
# ---------------------------------------------------------------------------

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

# Moneyness window kept for most plots — excludes deep ITM/OTM noise
MONEYNESS_MIN = 0.80
MONEYNESS_MAX = 1.20

# How close to ATM is considered "at-the-money"
ATM_BAND = 0.02  # ±2 % of spot


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_iv_column(df: pd.DataFrame) -> str:
    """Return the name of the implied-volatility column to use.

    Preference order:
    1. ``implied_volatility_bs``  (our own Black-Scholes solver)
    2. ``impliedVolatility``      (Yahoo Finance field, already in the CSV)

    Raises ``KeyError`` if neither is present.
    """
    for col in ("implied_volatility_bs", "impliedVolatility"):
        if col in df.columns:
            return col
    raise KeyError(
        "DataFrame has no IV column. Expected 'implied_volatility_bs' or "
        "'impliedVolatility'. Run iv_calculation.py first, or use the raw "
        "yfinance data which includes 'impliedVolatility'."
    )


def _filter(df: pd.DataFrame, iv_col: str) -> pd.DataFrame:
    """Drop rows with missing IV or extreme moneyness, then return a copy."""
    mask = (
        df[iv_col].notna()
        & (df[iv_col] > 0)
        & (df["moneyness"] >= MONEYNESS_MIN)
        & (df["moneyness"] <= MONEYNESS_MAX)
    )
    return df.loc[mask].copy()


def _pct_formatter() -> mticker.FuncFormatter:
    """Return a matplotlib formatter that renders decimals as '20 %'."""
    return mticker.FuncFormatter(lambda x, _: f"{x:.0%}")


def _save(fig: plt.Figure, filename: str) -> str:
    """Save *fig* to ``data/figures/<filename>`` and return the full path."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Plot 1 — 3-D Implied-Volatility Surface
# ---------------------------------------------------------------------------

def plot_iv_surface(
    df: pd.DataFrame,
    ticker: str = "SPY",
    iv_col: str | None = None,
) -> plt.Figure:
    """Generate a 3-D surface of implied volatility over moneyness and DTE.

    The raw data sits on an irregular (moneyness, DTE) grid, so we
    interpolate onto a regular mesh using ``scipy.interpolate.griddata``
    before calling ``Axes3D.plot_surface``.

    Parameters
    ----------
    df : pd.DataFrame
        Options data with columns ``moneyness``, ``days_to_expiration``,
        ``option_type``, and an IV column.
    ticker : str
        Ticker symbol shown in the plot title.
    iv_col : str, optional
        Name of the IV column.  Auto-detected if omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    iv_col = iv_col or _resolve_iv_column(df)
    data = _filter(df, iv_col)

    # Use only calls for the surface (calls and puts should give similar IVs
    # near ATM; mixing them creates duplicate grid points that confuse griddata)
    calls = data[data["option_type"] == "call"].copy()

    if calls.empty:
        raise ValueError("No valid call data found for the IV surface plot.")

    # Build a regular mesh
    x_vals = np.linspace(calls["moneyness"].min(), calls["moneyness"].max(), 60)
    y_vals = np.linspace(calls["days_to_expiration"].min(), calls["days_to_expiration"].max(), 60)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)

    # Interpolate scattered (moneyness, DTE) → IV onto the regular grid
    grid_z = griddata(
        points=calls[["moneyness", "days_to_expiration"]].values,
        values=calls[iv_col].values,
        xi=(grid_x, grid_y),
        method="cubic",   # smooth surface; falls back to nearest outside convex hull
    )

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            grid_x,
            grid_y,
            grid_z,
            cmap="RdYlGn_r",   # red = high vol, green = low vol
            alpha=0.85,
            linewidth=0,
            antialiased=True,
        )

        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label("Implied Volatility", fontsize=9)
        cbar.ax.yaxis.set_major_formatter(_pct_formatter())

        ax.set_xlabel("Moneyness (K/S)", labelpad=8)
        ax.set_ylabel("Days to Expiration", labelpad=8)
        ax.set_zlabel("Implied Volatility", labelpad=8)
        ax.zaxis.set_major_formatter(_pct_formatter())

        ax.set_title(f"{ticker} — Implied Volatility Surface (Calls)", pad=12)

        # Viewing angle that makes skew clearly visible
        ax.view_init(elev=28, azim=-55)

    return fig


# ---------------------------------------------------------------------------
# Plot 2 — 2-D Volatility Smile (single expiration)
# ---------------------------------------------------------------------------

def plot_volatility_smile(
    df: pd.DataFrame,
    target_dte: int = 30,
    ticker: str = "SPY",
    iv_col: str | None = None,
) -> plt.Figure:
    """Plot the volatility smile for the expiration nearest to *target_dte*.

    Calls and puts are overlaid on the same axes so the skew between the two
    option types is immediately visible.

    Parameters
    ----------
    df : pd.DataFrame
        Options data.
    target_dte : int
        Desired days-to-expiration.  The closest available expiration is used.
    ticker : str
        Ticker symbol shown in the title.
    iv_col : str, optional
        Name of the IV column.  Auto-detected if omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    iv_col = iv_col or _resolve_iv_column(df)
    data = _filter(df, iv_col)

    # Find the expiration whose DTE is closest to target_dte
    available_dtes = data["days_to_expiration"].unique()
    chosen_dte = available_dtes[np.argmin(np.abs(available_dtes - target_dte))]
    exp_data = data[data["days_to_expiration"] == chosen_dte]

    print(f"  Smile plot: using DTE={chosen_dte} (requested {target_dte})")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        colors = {"call": "#2196F3", "put": "#E53935"}
        markers = {"call": "o", "put": "s"}

        for opt_type in ["call", "put"]:
            subset = exp_data[exp_data["option_type"] == opt_type].sort_values("moneyness")
            ax.plot(
                subset["moneyness"],
                subset[iv_col],
                color=colors[opt_type],
                marker=markers[opt_type],
                markersize=4,
                linewidth=1.8,
                label=f"{opt_type.capitalize()}s",
            )

        # Mark the ATM point (moneyness = 1)
        ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, label="ATM (K/S=1)")

        ax.set_xlabel("Moneyness (K / S)")
        ax.set_ylabel("Implied Volatility")
        ax.yaxis.set_major_formatter(_pct_formatter())
        ax.set_title(f"{ticker} — Volatility Smile  (DTE ≈ {chosen_dte})")
        ax.legend()

    return fig


# ---------------------------------------------------------------------------
# Plot 3 — Put Volatility Skew (moneyness vs IV)
# ---------------------------------------------------------------------------

def plot_put_skew(
    df: pd.DataFrame,
    ticker: str = "SPY",
    iv_col: str | None = None,
    max_dtes: int = 5,
) -> plt.Figure:
    """Show volatility skew for puts across multiple expirations.

    Each expiration is a separate line so the term structure of skew is
    visible.  The classic downward slope to the left (OTM puts → high IV)
    illustrates the "crash premium" embedded in the options market.

    Parameters
    ----------
    df : pd.DataFrame
        Options data.
    ticker : str
        Ticker symbol.
    iv_col : str, optional
        Name of the IV column.  Auto-detected if omitted.
    max_dtes : int
        How many expirations to overlay.  Sorted by DTE, shortest first.

    Returns
    -------
    matplotlib.figure.Figure
    """
    iv_col = iv_col or _resolve_iv_column(df)
    data = _filter(df, iv_col)

    puts = data[data["option_type"] == "put"].copy()
    dtes = sorted(puts["days_to_expiration"].unique())[:max_dtes]

    # Use a sequential colormap so shorter DTEs are darker
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(dtes) - 1, 1)) for i in range(len(dtes))]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        for dte, color in zip(dtes, colors):
            subset = puts[puts["days_to_expiration"] == dte].sort_values("moneyness")
            ax.plot(
                subset["moneyness"],
                subset[iv_col],
                color=color,
                linewidth=1.8,
                label=f"{dte} DTE",
            )

        # Shade the OTM put region (moneyness < 1) to highlight the skew area
        ax.axvspan(MONEYNESS_MIN, 1.0, alpha=0.05, color="red", label="OTM puts")
        ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_xlabel("Moneyness (K / S)")
        ax.set_ylabel("Implied Volatility")
        ax.yaxis.set_major_formatter(_pct_formatter())
        ax.set_title(f"{ticker} — Put Volatility Skew by Expiration")
        ax.legend(title="Expiration", loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# Plot 4 — ATM Implied Volatility Term Structure
# ---------------------------------------------------------------------------

def plot_atm_term_structure(
    df: pd.DataFrame,
    ticker: str = "SPY",
    iv_col: str | None = None,
) -> plt.Figure:
    """Plot ATM implied volatility as a function of days to expiration.

    "ATM" is defined as options whose moneyness falls within
    ``1.0 ± ATM_BAND``.  The median IV within that band is taken for each
    expiration.

    With a single-date snapshot this shows the *term structure* of volatility
    — the same information as a time-series of ATM vol but captured across
    expirations rather than calendar dates.  When multiple data files are
    available, ``main()`` can stack them to produce a true time series.

    Parameters
    ----------
    df : pd.DataFrame
        Options data.
    ticker : str
        Ticker symbol.
    iv_col : str, optional
        Name of the IV column.  Auto-detected if omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    iv_col = iv_col or _resolve_iv_column(df)
    data = _filter(df, iv_col)

    # Restrict to near-ATM options
    atm = data[
        (data["moneyness"] >= 1.0 - ATM_BAND) &
        (data["moneyness"] <= 1.0 + ATM_BAND)
    ]

    # Median IV per (option_type, DTE) cell — robust to outliers
    term = (
        atm.groupby(["option_type", "days_to_expiration"])[iv_col]
        .median()
        .reset_index()
        .rename(columns={iv_col: "atm_iv"})
        .sort_values("days_to_expiration")
    )

    colors = {"call": "#2196F3", "put": "#E53935"}
    markers = {"call": "o", "put": "s"}

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        for opt_type in ["call", "put"]:
            subset = term[term["option_type"] == opt_type]
            ax.plot(
                subset["days_to_expiration"],
                subset["atm_iv"],
                color=colors[opt_type],
                marker=markers[opt_type],
                markersize=5,
                linewidth=1.8,
                label=f"{opt_type.capitalize()}s",
            )

        ax.set_xlabel("Days to Expiration")
        ax.set_ylabel("ATM Implied Volatility (median)")
        ax.yaxis.set_major_formatter(_pct_formatter())
        ax.set_title(f"{ticker} — ATM Implied Volatility Term Structure")
        ax.legend()

    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all four plots in one call
# ---------------------------------------------------------------------------

def plot_all(
    df: pd.DataFrame,
    ticker: str = "SPY",
    save: bool = True,
    show: bool = False,
) -> dict[str, plt.Figure]:
    """Generate all four standard volatility plots.

    Parameters
    ----------
    df : pd.DataFrame
        Options data returned by ``data_collection.clean_options_data`` or
        enriched with ``iv_calculation.calculate_iv_surface``.
    ticker : str
        Ticker symbol used in titles and file names.
    save : bool
        If ``True``, write each figure to ``data/figures/``.
    show : bool
        If ``True``, call ``plt.show()`` for each figure (useful in
        notebooks or interactive sessions).

    Returns
    -------
    dict
        Mapping of plot name → ``matplotlib.figure.Figure``.
    """
    plots = {
        "iv_surface": (plot_iv_surface, f"{ticker.lower()}_iv_surface.png"),
        "volatility_smile": (plot_volatility_smile, f"{ticker.lower()}_volatility_smile.png"),
        "put_skew": (plot_put_skew, f"{ticker.lower()}_put_skew.png"),
        "atm_term_structure": (plot_atm_term_structure, f"{ticker.lower()}_atm_term_structure.png"),
    }

    figures = {}
    for name, (func, filename) in plots.items():
        print(f"Generating: {name} …")
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
            # Don't abort the whole run if one plot fails
            print(f"  WARNING — {name} failed: {exc}")

    return figures


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Load the most recent options CSV and generate all plots."""
    processed_dir = os.path.join(_SRC_DIR, "..", "data", "processed")

    # Prefer a file that already has our BS-derived IV; fall back to raw clean file
    iv_files = sorted(glob.glob(os.path.join(processed_dir, "*_with_iv.csv")))
    raw_files = sorted(glob.glob(os.path.join(processed_dir, "*.csv")))

    # Exclude _with_iv files from raw_files to avoid double-listing
    raw_files = [f for f in raw_files if "_with_iv" not in f]

    if iv_files:
        source = iv_files[-1]
    elif raw_files:
        source = raw_files[-1]
        print(
            "NOTE: No _with_iv.csv found; using Yahoo Finance impliedVolatility.\n"
            "      Run iv_calculation.py for Black-Scholes IV instead."
        )
    else:
        print(f"No CSV files found in {processed_dir}. Run data_collection.py first.")
        return

    print(f"Loading {source}")
    df = pd.read_csv(source)
    print(f"Loaded {len(df):,} rows.")

    # Infer ticker from filename (e.g. "spy_options_…" → "SPY")
    basename = os.path.basename(source)
    ticker = basename.split("_")[0].upper()

    plot_all(df, ticker=ticker, save=True, show=False)
    print("\nDone. Check data/figures/ for outputs.")


if __name__ == "__main__":
    main()
