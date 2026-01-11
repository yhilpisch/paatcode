from pathlib import Path  # filesystem paths for output locations
from datetime import datetime, timedelta, timezone  # date and time handling
import sys  # access to Python's module search path

import numpy as np  # numerical arrays and statistics
import pandas as pd  # tabular time-series data structures
import matplotlib.pyplot as plt  # plotting backend for figures

"""
Python & AI for Algorithmic Trading
Chapter 15 -- IG Markets and the Anatomy of an API Wrapper

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

IG Markets EUR/USD end-of-day diagnostics for Chapter 15.

This script uses the lightweight IG client from ``wrappers/tpqig.py`` to
retrieve daily mid-price candles for the EUR/USD contract in the IG demo
environment. It then computes simple return-based statistics, generates a
compact diagnostic figure, and writes a LaTeX helper file with macros so
that the Chapter 15 manuscript can stay in sync with the underlying data.

The script performs three main tasks.

1. Fetch daily EUR/USD candles from the IG demo API using credentials stored
   in ``ref/trading_ig_config.py``.
2. Compute simple daily returns, sample mean and volatility, and a handful of
   summarising numbers such as the number of trading days and the date range.
3. Create a figure with the normalised EUR/USD price path and a histogram of
   daily returns, and export a small macro file with statistics used in the
   text and tables.

The code is intentionally compact and fully commented so that it can serve
as a template for other IG-based diagnostics.
"""

# Ensure that the project root (which contains the ``wrappers`` package) is
# on the Python module search path even when executing this file via
# ``python code/ch15_ig_eurusd_eod.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # prefer project-local wrappers

from wrappers.tpqig import IGClient  # minimal IG Markets API wrapper

plt.style.use("seaborn-v0_8")  # consistent plotting style across figures


def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """Compute simple percentage returns from a price series."""
    prices = prices.astype(float)
    rets = prices.pct_change(fill_method=None)
    return rets.dropna()


def write_tex_macros(
    stats: dict[str, float | int | str],
    outfile: str = "figures/ch15_ig_stats.tex",
) -> None:
    """Write LaTeX macros with Chapter 15 summary statistics."""
    lines: list[str] = []

    lines.append(
        "\\newcommand{\\chFifteenIgNumObs}{"
        f"{stats['n_obs']}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chFifteenIgStartDate}{"
        f"{stats['start_date']}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chFifteenIgEndDate}{"
        f"{stats['end_date']}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chFifteenIgMeanRet}{"
        f"{stats['mean_ret']:.4f}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chFifteenIgVol}{"
        f"{stats['vol_ret']:.4f}"
        "}\n"
    )

    path = Path(outfile)
    path.write_text("".join(lines), encoding="utf8")


def plot_eurusd_diagnostics(
    candles: pd.DataFrame,
    returns: pd.Series,
    outfile: str = "figures/ch15_ig_eurusd_daily.pdf",
) -> None:
    """Plot normalised EUR/USD price path and histogram of daily returns."""
    fig, (ax_price, ax_ret) = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.6),
        sharex=False,
    )

    cmap = plt.cm.coolwarm

    normed = candles["mid_close"] / candles["mid_close"].iloc[0]
    ax_price.plot(
        normed.index,
        normed.values,
        color=cmap(0.10),
        lw=1.0,
        label="EUR/USD normalised close (IG demo)",
    )
    ax_price.set_ylabel("price (normalised)")
    ax_price.set_title("EUR/USD: IG Markets daily mid-price diagnostics")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, alpha=0.3)

    ax_ret.hist(
        returns.values,
        bins=60,
        density=False,
        color=cmap(0.85),
        alpha=0.8,
    )
    ax_ret.set_xlabel("daily simple return")
    ax_ret.set_ylabel("frequency")
    ax_ret.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the IG EUR/USD daily diagnostics workflow."""
    client = IGClient.from_creds()

    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=365)

        candles = client.get_candles(
            "EUR/USD",
            granularity="D",
            start=start,
            end=end,
        )
        if candles.empty:
            raise RuntimeError("No EUR/USD candles retrieved from IG Markets.")

        closes = candles["mid_close"]
        rets = compute_simple_returns(closes)

        n_obs = int(rets.shape[0])
        mean_ret = float(rets.mean())
        vol_ret = float(rets.std(ddof=1))

        start_date = closes.index.min().date().isoformat()
        end_date = closes.index.max().date().isoformat()

        stats = {
            "n_obs": n_obs,
            "start_date": start_date,
            "end_date": end_date,
            "mean_ret": mean_ret,
            "vol_ret": vol_ret,
        }

        write_tex_macros(stats=stats)
        plot_eurusd_diagnostics(candles=candles, returns=rets)

        print("IG Markets EUR/USD daily diagnostics (mid prices):")
        print(f"  observations : {n_obs}")
        print(f"  date range   : {start_date} to {end_date}")
        print(f"  mean return  : {mean_ret: .5f}")
        print(f"  daily vol    : {vol_ret: .5f}")
        print("  last 3 candles:")
        cols = ["bid_open", "bid_high", "bid_low", "bid_close", "volume"]
        cols = [c for c in cols if c in candles.columns]
        print(candles[cols].tail(3))
    except Exception as exc:
        print("Error while running IG EUR/USD diagnostics:")  # header
        print(f"  {type(exc).__name__}: {exc}")  # exception details


if __name__ == "__main__":
    main()
