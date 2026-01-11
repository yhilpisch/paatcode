from pathlib import Path  # filesystem paths for relative imports
import sys  # access to Python's module search path

import numpy as np  # numerical helpers for returns and autocorrelations
import pandas as pd  # tabular time-series structures
import matplotlib.pyplot as plt  # plotting library for figures

"""
Python & AI for Algorithmic Trading
Chapter 6 -- EODHD API: From Static CSV to Live Data

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

EODHD end-of-day and intraday autocorrelation diagnostics for Chapter 6.

This script illustrates how the lightweight EODHD client from
``wrappers/tpqeod.py`` can be used to retrieve historical data for a single
symbol, compute log-returns, and inspect simple autocorrelation statistics.

The script performs three main tasks.

1. Fetch a daily end-of-day history for ``SPY.US`` and compute daily
   log-returns.
2. Fetch a short intraday history for ``SPY.US`` with five-minute bars and
   compute log-returns for those bars.
3. Compute lag-one autocorrelations for both series and visualise the price
   path together with a histogram of daily log-returns.

It is intentionally compact but fully commented so that it can serve as a
template for later chapters that build more elaborate diagnostics.
"""

# Ensure that the project root (which contains the ``wrappers`` package) is
# on the Python module search path even when executing this file via
# ``python code/ch06_eodhd_autocorr_demo.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # add once, preserve existing order
    sys.path.insert(0, str(PROJECT_ROOT))

from wrappers.tpqeod import EODHDClient  # minimal EODHD API wrapper

plt.style.use("seaborn-v0_8")  # consistent figure style across the book


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log-returns from a price series indexed by time."""
    prices = prices.astype(float)  # ensure numeric dtype
    log_ret = np.log(prices / prices.shift(1))  # log-returns from prices
    return log_ret.dropna()  # drop first NaN observation


def main() -> None:
    """Run the EODHD autocorrelation diagnostics."""
    client = EODHDClient.from_creds()  # load API key from eod/creds.py

    # ------------------------------------------------------------------
    # 1. End-of-day history and daily autocorrelation
    # ------------------------------------------------------------------
    eod_df = client.get_eod("SPY", "US")  # daily EOD prices for SPY
    eod_df.columns = [c.strip().lower() for c in eod_df.columns]  # normalise
    daily_log_ret = compute_log_returns(eod_df["close"])  # daily log-returns

    daily_acf1 = float(daily_log_ret.autocorr(lag=1))  # lag-1 autocorrelation

    # ------------------------------------------------------------------
    # 2. Intraday history and autocorrelation for five-minute bars
    # ------------------------------------------------------------------
    intraday_df = client.get_intraday("SPY", "US", period="5m")  # 5m bars
    intraday_df.columns = [c.strip().lower() for c in intraday_df.columns]
    intraday_log_ret = compute_log_returns(  # intraday log-returns from close
        intraday_df["close"]
    )

    intraday_acf1 = float(intraday_log_ret.autocorr(lag=1))  # lag-1 ACF

    # ------------------------------------------------------------------
    # 2b. Write LaTeX helper macros with the summary statistics
    # ------------------------------------------------------------------
    lines = []  # collect macro definition strings
    lines.append(
        "\\newcommand{\\chSixSpyDailyAcfOne}{"
        f"{daily_acf1: .3f}"
        "}\n"
    )  # daily lag-1 autocorrelation for SPY
    lines.append(
        "\\newcommand{\\chSixSpyIntradayAcfOne}{"
        f"{intraday_acf1: .3f}"
        "}\n"
    )  # 5m lag-1 autocorrelation for SPY

    stats_path = Path("figures/ch06_eodhd_stats.tex")  # output location
    stats_path.write_text("".join(lines), encoding="utf8")  # save macros

    # ------------------------------------------------------------------
    # 3. Simple visualisation: price path and daily return histogram
    # ------------------------------------------------------------------
    fig, (ax_price, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.8),
        sharex=False,
    )  # stacked panels

    # Top panel: daily close prices (normalised)
    normed_price = eod_df["close"] / eod_df["close"].iloc[0]  # start at 1.0
    ax_price.plot(
        normed_price.index,
        normed_price.values,
        color=plt.cm.coolwarm(0.10),
        lw=1.0,
        label="SPY normalised close",
    )
    ax_price.set_ylabel("price (normalised)")  # y-axis label
    ax_price.set_title(
        "SPY: EOD price path and intraday return distribution"
    )  # title
    ax_price.legend(loc="upper left")  # legend for price series
    ax_price.grid(True, alpha=0.3)  # light grid

    # Bottom panel: histogram of daily log-returns
    ax_hist.hist(
        daily_log_ret.values,
        bins=80,
        density=True,
        color=plt.cm.coolwarm(0.85),
        alpha=0.8,
    )  # daily return histogram
    ax_hist.set_xlabel("daily log-return")  # x-axis label
    ax_hist.set_ylabel("frequency")  # y-axis label
    ax_hist.grid(True, alpha=0.3)  # light grid

    fig.tight_layout()  # reduce unused whitespace
    fig.savefig("figures/ch06_eodhd_autocorr_demo.pdf", bbox_inches="tight")
    plt.close(fig)  # free Matplotlib resources

    # Print simple summary statistics to the console for inspection.
    print(f"Daily lag-1 autocorrelation (SPY): {daily_acf1: .3f}")  # summary
    print(f"5m lag-1 autocorrelation (SPY):    {intraday_acf1: .3f}")  # summary


if __name__ == "__main__":
    main()  # run diagnostics when executed as a script
