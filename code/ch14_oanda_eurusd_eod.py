from pathlib import Path  # filesystem paths for output locations
from datetime import datetime, timedelta, timezone  # date and time handling
import sys  # access to Python's module search path

import numpy as np  # numerical arrays and statistics
import pandas as pd  # tabular time-series data structures
import matplotlib.pyplot as plt  # plotting library for figures

"""
Python & AI for Algorithmic Trading
Chapter 14 -- Working with Oanda Demo Accounts

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Oanda EUR_USD end-of-day diagnostics for Chapter 14.

This script uses the lightweight Oanda client from ``wrappers/tpqoa.py`` to
retrieve daily mid-price candles for the EUR_USD instrument over a recent
history window. It then computes simple return-based statistics, generates a
compact diagnostic figure, and writes a LaTeX helper file with macros so that
the Chapter 14 manuscript can stay in sync with the underlying data.

The script performs three main tasks.

1. Fetch daily EUR_USD candles from the Oanda practice API using credentials
   stored in ``code/creds.py``.
2. Compute simple daily returns, sample mean and volatility, and a handful of
   summarising numbers such as the number of trading days and the date range.
3. Create a figure with the normalised EUR_USD price path and a histogram of
   daily returns, and export a small macro file with statistics used in the
   text and tables.

It is intentionally compact and fully commented so that it can serve as a
template for other broker-specific diagnostics later in the book.
"""

# Ensure that the project root (which contains the ``wrappers`` package) is
# on the Python module search path even when executing this file via
# ``python code/ch14_oanda_eurusd_eod.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # add once, preserve existing order
    sys.path.insert(0, str(PROJECT_ROOT))

from wrappers.tpqoa import OandaClient  # minimal Oanda API wrapper

plt.style.use("seaborn-v0_8")  # consistent plotting style across figures


def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """Compute simple percentage returns from a price series."""
    prices = prices.astype(float)  # ensure numeric dtype
    rets = prices.pct_change(fill_method=None)  # simple returns R_t
    return rets.dropna()  # drop initial NaN observation


def write_tex_macros(
    stats: dict[str, float | int | str],
    outfile: str="figures/ch14_oanda_stats.tex",
) -> None:
    """Write LaTeX macros with Chapter 14 summary statistics."""
    lines: list[str] = []  # collect macro definition strings

    lines.append(
        "\\newcommand{\\chFourteenOandaNumObs}{"
        f"{stats['n_obs']}"
        "}\n"
    )  # number of daily observations
    lines.append(
        "\\newcommand{\\chFourteenOandaStartDate}{"
        f"{stats['start_date']}"
        "}\n"
    )  # first trading date in ISO format
    lines.append(
        "\\newcommand{\\chFourteenOandaEndDate}{"
        f"{stats['end_date']}"
        "}\n"
    )  # last trading date in ISO format
    lines.append(
        "\\newcommand{\\chFourteenOandaMeanRet}{"
        f"{stats['mean_ret']:.4f}"
        "}\n"
    )  # sample mean daily return
    lines.append(
        "\\newcommand{\\chFourteenOandaVol}{"
        f"{stats['vol_ret']:.4f}"
        "}\n"
    )  # sample daily return volatility

    path = Path(outfile)  # macro file location
    path.write_text("".join(lines), encoding="utf8")  # save macros


def plot_eurusd_diagnostics(
    candles: pd.DataFrame,
    returns: pd.Series,
    outfile: str="figures/ch14_oanda_eurusd_daily.pdf",
) -> None:
    """Plot normalised EUR_USD price path and histogram of daily returns."""
    fig, (ax_price, ax_ret) = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.6),
        sharex=False,
    )  # stacked panels for price and returns

    cmap = plt.cm.coolwarm  # shared colour map

    # Top panel: normalised close prices
    normed = candles["c"] / candles["c"].iloc[0]  # start at 1.0
    ax_price.plot(
        normed.index,
        normed.values,
        color=cmap(0.10),
        lw=1.0,
        label="EUR_USD normalised close",
    )
    ax_price.set_ylabel("price (normalised)")  # y-axis label
    ax_price.set_title("EUR_USD: Oanda daily mid-price diagnostics")  # title
    ax_price.legend(loc="upper left")  # legend for price series
    ax_price.grid(True, alpha=0.3)  # light grid for readability

    # Bottom panel: histogram of daily returns
    ax_ret.hist(
        returns.values,
        bins=60,
        density=False,
        color=cmap(0.85),
        alpha=0.8,
    )  # daily return histogram
    ax_ret.set_xlabel("daily simple return")  # x-axis label
    ax_ret.set_ylabel("frequency")  # y-axis label
    ax_ret.grid(True, alpha=0.3)  # light grid

    fig.tight_layout()  # reduce unused whitespace
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # free Matplotlib resources


def main() -> None:
    """Run the Oanda EUR_USD daily diagnostics workflow."""
    client = OandaClient.from_creds()  # configured Oanda v20 client

    try:
        end = datetime.now(timezone.utc)  # end date for history window
        start = end - timedelta(days=365)  # look back roughly one calendar year

        candles = client.get_candles(
            "EUR_USD",
            start=start,
            end=end,
            granularity="D",
            price="M",
        )  # daily mid-price candles
        if candles.empty:
            raise RuntimeError("No EUR_USD candles retrieved from Oanda.")

        closes = candles["c"]  # closing mid-prices per day
        rets = compute_simple_returns(closes)  # simple daily returns

        n_obs = int(rets.shape[0])  # number of usable return observations
        mean_ret = float(rets.mean())  # sample mean daily return
        vol_ret = float(rets.std(ddof=1))  # sample daily return volatility

        start_date = closes.index.min().date().isoformat()  # first trading date
        end_date = closes.index.max().date().isoformat()  # last trading date

        stats = {
            "n_obs": n_obs,
            "start_date": start_date,
            "end_date": end_date,
            "mean_ret": mean_ret,
            "vol_ret": vol_ret,
        }  # dictionary of summary statistics

        write_tex_macros(stats=stats)  # keep LaTeX macros in sync with data
        plot_eurusd_diagnostics(candles=candles, returns=rets)  # create figure

        # Print a concise console summary for interactive inspection.
        print("Oanda EUR_USD daily diagnostics (mid prices):")  # header
        print(f"  observations : {n_obs}")  # number of daily returns
        print(f"  date range   : {start_date} to {end_date}")  # coverage
        print(f"  mean return  : {mean_ret: .5f}")  # average daily return
        print(f"  daily vol    : {vol_ret: .5f}")  # daily return volatility
        print("  last 3 candles:")  # trailing sample for quick inspection
        print(candles[["o", "h", "l", "c", "volume"]].tail(3))
    except Exception as exc:
        print("Error while running Oanda EUR_USD diagnostics:")  # header
        print(f"  {type(exc).__name__}: {exc}")  # exception details


if __name__ == "__main__":
    main()  # execute diagnostics when run as a script
