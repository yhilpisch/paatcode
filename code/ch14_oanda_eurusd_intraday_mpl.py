from pathlib import Path  # filesystem paths for output locations
from datetime import datetime, timedelta, timezone  # time window selection
import sys  # access to Python's module search path

import pandas as pd  # tabular time-series data structures
import matplotlib.pyplot as plt  # base plotting backend
import mplfinance as mpf  # Matplotlib-based OHLC plotting

"""
Python & AI for Algorithmic Trading
Chapter 14 -- Working with Oanda Demo Accounts

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Oanda EUR_USD five-second vs 45-second candles for Chapter 14.

This script uses the lightweight Oanda client from ``wrappers/tpqoa.py`` to
retrieve five-second mid-price candles for the EUR_USD instrument over a
short intraday window. It then resamples these candles into 45-second bars,
visualises both series as candlestick charts in stacked panels using
``mplfinance``, and saves the resulting figure to the ``figures/`` directory
for inclusion in the Chapter 14 manuscript.

The script performs three main tasks.

1. Fetch a rolling window of five-second EUR_USD candles from the Oanda
   practice API.
2. Resample the five-second candles to 45-second bars using OHLC aggregation.
3. Generate a candlestick figure that compares the fine-grained and
   aggregated bars and save it as a PDF file.

The code is intentionally compact and fully commented so that it can
serve as an example for other custom-bar visualisations.
"""

# Ensure that the project root (which contains the ``wrappers`` package) is
# on the Python module search path even when executing this file via
# ``python code/ch14_oanda_eurusd_intraday_mpl.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # add once, preserve existing order
    sys.path.insert(0, str(PROJECT_ROOT))

from wrappers.tpqoa import OandaClient  # minimal Oanda API wrapper

plt.style.use("seaborn-v0_8")  # base style for consistent aesthetics


def resample_to_45s(bars_5s: pd.DataFrame) -> pd.DataFrame:
    """Resample five-second candles to 45-second OHLC bars."""
    bars_5s = bars_5s.astype({"o": float, "h": float,
                              "l": float, "c": float})  # ensure floats

    agg = {  # OHLC aggregation rules for 45-second bars
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "volume": "sum",
    }
    bars_45s = bars_5s.resample(
        "45s",
        label="right",
        closed="right",
    ).agg(agg).dropna()
    return bars_45s


def main() -> None:
    """Retrieve S5 candles, resample to 45s, and create candlestick figure."""
    client = OandaClient.from_creds()  # configured Oanda v20 client

    end = datetime.now(timezone.utc)  # end of intraday window
    start = end - timedelta(minutes=30)  # last 30 minutes of data

    bars_5s = client.get_candles(
        "EUR_USD",
        start=start,
        end=end,
        granularity="S5",  # five-second candles
        price="M",  # mid prices
    )
    if bars_5s.empty:
        raise RuntimeError("No EUR_USD S5 candles retrieved from Oanda.")

    bars_45s = resample_to_45s(bars_5s)  # aggregated 45-second bars

    # Focus the visualisation on the most recent segment for readability.
    bars_5s_short = bars_5s.tail(200)  # last few minutes of five-second bars
    xmin = bars_5s_short.index.min()  # earliest timestamp in segment
    xmax = bars_5s_short.index.max()  # latest timestamp in segment

    # Align 45-second bars with the visible five-second window so that both
    # panels share the same time axis. Reindexing places each 45-second bar
    # at the right-edge timestamp on the five-second grid and leaves gaps
    # (NaN rows) elsewhere.
    mask_45s = (bars_45s.index >= xmin) & (bars_45s.index <= xmax)
    bars_45s_short = bars_45s.loc[mask_45s]
    bars_45s_aligned = bars_45s_short.reindex(bars_5s_short.index)

    # Prepare the output path for the figure.
    fig_path = Path("figures/ch14_oanda_eurusd_5s_45s.pdf")
    fig_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory

    # Generate the candlestick figure with mplfinance using two explicit axes.
    bars_5s_plot = bars_5s_short.rename(
        columns={"o": "Open", "h": "High", "l": "Low", "c": "Close"},
    )  # rename columns for mplfinance
    bars_45s_plot = bars_45s_aligned.rename(
        columns={"o": "Open", "h": "High", "l": "Low", "c": "Close"},
    )  # same convention for 45-second bars

    # Use a style with the y-axis on the left for readability.
    mpf_style = mpf.make_mpf_style(base_mpf_style="charles", y_on_right=False)

    fig = plt.figure(figsize=(7.0, 5.0))  # base Matplotlib figure
    ax1 = fig.add_subplot(2, 1, 1)  # top panel for 5s candles
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)  # bottom panel for 45s candles

    mpf.plot(
        bars_5s_plot,
        type="candle",
        style=mpf_style,
        volume=False,
        ax=ax1,
        show_nontrading=False,
    )

    mpf.plot(
        bars_45s_plot,
        type="candle",
        style=mpf_style,
        volume=False,
        ax=ax2,
        show_nontrading=False,
    )

    # Enforce a shared time axis across both panels using integer coordinates
    # understood by mplfinance (0, 1, ..., N-1 for N visible five-second bars).
    n_points = len(bars_5s_short.index)
    ax1.set_xlim(-0.5, n_points - 0.5)

    ax1.set_title("EUR_USD: five-second vs 45-second candles")  # figure title

    # Ensure that y-axes use plain numeric formatting on the left.
    for ax in (ax1, ax2):
        ax.yaxis.get_major_formatter().set_useOffset(False)  # no offsets
        ax.ticklabel_format(style="plain", axis="y")  # avoid scientific notation

    # Hide x tick labels on the top panel so that time labels appear only
    # on the lower axis, making the shared time axis easier to read.
    plt.setp(ax1.get_xticklabels(), visible=False)

    fig.tight_layout()  # reduce unused whitespace
    fig.savefig(fig_path, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # free Matplotlib resources

    # Print a concise console summary for interactive inspection.
    print("Oanda EUR_USD intraday candles:")  # header
    print(f"  5s bars    : {bars_5s.shape[0]} total")  # number of S5 bars
    print(f"  45s bars   : {bars_45s.shape[0]} total")  # number of 45s bars
    print(f"  figure path: {fig_path}")  # location of saved PDF


if __name__ == "__main__":
    main()  # execute workflow when run as a script
