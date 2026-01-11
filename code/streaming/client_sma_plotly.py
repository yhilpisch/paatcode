from __future__ import annotations

from collections import deque  # rolling window for recent ticks
from dataclasses import dataclass  # configuration container
from typing import Deque, List, Tuple  # type hints

from datetime import datetime  # local timestamps for received ticks
from pathlib import Path  # filesystem paths for HTML output
import argparse  # command-line argument parsing

import pandas as pd  # tabular data handling
import plotly.graph_objects as go  # Plotly figure construction
import plotly.io as pio  # HTML export for Plotly figures
import zmq  # ZeroMQ messaging library

"""
Python & AI for Algorithmic Trading
Chapter 12 -- ZeroMQ and Real-Time Market Data Sandboxes

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Streaming SMA visualisation client using Plotly.

This script connects to the local tick server from ``tick_server.py`` via a
ZeroMQ ``SUB`` socket, subscribes to a single symbol, and maintains a rolling
window of recent ticks in memory. Every few seconds it:

* converts the tick buffer into a :class:`pandas.Series`,
* computes a simple moving average (SMA) over a configurable window, and
* writes a Plotly-based HTML file that visualises price, SMA, and basic
  buy/sell markers whenever price crosses the SMA.

The resulting HTML file can be opened in a browser and refreshed
periodically to see an updated snapshot of the synthetic price path.
"""


@dataclass
class SmaClientConfig:
    """Configuration for the SMA visualisation client."""

    address: str="tcp://127.0.0.1:5555"  # ZeroMQ PUB address
    symbol: str="SPY"  # instrument symbol to subscribe to
    window: int=600  # maximum number of ticks stored in buffer
    sma_length: int=40  # number of ticks for SMA
    update_every: int=20  # update figure after this many ticks


def create_sub_socket(address: str, symbol: str) -> zmq.Socket:
    """Create and configure a subscriber socket for a single symbol."""
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)
    socket.connect(address)
    socket.setsockopt_string(zmq.SUBSCRIBE, symbol)  # filter by symbol
    return socket


def parse_tick(msg: str) -> Tuple[str, datetime, float]:
    """Parse a raw tick message of the form 'SYMBOL ISO_TIMESTAMP PRICE'."""
    symbol_str, time_str, price_str = msg.split()
    ts = datetime.fromisoformat(time_str)
    price = float(price_str)
    return symbol_str, ts, price


def make_figure(
    series: pd.Series,
    sma: pd.Series,
    html_path: str,
    symbol: str,
) -> None:
    """Create and write a Plotly figure with price, SMA, and signals."""
    if series.empty or sma.empty:
        return

    # Align SMA with price index and compute simple crossover signals.
    aligned_sma = sma.reindex(series.index)
    signal = (series > aligned_sma).astype(int)  # 1 when price above SMA
    signal_shifted = signal.shift(1).fillna(method="bfill")
    crossings = signal != signal_shifted  # True at crossover points

    buy_mask = crossings & (signal == 1)
    sell_mask = crossings & (signal == 0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=f"{symbol} price",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=aligned_sma.index,
            y=aligned_sma.values,
            mode="lines",
            name=f"{sma.name}",
            line=dict(color="#ff7f0e", width=1.3, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series.index[buy_mask],
            y=series[buy_mask],
            mode="markers",
            name="buy signal",
            marker=dict(color="#2ca02c", size=8, symbol="triangle-up"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series.index[sell_mask],
            y=series[sell_mask],
            mode="markers",
            name="sell signal",
            marker=dict(color="#d62728", size=8, symbol="triangle-down"),
        )
    )

    fig.update_layout(
        title=f"{symbol}: price and {sma.name} (streaming snapshot)",
        xaxis_title="time (most recent ticks)",
        yaxis_title="price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
        margin=dict(l=50, r=20, t=60, b=40),
    )

    pio.write_html(fig, html_path, auto_open=False)  # write HTML file


def main() -> None:
    """Run the SMA visualisation client until interrupted."""
    parser = argparse.ArgumentParser(
        description="Streaming SMA visualisation client.",
    )
    parser.add_argument(
        "--symbol",
        default="SPY",
        help="instrument symbol to subscribe to (default: SPY)",
    )
    args = parser.parse_args()

    cfg = SmaClientConfig(symbol=args.symbol)
    socket = create_sub_socket(cfg.address, cfg.symbol)

    buffer: Deque[Tuple[datetime, float]] = deque(maxlen=cfg.window)
    script_dir = Path(__file__).resolve().parent  # directory of this script
    html_path = script_dir / f"fig_sma_{cfg.symbol}.html"
    tick_counter = 0

    print("Starting SMA Plotly client...")  # console header
    print(f"  address : {cfg.address}")
    print(f"  symbol  : {cfg.symbol}")
    print(f"  SMA len : {cfg.sma_length} ticks")
    print(f"  HTML    : {html_path}")
    print("Open the HTML file in a browser and reload to see updates.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            msg = socket.recv_string()  # blocking receive
            symbol, ts, price = parse_tick(msg)
            buffer.append((ts, price))
            tick_counter += 1

            if tick_counter % cfg.update_every != 0:
                continue

            if not buffer:
                continue

            times: List[datetime] = []
            prices: List[float] = []
            for ts_buf, p_buf in buffer:
                times.append(ts_buf)
                prices.append(p_buf)

            series = pd.Series(prices, index=pd.to_datetime(times))
            series = series.sort_index()
            sma = series.rolling(cfg.sma_length).mean()
            sma.name = f"SMA({cfg.sma_length})"

            make_figure(series, sma, html_path, cfg.symbol)
            latest_price = float(series.iloc[-1])
            latest_sma = float(sma.iloc[-1])
            print(
                f"[{series.index[-1].isoformat()}] "
                f"{cfg.symbol} price={latest_price:.4f}, "
                f"SMA={latest_sma:.4f}"
            )
    except KeyboardInterrupt:
        print("\nSMA client interrupted; shutting down...")


if __name__ == "__main__":
    main()  # execute when run as a script
