from __future__ import annotations

from collections import deque  # rolling buffers for recent ticks
from dataclasses import dataclass  # configuration container
from typing import Deque, Dict, List, Tuple  # type hints

from datetime import datetime  # local timestamps for received ticks

import numpy as np  # numerical helpers for returns and statistics
import pandas as pd  # tabular data structures for tick storage
import zmq  # ZeroMQ messaging library

"""
Python & AI for Algorithmic Trading
Chapter 12 -- ZeroMQ and Real-Time Market Data Sandboxes

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

ZeroMQ tick recorder and momentum demo client.

This script connects to the local tick server from ``tick_server.py`` via a
ZeroMQ ``SUB`` socket, subscribes to all instruments, and maintains a rolling
window of ticks in memory as a pair of :class:`pandas.DataFrame` objects:
one for raw ticks and one for resampled bar data.

Every few seconds the script:

* resamples ticks for each symbol to one-second bars,
* computes simple returns and a rolling mean as a momentum signal, and
* prints a compact summary table to the console.

The goal is to provide a small but complete example of how to collect and
analyse streaming ticks using familiar ``pandas`` tools.
"""


@dataclass
class RecorderConfig:
    """Configuration for the recorder client."""

    address: str="tcp://127.0.0.1:5555"  # ZeroMQ PUB address
    window: int=1_200  # maximum number of ticks to keep in memory
    resample_rule: str="1S"  # pandas resampling rule for bars
    momentum_window: int=20  # number of bars for momentum signal


def create_sub_socket(address: str) -> zmq.Socket:
    """Create and configure a subscriber socket."""
    ctx = zmq.Context.instance()  # shared context for process
    socket = ctx.socket(zmq.SUB)  # subscriber socket
    socket.connect(address)  # connect to tick server
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all topics
    return socket


def parse_tick(msg: str) -> Tuple[str, datetime, float]:
    """Parse a raw tick message into symbol, timestamp, and price."""
    symbol_str, time_str, price_str = msg.split()
    ts = datetime.fromisoformat(time_str)  # parse ISO timestamp string
    price = float(price_str)  # convert price token to float
    return symbol_str, ts, price


def update_buffers(
    tick_buffers: Dict[str, Deque[Tuple[datetime, float]]],
    symbol: str,
    ts: datetime,
    price: float,
    cfg: RecorderConfig,
) -> None:
    """Append a new tick to the appropriate rolling buffer."""
    if symbol not in tick_buffers:
        tick_buffers[symbol] = deque(maxlen=cfg.window)  # one buffer per key
    tick_buffers[symbol].append((ts, price))  # store timestamp and price


def build_tick_frame(
    tick_buffers: Dict[str, Deque[Tuple[datetime, float]]],
) -> pd.DataFrame:
    """Convert rolling buffers into a single tick-level DataFrame."""
    rows: List[Tuple[str, datetime, float]] = []
    for symbol, buf in tick_buffers.items():
        rows.extend((symbol, ts, price) for ts, price in buf)
    if not rows:
        return pd.DataFrame(columns=["symbol", "time", "price"])
    frame = pd.DataFrame(rows, columns=["symbol", "time", "price"])
    frame = frame.set_index("time").sort_index()  # time-based index
    return frame


def compute_momentum(
    tick_frame: pd.DataFrame,
    cfg: RecorderConfig,
) -> pd.DataFrame:
    """Resample ticks to bars and compute momentum signals."""
    if tick_frame.empty:
        return pd.DataFrame()

    records: List[pd.DataFrame] = []
    for symbol, group in tick_frame.groupby("symbol"):
        # Resample to bars with simple OHLC aggregation; here only close is
        # used for momentum but open/high/low could be added if desired.
        prices = group["price"].resample(cfg.resample_rule).last().dropna()
        if prices.empty:
            continue
        rets = prices.pct_change().dropna()  # simple returns from prices
        mom = rets.rolling(cfg.momentum_window).mean()  # rolling mean signal
        df = pd.DataFrame(
            {
                "symbol": symbol,
                "price": prices.reindex(mom.index),
                "return": rets.reindex(mom.index),
                "momentum": mom,
            }
        )
        records.append(df.dropna())

    if not records:
        return pd.DataFrame()
    bars = pd.concat(records).sort_index()
    return bars


def print_momentum_snapshot(bars: pd.DataFrame) -> None:
    """Print a compact snapshot of the latest momentum values."""
    if bars.empty:
        return
    latest = bars.groupby("symbol").tail(1)  # last bar per symbol
    latest = latest.set_index("symbol")
    # Create a slightly rounded view for the console.
    view = latest[["price", "momentum"]].copy()
    view["price"] = view["price"].round(4)
    view["momentum"] = view["momentum"].round(4)
    print("\nMomentum snapshot (resampled bars):")  # header
    print(view.to_string())  # formatted table


def main() -> None:
    """Run the recorder client until interrupted."""
    cfg = RecorderConfig()  # default configuration
    socket = create_sub_socket(cfg.address)  # SUB socket

    print("Starting recorder client...")  # console header
    print(f"  address : {cfg.address}")
    print(f"  window  : {cfg.window} ticks per symbol")
    print("Press Ctrl+C to stop.\n")

    tick_buffers: Dict[str, Deque[Tuple[datetime, float]]] = {}
    step_counter = 0  # counts how many ticks have been processed

    try:
        while True:
            msg = socket.recv_string()  # blocking receive
            symbol, ts, price = parse_tick(msg)  # parse raw message
            update_buffers(tick_buffers, symbol, ts, price, cfg)
            step_counter += 1

            # Every few dozen ticks, rebuild the DataFrame and print a
            # short momentum snapshot to keep the console output readable.
            if step_counter % 50 == 0:
                tick_frame = build_tick_frame(tick_buffers)
                bars = compute_momentum(tick_frame, cfg)
                print_momentum_snapshot(bars)
    except KeyboardInterrupt:
        print("\nRecorder interrupted; shutting down...")  # footer


if __name__ == "__main__":
    main()  # execute when run as a script
