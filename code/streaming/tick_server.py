from __future__ import annotations

import time  # wall-clock timing between tick publications
from dataclasses import dataclass  # configuration container for instruments
from typing import Dict  # type hints for price dictionaries

from datetime import datetime, timezone  # UTC timestamps for ticks

import numpy as np  # numerical helper for random walks
import zmq  # ZeroMQ messaging library

"""
Python & AI for Algorithmic Trading
Chapter 12 -- ZeroMQ and Real-Time Market Data Sandboxes

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Local ZeroMQ tick server for three synthetic instruments.

This script implements a small publisher that broadcasts tick updates for
three instruments over a ZeroMQ ``PUB`` socket:

* ``SPY``  — stylised equity index proxy,
* ``GLD``  — stylised gold ETF proxy, and
* ``EURUSD`` — stylised FX spot rate.

Prices evolve as simple geometric random walks with configurable volatilities.
Subscribers such as ``client_recorder.py`` and ``client_sma_plotly.py``
can connect to the same address and filter by symbol.
"""


@dataclass
class InstrumentConfig:
    """Configuration for a single synthetic instrument."""

    symbol: str  # instrument symbol, for example "SPY"
    start_price: float  # initial price level
    vol: float  # volatility scale for random walk increments


def create_context_and_socket(bind_address: str) -> zmq.Socket:
    """Create a shared ZeroMQ context and publisher socket."""
    ctx = zmq.Context.instance()  # shared context for the process
    socket = ctx.socket(zmq.PUB)  # publisher socket
    socket.bind(bind_address)  # bind to TCP address
    return socket


def initialise_prices(configs: Dict[str, InstrumentConfig]) -> Dict[str, float]:
    """Initialise price dictionary from instrument configuration."""
    prices: Dict[str, float] = {}
    for symbol, cfg in configs.items():
        prices[symbol] = cfg.start_price  # starting level per instrument
    return prices


def main() -> None:
    """Publish synthetic ticks for SPY, GLD, and EURUSD."""
    bind_address = "tcp://127.0.0.1:5555"  # local PUB endpoint

    instruments = {
        "SPY": InstrumentConfig("SPY", start_price=500.0, vol=0.0005),
        "GLD": InstrumentConfig("GLD", start_price=200.0, vol=0.0007),
        "EURUSD": InstrumentConfig("EURUSD", start_price=1.10, vol=0.0003),
    }  # configuration for three synthetic streams

    socket = create_context_and_socket(bind_address)  # PUB socket
    prices = initialise_prices(instruments)  # price state per instrument
    rng = np.random.default_rng(seed=42)  # deterministic random numbers

    print("Starting local ZeroMQ tick server...")  # console header
    print(f"  address : {bind_address}")  # PUB endpoint information
    print(f"  symbols : {', '.join(sorted(instruments))}")  # instrument list
    print("Press Ctrl+C to stop.\n")  # shutdown instruction

    try:
        while True:
            now = datetime.now(timezone.utc).isoformat()  # UTC timestamp
            for symbol, cfg in instruments.items():
                shock = rng.normal(loc=0.0, scale=cfg.vol)  # random increment
                prices[symbol] *= 1.0 + shock  # geometric random walk update
                price = prices[symbol]
                msg = f"{symbol} {now} {price:.5f}"  # topic + payload string
                socket.send_string(msg)  # publish tick to all subscribers
            time.sleep(0.25)  # pause before next batch of ticks
    except KeyboardInterrupt:
        print("\nTick server interrupted; shutting down...")  # footer
    finally:
        socket.close(0)  # close socket without linger


if __name__ == "__main__":
    main()  # execute when run as a script
