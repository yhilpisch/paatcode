from __future__ import annotations

from dataclasses import dataclass  # configuration container
from typing import Optional  # type hints for optional values

from datetime import datetime  # timestamps for tick arrivals
from pathlib import Path  # filesystem paths for project root
import argparse  # command-line argument parsing
import sys  # access to Python's module search path

import pandas as pd  # timestamp parsing and time-series handling
import zmq  # ZeroMQ messaging library

"""
Python & AI for Algorithmic Trading
Chapter 12 -- ZeroMQ and Real-Time Market Data Sandboxes

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Streaming event-based SMA client driven by ZeroMQ ticks.

This script connects to the local ZeroMQ tick server from
``tick_server.py`` via a ``SUB`` socket and converts incoming ticks
into the event types used by the Chapter 11 event-based backtester:
``MarketEvent``, ``SignalEvent``, ``OrderEvent``, and ``FillEvent``.

The client reuses the strategy, portfolio, and execution components
from :mod:`code.ch11_event_backtester`. Each received tick is wrapped
as a ``MarketEvent`` and pushed into an ``EventQueue``. The event loop
then performs the usual dispatch:

* market events update the execution handler and the SMA strategy,
* signal events are turned into orders by the portfolio,
* orders become fills at the latest known price, and
* fills update the portfolio's internal position series.

For demonstration, the script logs signals and fills to the console so
that you can see how an event-based strategy reacts to streaming
prices in real time.
"""

# Ensure that the project root (which contains the ``code`` package) is
# on the Python module search path even when executing this file via
# ``python code/streaming/client_event_backtester.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # add once, preserve existing order
    sys.path.insert(0, str(PROJECT_ROOT))

from code.ch11_event_backtester import (  # type: ignore  # local module
    EventQueue,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    SmaCrossStrategy,
    NaivePortfolio,
    SimulatedExecutionHandler,
)


@dataclass
class StreamBacktestConfig:
    """Configuration for the streaming SMA backtest client."""

    address: str="tcp://127.0.0.1:5555"  # ZeroMQ PUB address
    symbol: str="SPY"  # symbol to subscribe to from the tick server
    fast_window: int=20  # length of fast SMA in ticks
    slow_window: int=60  # length of slow SMA in ticks


def create_sub_socket(address: str, symbol: str) -> zmq.Socket:
    """Create and configure a subscriber socket for one symbol."""
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)
    socket.connect(address)
    socket.setsockopt_string(zmq.SUBSCRIBE, symbol)  # filter by topic
    return socket


def parse_tick(msg: str) -> tuple[str, datetime, float]:
    """Parse a raw tick message ``SYMBOL ISO_TIMESTAMP PRICE``."""
    symbol_str, time_str, price_str = msg.split()
    ts = datetime.fromisoformat(time_str)
    price = float(price_str)
    return symbol_str, ts, price


def dispatch_events(
    events: EventQueue,
    strategy: SmaCrossStrategy,
    portfolio: NaivePortfolio,
    execution: SimulatedExecutionHandler,
    symbol: str,
) -> Optional[float]:
    """Process all pending events and return the latest position."""
    latest_position: Optional[float]=None

    while not events.empty():
        event = events.get()
        if isinstance(event, MarketEvent):
            execution.on_market_event(event)
            strategy.on_market_event(event, events)
        elif isinstance(event, SignalEvent):
            portfolio.on_signal(event, events)
            print(
                f"[{event.time}] {symbol} SIGNAL "
                f"{event.signal:+.0f}"
            )
        elif isinstance(event, OrderEvent):
            execution.on_order(event, events)
        elif isinstance(event, FillEvent):
            portfolio.on_fill(event)
            latest_position = event.position
            print(
                f"[{event.time}] {symbol} FILL   "
                f"position={event.position:+.1f} "
                f"price={event.price:.4f}"
            )
    return latest_position


def main() -> None:
    """Run the streaming SMA backtest client until interrupted."""
    parser = argparse.ArgumentParser(
        description="Streaming SMA backtest client for ZeroMQ ticks.",
    )
    parser.add_argument(
        "--symbol",
        default="SPY",
        help="instrument symbol to subscribe to (default: SPY)",
    )
    args = parser.parse_args()

    cfg = StreamBacktestConfig(symbol=args.symbol)
    socket = create_sub_socket(cfg.address, cfg.symbol)

    events = EventQueue()
    strategy = SmaCrossStrategy(
        fast_window=cfg.fast_window,
        slow_window=cfg.slow_window,
    )
    portfolio = NaivePortfolio()
    execution = SimulatedExecutionHandler()

    print("Starting streaming SMA client...")  # console header
    print(f"  address : {cfg.address}")
    print(f"  symbol  : {cfg.symbol}")
    print(
        f"  windows : fast={cfg.fast_window}, "
        f"slow={cfg.slow_window}"
    )
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            msg = socket.recv_string()  # blocking receive from server
            symbol, ts, price = parse_tick(msg)
            if symbol != cfg.symbol:
                continue  # filter defensively, even with SUBSCRIBE

            # Wrap incoming tick as a MarketEvent and push into the
            # event queue. The event loop below then takes over and
            # propagates the update through strategy, portfolio, and
            # execution components.
            market_event = MarketEvent(
                time=pd.Timestamp(ts),
                price=price,
            )
            events.put(market_event)

            dispatch_events(
                events=events,
                strategy=strategy,
                portfolio=portfolio,
                execution=execution,
                symbol=cfg.symbol,
            )
    except KeyboardInterrupt:
        print("\nStreaming SMA client interrupted; shutting down...")


if __name__ == "__main__":
    main()  # execute when run as a script
