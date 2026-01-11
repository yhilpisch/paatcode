from __future__ import annotations

from dataclasses import dataclass  # configuration container
from typing import Dict, Tuple  # type hints for price statistics

from datetime import datetime  # timestamp parsing

import zmq  # ZeroMQ messaging library
from q import q  # lightweight logging helper

"""
Python & AI for Algorithmic Trading
Chapter 12 -- ZeroMQ and Real-Time Market Data Sandboxes

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Streaming tick monitor with q-based logging.

This client subscribes to the local tick server and keeps simple
per-symbol statistics in memory: last price, last timestamp, and tick
count. Every few dozen ticks it writes a compact heartbeat entry via
the :mod:`q` logging helper and prints a short summary to the console.

The example illustrates how a minimal logging layer can sit alongside
other streaming clients and provide quick insight into feed health and
recent prices without introducing heavy dependencies.
"""


@dataclass
class MonitorConfig:
    """Configuration for the tick monitor."""

    address: str="tcp://127.0.0.1:5555"  # ZeroMQ PUB address
    sample_interval: int=50  # log heartbeat after this many ticks


def create_sub_socket(address: str) -> zmq.Socket:
    """Create and configure a subscriber socket that listens to all topics."""
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)
    socket.connect(address)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all symbols
    return socket


def parse_tick(msg: str) -> Tuple[str, datetime, float]:
    """Parse ``SYMBOL ISO_TIMESTAMP PRICE`` into components."""
    symbol_str, time_str, price_str = msg.split()
    ts = datetime.fromisoformat(time_str)
    price = float(price_str)
    return symbol_str, ts, price


def main() -> None:
    """Run the tick monitor until interrupted."""
    cfg = MonitorConfig()
    socket = create_sub_socket(cfg.address)

    print("Starting q-based tick monitor...")  # console header
    print(f"  address : {cfg.address}")
    print(f"  interval: {cfg.sample_interval} ticks per heartbeat")
    print("Press Ctrl+C to stop.\n")

    last_info: Dict[str, Tuple[datetime, float]]={}
    total_ticks = 0

    try:
        while True:
            msg = socket.recv_string()
            symbol, ts, price = parse_tick(msg)
            last_info[symbol] = (ts, price)
            total_ticks += 1

            if total_ticks % cfg.sample_interval != 0:
                continue

            # Prepare a small heartbeat payload for q-based logging.
            snapshot = {
                "kind": "tick_heartbeat",
                "total_ticks": total_ticks,
                "last": {
                    sym: {
                        "time": t.isoformat(),
                        "price": f"{p:.5f}",
                    }
                    for sym, (t, p) in sorted(last_info.items())
                },
            }
            q(snapshot)  # write heartbeat entry via q

            print("Heartbeat:", total_ticks, "ticks")  # console feedback
            for sym, (t, p) in sorted(last_info.items()):
                print(f"  {sym}: {p:.5f} @ {t.isoformat()}")
            print()
    except KeyboardInterrupt:
        print("\nTick monitor interrupted; shutting down...")


if __name__ == "__main__":
    main()  # execute when run as a script
