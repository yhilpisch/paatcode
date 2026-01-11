from __future__ import annotations

from dataclasses import dataclass  # configuration container
from typing import Tuple  # type hints for parsed ticks

from datetime import datetime  # timestamp parsing
from pathlib import Path  # filesystem paths for database file
import sqlite3  # lightweight embedded database

import zmq  # ZeroMQ messaging library

"""
Python & AI for Algorithmic Trading
Chapter 12 -- ZeroMQ and Real-Time Market Data Sandboxes

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

ZeroMQ tick recorder that writes to a local SQLite database.

This client subscribes to all symbols from the local tick server
(``tick_server.py``) and writes each tick into a SQLite database in the
same folder. The database schema is intentionally simple:

* table ``ticks`` with columns
  ``id`` (INTEGER PRIMARY KEY),
  ``symbol`` (TEXT),
  ``ts`` (TEXT, ISO-8601 UTC timestamp),
  ``price`` (REAL).

An index on ``symbol`` and ``ts`` makes later retrieval efficient. The
database file is ignored by version control via an entry in the project's
``.gitignore``.
"""


@dataclass
class WriterConfig:
    """Configuration for the SQLite tick writer."""

    address: str="tcp://127.0.0.1:5555"  # ZeroMQ PUB address
    db_filename: str="stream_ticks.sqlite3"  # SQLite database file
    commit_interval: int=100  # number of inserts between commits


def create_sub_socket(address: str) -> zmq.Socket:
    """Create and configure a subscriber socket that listens to all topics."""
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)
    socket.connect(address)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all symbols
    return socket


def parse_tick(msg: str) -> Tuple[str, str, float]:
    """Parse a raw tick message and return (symbol, iso_timestamp, price)."""
    symbol_str, time_str, price_str = msg.split()
    # Validate that the timestamp parses; store the original ISO string.
    datetime.fromisoformat(time_str)
    price = float(price_str)
    return symbol_str, time_str, price


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the ticks table and index if they do not yet exist."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            ts TEXT NOT NULL,
            price REAL NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts
        ON ticks (symbol, ts)
        """
    )
    conn.commit()


def main() -> None:
    """Run the SQLite writer client until interrupted."""
    cfg = WriterConfig()

    script_dir = Path(__file__).resolve().parent
    db_path = script_dir / cfg.db_filename

    socket = create_sub_socket(cfg.address)
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)

    print("Starting SQLite tick writer...")  # console header
    print(f"  address : {cfg.address}")
    print(f"  db file : {db_path}")
    print("Press Ctrl+C to stop.\n")

    cur = conn.cursor()
    pending = 0

    try:
        while True:
            msg = socket.recv_string()  # blocking receive
            symbol, ts_str, price = parse_tick(msg)
            cur.execute(
                "INSERT INTO ticks (symbol, ts, price) VALUES (?, ?, ?)",
                (symbol, ts_str, price),
            )
            pending += 1

            if pending >= cfg.commit_interval:
                conn.commit()
                print(
                    f"Committed {pending} ticks "
                    f"at {datetime.utcnow().isoformat()}."
                )
                pending = 0
    except KeyboardInterrupt:
        print("\nSQLite writer interrupted; flushing pending data...")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()  # execute when run as a script
