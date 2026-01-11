from __future__ import annotations

from dataclasses import dataclass  # configuration container
from typing import Dict  # type hints for portfolio positions

from datetime import datetime  # timestamp parsing and reporting
from pathlib import Path  # filesystem paths for database file
import sqlite3  # database connection

import pandas as pd  # tabular processing and resampling

"""
Python & AI for Algorithmic Trading
Chapter 12 -- ZeroMQ and Real-Time Market Data Sandboxes

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Simple mark-to-market and reporting script for SQLite tick data.

This script reads the tick data written by ``client_sqlite_writer.py``
from a local SQLite database and produces a few basic reports:

* last price per symbol and the time of the most recent tick,
* a time-series of one-second mid-prices per symbol, and
* a mark-to-market valuation for a small example portfolio.

The goal is to illustrate how a separate process can consume the same
tick database that is being populated by a streaming writer and use it
for reporting tasks such as end-of-day summaries or periodic risk checks.
"""


@dataclass
class ReportConfig:
    """Configuration for the portfolio reporting script."""

    db_filename: str="stream_ticks.sqlite3"  # SQLite database file
    lookback_seconds: int=600  # history window for intraday report


def load_ticks(db_path: Path, lookback_seconds: int) -> pd.DataFrame:
    """Load recent ticks from the SQLite database into a DataFrame."""
    conn = sqlite3.connect(db_path)
    try:
        # Select all ticks in the lookback window relative to now.
        now = datetime.utcnow()
        cutoff = now.timestamp() - lookback_seconds
        query = """
            SELECT symbol, ts, price
            FROM ticks
            WHERE strftime('%s', ts) >= ?
            ORDER BY ts
        """
        df = pd.read_sql_query(
            query,
            conn,
            params=(cutoff,),
            parse_dates=["ts"],
        )
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame(columns=["symbol", "ts", "price"])

    df = df.rename(columns={"ts": "time"})
    df = df.set_index("time").sort_index()
    return df


def compute_last_prices(ticks: pd.DataFrame) -> pd.DataFrame:
    """Return last price and timestamp per symbol."""
    if ticks.empty:
        return pd.DataFrame(columns=["symbol", "last_time", "last_price"])

    grouped = ticks.groupby("symbol")
    last_rows = grouped.tail(1).reset_index()
    last_rows = last_rows.rename(
        columns={"time": "last_time", "price": "last_price"},
    )
    return last_rows[["symbol", "last_time", "last_price"]]


def compute_resampled_series(ticks: pd.DataFrame) -> pd.DataFrame:
    """Resample ticks to one-second bars and return closing prices."""
    if ticks.empty:
        return pd.DataFrame()

    out_frames: list[pd.Series]=[]
    for symbol, group in ticks.groupby("symbol"):
        closes = group["price"].resample("1S").last().dropna()
        closes.name = symbol
        out_frames.append(closes)

    if not out_frames:
        return pd.DataFrame()

    prices = pd.concat(out_frames, axis=1)
    return prices


def mark_to_market(
    last_prices: pd.DataFrame,
    positions: Dict[str, float],
    base_ccy: str="USD",
) -> pd.DataFrame:
    """Compute mark-to-market values for a simple portfolio."""
    if last_prices.empty:
        return pd.DataFrame(columns=["symbol", "units", "price", "value"])

    rows = []
    for _, row in last_prices.iterrows():
        symbol = str(row["symbol"])
        price = float(row["last_price"])
        units = float(positions.get(symbol, 0.0))
        value = units * price
        rows.append(
            {"symbol": symbol, "units": units, "price": price, "value": value},
        )
    portfolio = pd.DataFrame(rows)
    portfolio["base_ccy"] = base_ccy
    return portfolio


def main() -> None:
    """Load recent ticks and generate a simple portfolio report."""
    cfg = ReportConfig()
    script_dir = Path(__file__).resolve().parent
    db_path = script_dir / cfg.db_filename

    print("Streaming tick portfolio report")  # header
    print(f"  database : {db_path}")
    print(f"  lookback : {cfg.lookback_seconds} seconds\n")

    ticks = load_ticks(db_path, cfg.lookback_seconds)
    if ticks.empty:
        print("No recent ticks found in the database.")
        return

    last_prices = compute_last_prices(ticks)
    prices_1s = compute_resampled_series(ticks)

    print("Last prices per symbol:")  # section header
    print(last_prices.to_string(index=False))
    print()

    # Example portfolio: long 10 units of SPY, 5 units of GLD, and
    # 20,000 units of EURUSD (positive number corresponds to long EUR).
    positions = {"SPY": 10.0, "GLD": 5.0, "EURUSD": 20_000.0}

    portfolio = mark_to_market(last_prices, positions)
    print("Mark-to-market portfolio valuation:")  # section header
    print(portfolio.to_string(index=False, float_format=lambda x: f"{x: .2f}"))

    if not prices_1s.empty:
        print("\nSample of resampled one-second prices:")
        print(prices_1s.tail(10).to_string())


if __name__ == "__main__":
    main()  # execute when run as a script
