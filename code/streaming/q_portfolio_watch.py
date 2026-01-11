from __future__ import annotations

from dataclasses import dataclass  # configuration container
from typing import Dict  # type hints for position dictionaries

from datetime import datetime  # timestamps for log entries
from pathlib import Path  # filesystem paths for database

from q import q  # lightweight logging helper

from code.streaming.sqlite_portfolio_report import (  # type: ignore
    load_ticks,
    compute_last_prices,
    mark_to_market,
)

"""
Python & AI for Algorithmic Trading
Chapter 12 -- ZeroMQ and Real-Time Market Data Sandboxes

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Portfolio watcher with q-based logging.

This script reads recent ticks from the SQLite database maintained by
``client_sqlite_writer.py``, computes last prices per symbol, marks a
small example portfolio to market, and writes a concise summary and any
threshold breaches to the :mod:`q` log.

In a production setup, a variant of this script could be called
periodically (for example via ``cron``) to provide lightweight
monitoring of exposures and account value.
"""


@dataclass
class WatchConfig:
    """Configuration for the portfolio watcher."""

    db_filename: str="stream_ticks.sqlite3"  # SQLite database file
    lookback_seconds: int=600  # history window for tick retrieval
    max_notional: float=200_000.0  # simple exposure threshold

    # Example portfolio in instrument units.
    positions: Dict[str, float]=None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.positions is None:
            self.positions = {
                "SPY": 10.0,
                "GLD": 5.0,
                "EURUSD": 20_000.0,
            }


def main() -> None:
    """Compute a portfolio snapshot and log with q."""
    cfg = WatchConfig()
    script_dir = Path(__file__).resolve().parent
    db_path = script_dir / cfg.db_filename

    ticks = load_ticks(db_path, cfg.lookback_seconds)
    if ticks.empty:
        q({"kind": "portfolio_watch", "status": "no_ticks", "time": datetime.utcnow().isoformat()})
        print("No recent ticks found in the SQLite database.")
        return

    last_prices = compute_last_prices(ticks)
    portfolio = mark_to_market(last_prices, cfg.positions)

    total_value = float(portfolio["value"].sum())
    total_notional = float((portfolio["price"].abs() * portfolio["units"].abs()).sum())

    snapshot = {
        "kind": "portfolio_watch",
        "time": datetime.utcnow().isoformat(),
        "total_value": total_value,
        "total_notional": total_notional,
        "rows": portfolio.to_dict(orient="records"),
    }
    q(snapshot)  # log compact portfolio summary

    print("Portfolio watch snapshot")  # console feedback
    print(portfolio.to_string(index=False, float_format=lambda x: f"{x: .2f}"))
    print(f"\nTotal value   : {total_value: .2f}")
    print(f"Total notional: {total_notional: .2f}")

    if total_notional > cfg.max_notional:
        alert = {
            "kind": "portfolio_alert",
            "time": datetime.utcnow().isoformat(),
            "reason": "notional_limit_exceeded",
            "limit": cfg.max_notional,
            "total_notional": total_notional,
        }
        q(alert)
        print(
            f"\nWARNING: total notional {total_notional: .2f} "
            f"exceeds limit {cfg.max_notional: .2f}",
        )


if __name__ == "__main__":
    main()  # execute when run as a script
