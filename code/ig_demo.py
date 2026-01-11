"""Python & AI for Algorithmic Trading
Chapter 15 -- IG Markets and the Anatomy of an API Wrapper

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Simple demo showing how to use :mod:`wrappers.IGClient` with the IG Markets demo API.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from typing import Iterable

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from wrappers import IGClient


def _parse_iso_date(value: str) -> datetime:
    """Parse calendar dates expressed as ``YYYY-MM-DD``."""
    return datetime.fromisoformat(value)


def _format_account_summary(summary: dict[str, object]) -> Iterable[str]:
    fields = [
        ("Account", summary.get("accountId")),
        ("Alias", summary.get("accountAlias")),
        ("Balance", summary.get("balance")),
        ("Profit/Loss", summary.get("profitLoss")),
    ]
    return [f"{label:15s}: {val}" for label, val in fields if val is not None]


def main() -> None:
    parser = ArgumentParser(description="Quick IG Markets demo using wrappers.IGClient")
    parser.add_argument("--instrument", default="EUR/USD", help="Instrument name or epic to query")
    parser.add_argument("--days", type=int, default=7, help="Number of days to request (default: 7)")
    parser.add_argument("--granularity", default="D", help="IG resolution (for example D, 1H, 1Min)")
    parser.add_argument(
        "--end",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        help="End timestamp (ISO).",
    )
    args = parser.parse_args()

    end = _parse_iso_date(args.end)
    start = end - timedelta(days=args.days)

    client = IGClient.from_creds()
    try:
        summary = client.get_account_summary()
        print("Account summary:")
        for line in _format_account_summary(summary):
            print(" ", line)

        print(f"\nFetching {args.granularity} candles for {args.instrument} "
              f"from {start.date()} to {end.date()} ...")
        candles = client.get_candles(
            args.instrument,
            granularity=args.granularity,
            start=start,
            end=end,
        )
        print("\nLatest candles (tail):")
        print(candles.tail())
    finally:
        client.close_session()


if __name__ == "__main__":
    main()
