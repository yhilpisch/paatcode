from pathlib import Path  # filesystem paths for project root resolution
from datetime import datetime, timezone  # timestamps for logging
import sys  # access to Python's module search path
from typing import Any  # type hints for dictionaries

"""
Python & AI for Algorithmic Trading
Chapter 15 -- IG Markets and the Anatomy of an API Wrapper

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

IG Markets EUR/USD order placement demo for Chapter 15.

This script illustrates how to use the :class:`IGClient` wrapper from
``wrappers/tpqig.py`` to:

1. retrieve a compact account summary from the IG demo API,
2. place a small EUR/USD market order with optional stop-loss and
   take-profit distances, and
3. close the resulting position again so that the demo account returns
   to its original state.

The goal is to demonstrate a complete ``request -- trade -- inspect`` loop
in the safe demo environment while keeping the code compact and readable.
All trade sizes are intentionally small so that the example remains a
purely educational illustration.
"""

# Ensure that the project root (which contains the ``wrappers`` package) is
# on the Python module search path even when executing this file via
# ``python code/ch15_ig_orders_demo.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # add once, preserve existing order
    sys.path.insert(0, str(PROJECT_ROOT))

from wrappers.tpqig import IGClient  # minimal IG Markets API wrapper


def _pretty_account_line(summary: dict[str, Any]) -> str:
    """Create one human-readable account line for console output."""
    bal = summary.get("balance", "n/a")
    cur = summary.get("currency", "n/a")
    pl = summary.get("profitLoss", "n/a")
    return f"balance={bal} {cur}, P/L={pl}"


def main() -> None:
    """Place a tiny EUR/USD trade and close it again."""
    instrument = "EUR/USD"  # FX pair used throughout this example
    size = 0.5  # contract size for the demo position

    client = IGClient.from_creds()  # configured IG demo client

    try:
        # 1) Account snapshot before any trading activity.
        summary_before = client.get_account_summary()
        print("IG Markets EUR/USD order demo (demo account)")  # header
        print(f"  account id   : {summary_before.get('accountId')}")  # id
        print(f"  snapshot @   : {datetime.now(timezone.utc).isoformat()}")  # time
        print(f"  before trade : {_pretty_account_line(summary_before)}")

        # 2) Retrieve current prices and choose simple distance levels.
        time_str, bid, ask = client.get_prices(instrument)  # current quotes
        mid = (bid + ask) / 2.0  # simple mid-price proxy

        sl_distance = 0.0020  # stop-loss distance in price units
        tp_distance = 0.0040  # take-profit distance in price units

        print("\nPlacing long market order with simple stops:")  # step header
        print(f"  quote time   : {time_str}")  # time of the quote
        print(f"  bid/ask      : {bid:.5f} / {ask:.5f}")  # current spread
        print(f"  mid price    : {mid:.5f}")  # mid-price used for level choice
        print(f"  size         : +{size} (long)  # demonstration size")
        print(f"  SL distance  : {sl_distance:.5f}")  # stop-loss distance
        print(f"  TP distance  : {tp_distance:.5f}")  # take-profit distance

        order_info = client.market_order(
            instrument=instrument,
            direction="BUY",
            size=size,
            stop_distance=sl_distance,
            limit_distance=tp_distance,
        )  # execute market order with basic stops

        print("\nOrder confirmation:")  # short summary of the executed order
        print(f"  deal id      : {order_info.get('dealId')}")
        print(f"  status       : {order_info.get('status')}")
        print(f"  reason       : {order_info.get('reason')}")

        deal_id = order_info.get("dealId")
        if not deal_id:
            print("\nNo deal id returned; unable to close position explicitly.")
            return

        # 3) Close the position again to keep the demo account tidy.
        print("\nClosing position for cleanup...")  # header
        close_info = client.close_position(
            deal_id=deal_id,
            instrument=instrument,
            direction="SELL",
            size=size,
        )
        print(f"  close status : {close_info.get('status')}")
        print(f"  close reason : {close_info.get('reason')}")

        # 4) Account snapshot after the round-trip trade.
        summary_after = client.get_account_summary()
        print("\nAccount snapshot after closing trades:")  # overview
        print(f"  after trade  : {_pretty_account_line(summary_after)}")

    finally:
        client.close_session()  # ensure the IG session is closed


if __name__ == "__main__":
    main()  # execute order demo when run as a script
