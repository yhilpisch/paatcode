from pathlib import Path  # filesystem paths for project root resolution
from datetime import datetime, timezone  # timestamps for logging
import sys  # access to Python's module search path
from typing import Any  # type hints for dictionaries

"""
Python & AI for Algorithmic Trading
Chapter 14 -- Working with Oanda Demo Accounts

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Oanda EUR_USD order placement and transaction demo for Chapter 14.

This script illustrates how to use the :class:`OandaClient` wrapper from
``wrappers/tpqoa.py`` to:

1. retrieve a compact account summary from the Oanda practice API,
2. place a small EUR_USD market order with attached stop-loss, take-profit,
   and trailing stop-loss instructions, and
3. close the resulting open trade again and inspect the recent transactions
   associated with this mini round-trip trade.

The goal is to demonstrate a complete ``request -- trade -- inspect`` loop
in the safe practice environment while keeping the code compact and
readable. All trade sizes are intentionally small so that the example
remains a purely educational illustration.
"""

# Ensure that the project root (which contains the ``wrappers`` package) is
# on the Python module search path even when executing this file via
# ``python code/ch14_oanda_orders_demo.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # add once, preserve existing order
    sys.path.insert(0, str(PROJECT_ROOT))

from wrappers.tpqoa import OandaClient  # minimal Oanda API wrapper


def _pretty_account_line(summary: dict[str, Any]) -> str:
    """Create one human-readable account line for console output."""
    bal = summary.get("balance", "n/a")
    cur = summary.get("currency", "n/a")
    open_trades = summary.get("openTradeCount", "n/a")
    open_pos = summary.get("openPositionCount", "n/a")
    return (
        f"balance={bal} {cur}, "
        f"open trades={open_trades}, open positions={open_pos}"
    )


def main() -> None:
    """Place a tiny EUR_USD trade with SL/TP/TSL and inspect transactions."""
    instrument = "EUR_USD"  # FX pair used throughout this example
    units = 1_000  # small position size for the practice account

    client = OandaClient.from_creds()  # configured Oanda v20 client

    # 1) Account snapshot before any trading activity.
    summary_before = client.get_account_summary(detailed=False)
    last_tx_before = summary_before.get("lastTransactionID")
    print("Oanda EUR_USD order demo (practice account)")  # header
    print(f"  account id   : {summary_before.get('id')}")  # account identifier
    print(f"  snapshot @   : {datetime.now(timezone.utc).isoformat()}")  # time
    print(f"  before trade : {_pretty_account_line(summary_before)}")

    # 2) Retrieve current prices and construct SL/TP/TSL levels for a long.
    time_str, bid, ask = client.get_prices(instrument)  # current quotes
    mid = (bid + ask) / 2.0  # simple mid-price proxy

    sl_distance = 0.0020  # stop-loss distance in price units
    tp_distance = 0.0040  # take-profit distance in price units
    tsl_distance = 0.0010  # trailing stop-loss distance in price units

    stop_loss = mid - sl_distance  # protective stop below entry
    take_profit = mid + tp_distance  # profit target above entry

    print("\nPlacing long market order with attached orders:")  # step header
    print(f"  quote time   : {time_str}")  # time of the quote
    print(f"  bid/ask      : {bid:.5f} / {ask:.5f}")  # current spread
    print(f"  mid price    : {mid:.5f}")  # mid-price used for level choice
    print(f"  units        : +{units} (long)  # demonstration size")
    print(f"  stop loss    : {stop_loss:.5f}")  # protective stop level
    print(f"  take profit  : {take_profit:.5f}")  # profit target level
    print(f"  trailing SL  : {tsl_distance:.5f} distance")  # trailing distance

    order_info = client.place_market_order(
        instrument=instrument,
        units=units,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop_distance=tsl_distance,
    )  # execute market order with attached orders

    print("\nOrder fill summary:")  # short summary of the executed order
    print(f"  order id     : {order_info.get('order_id')}")
    print(f"  trade id     : {order_info.get('trade_id')}")
    print(f"  fill time    : {order_info.get('time')}")
    fill_price = order_info.get("price")
    if fill_price is not None:
        print(f"  fill price   : {float(fill_price):.5f}")

    # 3) Close the open trade again to keep the practice account tidy.
    closed_trades = client.close_trades(instrument=instrument)
    print("\nClosing open EUR_USD trades for cleanup...")  # header
    if not closed_trades:
        print("  no open trades to close")  # nothing left to do
    else:
        for ct in closed_trades:
            print(
                "  closed trade : "
                f"id={ct.get('trade_id')}, "
                f"units={ct.get('closed_units')}, "
                f"price={ct.get('price')}",
            )

    # 4) Account snapshot after the round-trip trade.
    summary_after = client.get_account_summary(detailed=False)
    print("\nAccount snapshot after closing trades:")  # overview
    print(f"  after trade  : {_pretty_account_line(summary_after)}")

    # 5) Retrieve and display recent transactions created by the demo.
    if last_tx_before is not None:
        txs = client.get_transactions_since(last_transaction_id=last_tx_before)
        print("\nRecent transactions for this round-trip:")  # header
        for tx in txs:
            tx_type = tx.get("type")
            tx_id = tx.get("id")
            tx_instrument = tx.get("instrument", "")
            tx_units = tx.get("units", "")
            tx_price = tx.get("price", "")
            print(
                f"  id={tx_id}, type={tx_type}, "
                f"instrument={tx_instrument}, units={tx_units}, "
                f"price={tx_price}",
            )
    else:
        print("\nNo lastTransactionID found in initial snapshot;")
        print("unable to retrieve a filtered list of recent transactions.")


if __name__ == "__main__":
    main()  # execute order demo when run as a script
