"""Python & AI for Algorithmic Trading
Chapter 14 -- Working with Oanda Demo Accounts

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Lightweight Oanda v20 client for book examples.

This module provides a small, well-documented wrapper around the Oanda v20
application programming interface (API) that is tailored to the needs of the
book *Python & AI for Algorithmic Trading*. It focuses on three core tasks:

* retrieving historical candle data for a single instrument,
* requesting current bid/ask prices, and
* accessing basic account summary information.

The implementation is intentionally compact. It is inspired by the more
feature-rich :mod:`tpqoa` package used in other projects but omits advanced
order-management and streaming logic. For those use cases, consult the
original wrapper.

The class :class:`OandaClient` is designed for interactive use in notebooks,
scripts in the :mod:`code` directory, and small smoke tests. A convenience
constructor :meth:`OandaClient.from_creds` reads the Oanda credentials from
``code/creds.py`` so that examples do not have to hard-code secrets.
"""

from __future__ import annotations

from dataclasses import dataclass  # lightweight container for configuration
from datetime import datetime, timedelta, timezone  # date and time handling
from pathlib import Path  # filesystem paths for locating creds.py
import sys  # access to Python's module search path
import json  # JSON decoding for REST responses
import pandas as pd  # tabular time-series structures
import v20  # official Oanda v20 Python client


@dataclass
class OandaClient:
    """Minimal client for the Oanda v20 REST API.

    Parameters
    ----------
    access_token:
        Personal Oanda API token used for authentication. The token is stored
        as an attribute but never logged or printed by this module.
    account_id:
        Oanda account identifier, for example ``\"101-004-...\"``.
    account_type:
        Either ``\"practice\"`` (default) for demo accounts or ``\"live\"``
        for funded accounts. The account type determines which hostnames are
        used for REST and streaming endpoints.
    """

    access_token: str
    account_id: str
    account_type: str="practice"

    def __post_init__(self) -> None:
        """Initialise REST contexts and hostnames based on account type."""
        acc_type = self.account_type.lower()
        if acc_type == "live":
            self.hostname = "api-fxtrade.oanda.com"
            self.stream_hostname = "stream-fxtrade.oanda.com"
        else:
            self.hostname = "api-fxpractice.oanda.com"
            self.stream_hostname = "stream-fxpractice.oanda.com"

        self.ctx = v20.Context(
            hostname=self.hostname,
            port=443,
            token=self.access_token,
            poll_timeout=10,
        )
        self.ctx_stream = v20.Context(
            hostname=self.stream_hostname,
            port=443,
            token=self.access_token,
        )

        self._time_suffix = "Z"  # UTC designator for API timestamps

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    @classmethod
    def from_creds(cls) -> "OandaClient":
        """Create a client instance using a local ``creds.py`` module.

        The companion repository keeps Oanda credentials in a small module
        named ``creds.py`` under the ``code`` directory. This constructor
        looks for that file relative to the project root and imports
        ``oanda_access_token``, ``oanda_account_id``, and
        ``oanda_account_type`` from it.
        """
        project_root = Path(__file__).resolve().parents[1]
        code_dir = project_root/"code"
        if code_dir.is_dir() and str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))  # prefer project-local creds.py
        try:
            import creds as _creds  # type: ignore  # local credentials module

            token = _creds.oanda_access_token
            account_id = _creds.oanda_account_id
            account_type = _creds.oanda_account_type
        except Exception as exc:  # pragma: no cover - configuration errors
            msg = (
                "Unable to import Oanda credentials from code/creds.py; "
                "expected attributes oanda_access_token, "
                "oanda_account_id, and oanda_account_type."
            )
            raise RuntimeError(msg) from exc

        return cls(
            access_token=token,
            account_id=account_id,
            account_type=account_type,
        )

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------

    def _to_api_time(self, value: datetime | str) -> str:
        """Convert datetime or string to Oanda-compatible ISO timestamp."""
        if isinstance(value, str):  # parse string into datetime first
            dt_value = pd.Timestamp(value).to_pydatetime()
        else:
            dt_value = value
        # Ensure that the timestamp is treated as UTC and formatted with a
        # trailing 'Z' designator, as expected by the Oanda v20 API.
        if dt_value.tzinfo is None:
            dt_value = dt_value.replace(tzinfo=timezone.utc)
        iso = dt_value.isoformat("T")
        return iso.replace("+00:00", self._time_suffix)

    # ---------------------------------------------------------------------
    # Public API methods
    # ---------------------------------------------------------------------

    def get_account_summary(self, detailed: bool=False) -> dict:
        """Return a dictionary with basic account summary information.

        Parameters
        ----------
        detailed:
            When ``True``, request the full account object; when ``False``,
            request the compact account summary.
        """
        if detailed:
            response = self.ctx.account.get(self.account_id)
        else:
            response = self.ctx.account.summary(self.account_id)
        account = response.get("account")
        return account.dict()

    def get_prices(self, instrument: str) -> tuple[str, float, float]:
        """Return current bid and ask prices for a single instrument.

        Parameters
        ----------
        instrument:
            Instrument name in Oanda notation, for example ``\"EUR_USD\"``.
        """
        resp = self.ctx.pricing.get(self.account_id, instruments=instrument)
        parsed = json.loads(resp.raw_body)  # decode JSON payload
        price_entry = parsed.get("prices", [])[0]
        time_str = price_entry.get("time")
        bid = float(price_entry.get("closeoutBid"))
        ask = float(price_entry.get("closeoutAsk"))
        return time_str, bid, ask

    def get_candles(
        self,
        instrument: str,
        start: datetime | str,
        end: datetime | str,
        granularity: str="D",
        price: str="M",
        localize: bool=True,
    ) -> pd.DataFrame:
        """Retrieve historical candle data for a single instrument.

        Parameters
        ----------
        instrument:
            Instrument name in Oanda notation, for example ``\"EUR_USD\"``.
        start, end:
            Start and end timestamps for the history window. Both can be
            :class:`datetime.datetime` instances or ISO-format strings.
        granularity:
            Candle granularity such as ``\"D\"`` (daily), ``\"H1\"``, or
            ``\"M1\"``. See the Oanda v20 documentation for the full list.
        price:
            One of ``\"A\"`` (ask), ``\"B\"`` (bid), or ``\"M\"`` (mid).
        localize:
            When ``True`` (default), drop timezone information from the index.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by candle time with columns ``o`` (open),
            ``h`` (high), ``l`` (low), ``c`` (close), ``volume``, and
            ``complete``.
        """
        start_ts = self._to_api_time(start)
        end_ts = self._to_api_time(end)

        response = self.ctx.instrument.candles(
            instrument=instrument,
            fromTime=start_ts,
            toTime=end_ts,
            granularity=granularity,
            price=price,
        )
        raw_candles = response.get("candles")
        raw = [cs.dict() for cs in raw_candles]
        if price == "A":
            for cs in raw:
                cs.update(cs["ask"])
                del cs["ask"]
        elif price == "B":
            for cs in raw:
                cs.update(cs["bid"])
                del cs["bid"]
        elif price == "M":
            for cs in raw:
                cs.update(cs["mid"])
                del cs["mid"]
        else:
            msg = "price must be one of 'A', 'B', or 'M'"
            raise ValueError(msg)

        if not raw:
            return pd.DataFrame()

        data = pd.DataFrame(raw)
        data["time"] = pd.to_datetime(data["time"])
        data = data.set_index("time")
        if localize:
            data.index = data.index.tz_localize(None)
        for col in list("ohlc"):
            data[col] = data[col].astype(float)
        return data[["o", "h", "l", "c", "volume", "complete"]]

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: float | None=None,
        take_profit: float | None=None,
        trailing_stop_distance: float | None=None,
    ) -> dict:
        """Place a market order with optional SL/TP/TSL and return a summary.

        Parameters
        ----------
        instrument:
            Instrument name in Oanda notation, for example ``\"EUR_USD\"``.
        units:
            Number of units to trade; positive for a long position and
            negative for a short position.
        stop_loss:
            Absolute price level for a protective stop-loss order (optional).
        take_profit:
            Absolute price level for a take-profit order (optional).
        trailing_stop_distance:
            Distance in price units for a trailing stop-loss order
            (optional).
        """
        order_kwargs: dict[str, object] = {
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        tx = self.ctx.transaction
        if take_profit is not None:
            order_kwargs["takeProfitOnFill"] = tx.TakeProfitDetails(
                price=f"{take_profit:.5f}",
            )
        if stop_loss is not None:
            order_kwargs["stopLossOnFill"] = tx.StopLossDetails(
                price=f"{stop_loss:.5f}",
            )
        if trailing_stop_distance is not None:
            order_kwargs["trailingStopLossOnFill"] = tx.TrailingStopLossDetails(
                distance=f"{trailing_stop_distance:.5f}",
            )

        response = self.ctx.order.market(self.account_id, **order_kwargs)
        body = response.body or {}
        order_fill = body.get("orderFillTransaction")
        fill_dict: dict[str, object]
        if order_fill is not None:
            fill_dict = order_fill.dict()
        else:
            fill_dict = {}

        trade_open = fill_dict.get("tradeOpened") or {}
        if isinstance(trade_open, dict):
            trade_id = trade_open.get("tradeID")
        else:
            trade_id = None

        summary = {
            "instrument": instrument,
            "units": units,
            "order_id": fill_dict.get("orderID"),
            "trade_id": trade_id,
            "price": fill_dict.get("price"),
            "time": fill_dict.get("time"),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop_distance": trailing_stop_distance,
        }
        return summary

    def get_open_trades(self, instrument: str | None=None) -> list[dict]:
        """Return a list of currently open trades as dictionaries."""
        kwargs: dict[str, object] = {"state": "OPEN"}
        if instrument is not None:
            kwargs["instrument"] = instrument
        response = self.ctx.trade.list(self.account_id, **kwargs)
        body = response.body or {}
        trades = body.get("trades", []) or []
        return [tr.dict() for tr in trades]

    def close_trades(self, instrument: str | None=None) -> list[dict]:
        """Close all open trades, optionally restricted to one instrument."""
        trades = self.get_open_trades(instrument=instrument)
        closed: list[dict] = []
        for trade in trades:
            trade_id = trade.get("id")
            if trade_id is None:
                continue
            resp = self.ctx.trade.close(self.account_id, trade_id, units="ALL")
            body = resp.body or {}
            fill = body.get("orderFillTransaction")
            closed.append(
                {
                    "trade_id": trade_id,
                    "instrument": trade.get("instrument"),
                    "closed_units": trade.get("currentUnits"),
                    "price": getattr(fill, "price", None),
                    "time": getattr(fill, "time", None),
                },
            )
        return closed

    def get_transactions_since(
        self,
        last_transaction_id: str | int,
        type_filter: list[str] | None=None,
    ) -> list[dict]:
        """Return transactions that occurred after ``last_transaction_id``."""
        resp = self.ctx.transaction.since(
            self.account_id,
            id=str(last_transaction_id),
        )
        body = resp.body or {}
        transactions = body.get("transactions", []) or []
        result: list[dict] = []
        for tx in transactions:
            tx_dict = tx.dict()
            if type_filter is not None:
                if tx_dict.get("type") not in type_filter:
                    continue
            result.append(tx_dict)
        return result


def _smoke_test_account() -> None:
    """Run a tiny account-summary request as a smoke test."""
    client = OandaClient.from_creds()
    summary = client.get_account_summary(detailed=False)
    assert summary.get("id") == client.account_id
    balance = summary.get("balance")
    nav = summary.get("NAV")
    open_trades = summary.get("openTradeCount")
    print("Oanda account summary:")  # concise console feedback
    print(f"  id           : {summary.get('id')}")  # account identifier
    if balance is not None:
        print(f"  balance      : {balance}")  # account balance
    if nav is not None:
        print(f"  NAV          : {nav}")  # net asset value
    if open_trades is not None:
        print(f"  open trades  : {open_trades}")  # number of open trades


def _smoke_test_history() -> None:
    """Run a tiny EUR_USD history request as a smoke test."""
    client = OandaClient.from_creds()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    data = client.get_candles(
        "EUR_USD",
        start=start,
        end=end,
        granularity="D",
        price="M",
    )
    msg = "Oanda history for EUR_USD should not be empty"
    assert not data.empty, msg
    start_date = data.index[0].date()
    end_date = data.index[-1].date()
    print("\nEUR_USD daily candle history (Oanda):")  # concise header
    print(f"  rows         : {len(data)}")  # number of candles retrieved
    print(f"  date range   : {start_date} to {end_date}")  # coverage
    print("  last 3 rows  :")  # trailing sample for quick inspection
    print(data.tail(3))


if __name__ == "__main__":
    print("Running OandaClient smoke tests...\n")  # overall header
    _smoke_test_account()
    _smoke_test_history()
    print("\nAll OandaClient smoke tests completed successfully.")  # footer
