"""Python & AI for Algorithmic Trading
Chapter 6 -- Financial Data and EODHD Integration

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Lightweight EODHD client for book examples.

This module provides a small, well-documented wrapper around the EODHD
application programming interface (API) that is tailored to the needs of the
book *Python & AI for Algorithmic Trading*. It focuses on two core tasks:

* retrieving historical end-of-day (EOD) prices at daily, weekly, or monthly
  frequency, and
* retrieving historical intraday bars for short lookback windows.

The implementation is intentionally compact. It builds on the more general
`tpqeod.py` wrapper in the neighbouring `eod/` directory but leaves out
advanced functionality such as fundamentals, technical indicators, and
streaming sockets. For those use cases, consult the original wrapper.

The class :class:`EODHDClient` is designed for interactive use in notebooks,
scripts in the :mod:`code` directory, and unit-style tests. A convenience
constructor :meth:`EODHDClient.from_creds` reads the EODHD key from
``eod/creds.py`` so that examples do not have to hard-code credentials.
"""

from __future__ import annotations

from dataclasses import dataclass  # lightweight container for configuration
from datetime import date, datetime  # date and datetime handling
from io import StringIO  # convert CSV text to file-like objects
from pathlib import Path  # filesystem paths for locating creds.py
from typing import Optional  # optional arguments for type hints
import sys  # access to Python's module search path

import pandas as pd  # tabular time-series data structures
import requests  # HTTP client for API requests


@dataclass
class EODHDClient:
    """Minimal client for the EODHD HTTP API.

    Parameters
    ----------
    api_key:
        Personal EODHD API token used for authentication. The key is stored as
        an attribute but never logged or printed by this module.
    base_url:
        Base URL for the EODHD API. The default points to the public endpoint
        and should not need changing for normal usage.
    """

    api_key: str
    base_url: str="https://eodhd.com/api"  # base URL for all endpoints

    @classmethod
    def from_creds(cls) -> "EODHDClient":
        """Create a client instance using a local ``creds.py`` module.

        The companion repository keeps the EODHD key in a small module named
        ``creds.py``. For convenience, this method first looks for a copy in
        the ``code`` directory of the project and falls back to the original
        ``eod/creds.py`` module in the repository root.
        """
        # Attempt to load credentials from ``code/creds.py`` inside the
        # current project. This path is derived relative to this file so that
        # it works even when the working directory differs.
        project_root = Path(__file__).resolve().parents[1]
        code_dir = project_root/"code"
        if code_dir.is_dir() and str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))  # prefer project-local creds.py
        try:
            import creds as _creds  # type: ignore  # local credentials module

            eod_key = _creds.eod_key  # attribute defined in creds.py
        except Exception:
            # Fallback: use the original credentials module under ``eod/`` at
            # the repository root. This keeps the wrapper usable in contexts
            # where only the original layout is available.
            from eod.creds import eod_key  # type: ignore  # repo-level creds

        return cls(api_key=eod_key)  # client configured with stored key

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _join(self, path: str) -> str:
        """Build a full URL from a relative API path."""
        return f"{self.base_url}/{path}"  # simple string concatenation

    @staticmethod
    def _as_date_string(value: Optional[date | datetime]) -> Optional[str]:
        """Convert a date or datetime to ``YYYY-MM-DD`` format if given."""
        if value is None:  # no bound supplied
            return None
        if isinstance(value, datetime):  # datetime instance
            return value.strftime("%Y-%m-%d")
        if isinstance(value, date):  # plain date instance
            return value.strftime("%Y-%m-%d")
        msg = "start/stop must be datetime.date or datetime.datetime"  # error
        raise TypeError(msg)  # signal unsupported input

    @staticmethod
    def _as_unix_seconds(value: Optional[date | datetime]) -> Optional[int]:
        """Convert a date or datetime to a Unix timestamp in whole seconds."""
        if value is None:  # no bound supplied
            return None
        if isinstance(value, datetime):  # datetime instance
            return int(round(value.timestamp()))
        if isinstance(value, date):  # promote date to datetime at midnight
            dt_value = datetime.combine(value, datetime.min.time())
            return int(round(dt_value.timestamp()))
        msg = "start/stop must be datetime.date or datetime.datetime"  # error
        raise TypeError(msg)  # signal unsupported input

    # -------------------------------------------------------------------------
    # Core public methods: EOD and intraday history
    # -------------------------------------------------------------------------

    def get_eod(
        self,
        symbol: str,
        exchange: str="US",
        period: str="d",
        start: Optional[date | datetime]=None,
        stop: Optional[date | datetime]=None,
        order: str="a",
    ) -> pd.DataFrame:
        """Retrieve historical end-of-day prices for a single symbol.

        Parameters
        ----------
        symbol:
            Ticker symbol such as ``\"SPY\"`` or ``\"AAPL\"``.
        exchange:
            Exchange code used by EODHD (for example ``\"US\"`` or ``\"XETR\"``).
        period:
            Aggregation frequency ``\"d\"`` (daily), ``\"w\"`` (weekly), or
            ``\"m\"`` (monthly). The default is daily.
        start, stop:
            Optional start and end date for the history window. If omitted,
            EODHD returns the full available history for the instrument.
        order:
            ``\"a\"`` for ascending (oldest first) or ``\"d\"`` for descending
            (newest first) order in the result DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by trading date with columns such as ``open``,
            ``high``, ``low``, ``close``, ``adjusted_close``, and ``volume``.
        """
        if period not in {"d", "w", "m"}:  # validate frequency
            msg = "period must be one of 'd', 'w', or 'm'"  # error message
            raise ValueError(msg)
        if order not in {"a", "d"}:  # validate ordering
            msg = "order must be 'a' (ascending) or 'd' (descending)"  # error
            raise ValueError(msg)

        symbol_exchange = f"{symbol}.{exchange}"  # full symbol with exchange
        url = self._join(f"eod/{symbol_exchange}")  # endpoint for EOD data

        params: dict[str, object] = {  # base parameters for the request
            "api_token": self.api_key,
            "period": period,
            "order": order,
        }

        start_str = self._as_date_string(start)  # convert start date
        stop_str = self._as_date_string(stop)  # convert end date
        if start_str is not None:
            params["from"] = start_str  # lower bound for history
        if stop_str is not None:
            params["to"] = stop_str  # upper bound for history

        response = requests.get(url, params=params, timeout=30.0)  # HTTP GET
        if response.status_code != requests.codes.ok:  # non-success status
            code = response.status_code  # HTTP status code as integer
            reason = response.reason  # short description from server
            msg = f"EOD request failed ({code}): {reason}"  # error message
            raise RuntimeError(msg)

        # EODHD returns CSV-formatted text; pandas parses it into a DataFrame
        data = pd.read_csv(
            StringIO(response.text),
            parse_dates=[0],
            index_col=0,
            engine="python",
        )  # resulting index holds trading dates
        return data.sort_index()  # ensure chronological order

    def get_intraday(
        self,
        symbol: str,
        exchange: str="US",
        period: str="1m",
        start: Optional[date | datetime]=None,
        stop: Optional[date | datetime]=None,
    ) -> pd.DataFrame:
        """Retrieve historical intraday bars for a single symbol.

        Parameters
        ----------
        symbol:
            Ticker symbol such as ``\"SPY\"`` or ``\"AAPL\"``.
        exchange:
            Exchange code used by EODHD, for example ``\"US\"``.
        period:
            Bar length as a string; one of ``\"1m\"``, ``\"5m\"``, or ``\"1h\"``.
        start, stop:
            Optional start and end timestamps for the intraday window. Both
            can be :class:`datetime.date` or :class:`datetime.datetime`
            instances. When omitted, EODHD returns its default lookback.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by intraday timestamp with columns such as
            ``open``, ``high``, ``low``, ``close``, and ``volume``.
        """
        if period not in {"1m", "5m", "1h"}:  # validate bar length
            msg = "period must be one of '1m', '5m', or '1h'"  # error message
            raise ValueError(msg)

        symbol_exchange = f"{symbol}.{exchange}"  # full symbol with exchange
        url = self._join(f"intraday/{symbol_exchange}")  # intraday endpoint

        params: dict[str, object] = {
            "api_token": self.api_key,
            "period": period,
        }  # base intraday parameters

        start_ts = self._as_unix_seconds(start)  # lower bound timestamp
        stop_ts = self._as_unix_seconds(stop)  # upper bound timestamp
        if start_ts is not None:
            params["from"] = start_ts  # integer Unix timestamp
        if stop_ts is not None:
            params["to"] = stop_ts  # integer Unix timestamp

        response = requests.get(url, params=params, timeout=30.0)  # HTTP GET
        if response.status_code != requests.codes.ok:  # non-success status
            msg = (
                f"intraday request failed ({response.status_code}): "
                f"{response.reason}"
            )  # error message for callers
            raise RuntimeError(msg)

        data = pd.read_csv(
            StringIO(response.text),
            index_col=2,
            engine="python",
        )  # index is the datetime column returned by the API
        if "Timestamp" in data.columns:
            data = data.drop(columns=["Timestamp"])  # drop raw Unix timestamp
        data.index = pd.to_datetime(data.index)  # convert to pandas datetime
        return data.sort_index()  # ensure chronological order


def _smoke_test_eod() -> None:
    """Run a tiny EOD request as a smoke test.

    The test fetches a short daily price history for ``SPY.US`` using the
    credentials from ``eod/creds.py`` and asserts that the resulting DataFrame
    is non-empty. It is intended for manual execution during development.
    """
    client = EODHDClient.from_creds()  # client configured from local creds
    data = client.get_eod("SPY", "US")  # daily EOD history for SPY
    assert not data.empty, "EOD history for SPY should not be empty"  # check


def _smoke_test_intraday() -> None:
    """Run a tiny intraday request as a smoke test."""
    client = EODHDClient.from_creds()  # client configured from local creds
    data = client.get_intraday("SPY", "US", period="5m")  # short bar series
    msg = "intraday history for SPY should not be empty"  # assertion message
    assert not data.empty, msg  # check DataFrame content


if __name__ == "__main__":
    # When this module is executed directly, run the simple smoke tests. They
    # rely on a valid ``eod_key`` in ``eod/creds.py`` and network access to
    # the EODHD API. Failures are reported via assertion errors or exceptions.
    _smoke_test_eod()
    _smoke_test_intraday()
