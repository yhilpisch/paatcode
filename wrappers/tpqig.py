"""Python & AI for Algorithmic Trading
Chapter 15 -- IG Markets and the Anatomy of an API Wrapper

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Thin wrapper for the IG Markets REST and streaming APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any
import warnings

import pandas as pd
from trading_ig.rest import IGService

LOG = logging.getLogger(__name__)


@dataclass
class IGClient:
    """Minimal IG helper for the book's data, pricing, and account needs."""

    username: str
    password: str
    api_key: str
    account_type: str = "demo"
    account_number: str | None = None
    service: IGService | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.service is None:
            self.service = IGService(
                username=self.username,
                password=self.password,
                api_key=self.api_key,
                acc_type=self._normalize_account_type(self.account_type),
                acc_number=self.account_number,
            )
        self._session_active = False
        self._epic_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_creds(cls) -> "IGClient":
        """Build a client using the standard IG configuration module."""
        project_root = Path(__file__).resolve().parents[1]
        for candidate in ("code", "ref"):
            provider = project_root / candidate
            if provider.is_dir() and str(provider) not in sys.path:
                sys.path.insert(0, str(provider))
        try:
            from trading_ig_config import config as ig_config
        except Exception:
            try:
                from trading_ig.config import config as ig_config
            except Exception as exc:  # pragma: no cover - missing config
                raise RuntimeError(
                    "Unable to load IG credentials from trading_ig_config or "
                    "trading_ig.config; see ref/trading_ig_config.py."
                ) from exc

        return cls(
            username=ig_config.username,
            password=ig_config.password,
            api_key=ig_config.api_key,
            account_type=getattr(ig_config, "acc_type", "demo"),
            account_number=getattr(ig_config, "acc_number", None),
        )

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _ensure_session(self) -> None:
        if not self._session_active:
            self.service.create_session()
            self._session_active = True

    def close_session(self) -> None:
        if self._session_active:
            try:
                self.service.logout()
            except AttributeError:
                LOG.debug("IG logout endpoint not available in stubbed service.")
            self._session_active = False

    # ------------------------------------------------------------------
    # Public methods used in the book
    # ------------------------------------------------------------------

    def get_account_summary(self) -> dict[str, Any]:
        """Return the current account summary dictionary."""
        self._ensure_session()
        records = self._to_records(self.service.fetch_accounts())
        if not records:
            raise RuntimeError("No IG accounts were returned by the API.")
        if self.account_number:
            match = next(
                (
                    acct
                    for acct in records
                    if acct.get("accountId") == self.account_number
                ),
                None,
            )
            if match:
                return match
        return records[0]

    def get_prices(self, instrument: str) -> tuple[str, float, float]:
        """Return bid/ask prices for an instrument."""
        self._ensure_session()
        epic = self._resolve_epic(instrument)
        market = self.service.fetch_market_by_epic(epic)
        snapshot = self._extract_snapshot(market)
        timestamp = snapshot.get("snapshotTime") or snapshot.get("snapshotTimeUTC")
        bid = self._to_float(snapshot.get("bid"))
        ask = self._to_float(snapshot.get("offer"))
        if bid is None or ask is None:
            raise RuntimeError("IG snapshot did not contain bid/ask levels.")
        return timestamp or "", bid, ask

    def get_candles(
        self,
        instrument: str,
        *,
        granularity: str = "D",
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        num_points: int | None = None,
    ) -> pd.DataFrame:
        """Fetch candles for an instrument, either by range or by count."""
        self._ensure_session()
        epic = self._resolve_epic(instrument)
        if num_points is not None and (start or end):
            raise ValueError("Supply either num_points or start/end dates, not both.")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module=r".*trading_ig\.utils",
            )
            if num_points is not None:
                payload = self.service.fetch_historical_prices_by_epic(
                    epic=epic,
                    resolution=granularity,
                    numpoints=num_points,
                    format=self._format_prices,
                )
            else:
                if start is None or end is None:
                    raise ValueError(
                        "Start and end dates are required when num_points is missing."
                    )
                payload = self.service.fetch_historical_prices_by_epic_and_date_range(
                    epic=epic,
                    resolution=granularity,
                    start_date=self._format_datetime(start),
                    end_date=self._format_datetime(end),
                    format=self._format_prices,
                )
        candles = payload.get("prices")
        if not isinstance(candles, pd.DataFrame):
            raise RuntimeError("IG returned unexpected candle payload.")
        return candles

    def market_order(
        self,
        instrument: str,
        *,
        direction: str,
        size: float,
        stop_distance: float | None = None,
        limit_distance: float | None = None,
        guaranteed_stop: bool = False,
        force_open: bool = True,
        time_in_force: str | None = None,
    ) -> dict[str, Any]:
        """Place a simple market order and return the deal confirmation."""
        self._ensure_session()
        epic, expiry, currency = self._instrument_metadata(instrument)
        result = self.service.create_open_position(
            currency_code=currency,
            direction=direction.upper(),
            epic=epic,
            expiry=expiry,
            force_open=force_open,
            guaranteed_stop=guaranteed_stop,
            level=None,
            limit_distance=limit_distance,
            limit_level=None,
            order_type="MARKET",
            quote_id=None,
            size=size,
            stop_distance=stop_distance,
            stop_level=None,
            trailing_stop=False,
            trailing_stop_increment=None,
            time_in_force=time_in_force,
        )
        records = self._to_records(result)
        return records[0] if records else {}

    def close_position(
        self,
        deal_id: str,
        *,
        instrument: str,
        direction: str,
        size: float,
        time_in_force: str | None = None,
    ) -> dict[str, Any]:
        """Close an existing position via an offsetting market order.

        For the demo use cases in this book, positions are closed by
        sending a market order in the opposite direction with the same
        size. The original ``deal_id`` is kept for logging consistency
        but is not required by the IG endpoint used here.
        """
        _ = deal_id  # preserved for potential logging or extensions
        return self.market_order(
            instrument=instrument,
            direction=direction,
            size=size,
            stop_distance=None,
            limit_distance=None,
            guaranteed_stop=False,
            force_open=False,
            time_in_force=time_in_force,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_epic(self, term: str) -> str:
        term = term.strip()
        cached = self._epic_cache.get(term)
        if cached:
            return cached
        if "." in term:
            epic = term
        else:
            markets = self.service.search_markets(term)
            records = self._to_records(markets)
            if not records:
                raise ValueError(f"No IG markets found for '{term}'.")
            epic = records[0].get("epic")
            if not epic:
                raise RuntimeError("Market payload did not include an epic.")
        self._epic_cache[term] = epic
        return epic

    def _instrument_metadata(self, instrument: str) -> tuple[str, str, str]:
        """Return (epic, expiry, currency) tuple for an instrument."""
        epic = self._resolve_epic(instrument)
        market = self.service.fetch_market_by_epic(epic)
        info: dict[str, Any] = {}
        if isinstance(market, dict):
            info = market.get("instrument", market)
        currency = str(info.get("currency", "USD"))
        expiry = str(info.get("expiry", "DFB"))
        return epic, expiry, currency

    @staticmethod
    def _to_records(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, pd.DataFrame):
            return payload.to_dict(orient="records")
        if isinstance(payload, dict):
            for key in ("markets", "accounts", "prices", "data", "results"):
                if key in payload:
                    return IGClient._to_records(payload[key])
            return [dict(payload)]
        if isinstance(payload, list):
            return payload
        return []

    @staticmethod
    def _normalize_account_type(value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {"practice", "demo"}:
            return "demo"
        if normalized == "live":
            return "live"
        return normalized

    @staticmethod
    def _format_datetime(value: datetime | str) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        raise TypeError("Expected ISO datetime string or datetime-like object.")

    @staticmethod
    def _extract_snapshot(payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return payload.get("snapshot", payload)
        if hasattr(payload, "snapshot"):
            return getattr(payload, "snapshot", {})
        return {}

    @staticmethod
    def _to_float(raw: Any) -> float | None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _format_prices(self, prices: list[dict[str, Any]], version: str) -> pd.DataFrame:
        if isinstance(prices, pd.DataFrame):
            return prices
        rows = []
        for entry in prices:
            timestamp = self._parse_datetime(entry)
            row: dict[str, Any] = {"DateTime": timestamp}
            for field in ("openPrice", "highPrice", "lowPrice", "closePrice"):
                metric = field.replace("Price", "").lower()
                data = entry.get(field, {})
                bid = self._to_float(data.get("bid"))
                ask = self._to_float(data.get("ask"))
                row[f"bid_{metric}"] = bid
                row[f"ask_{metric}"] = ask
                if bid is not None and ask is not None:
                    row[f"mid_{metric}"] = (bid + ask) / 2
                else:
                    row[f"mid_{metric}"] = None
            row["volume"] = self._to_float(entry.get("lastTradedVolume"))
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame = frame.set_index("DateTime")
        frame.index = pd.to_datetime(frame.index)
        frame.sort_index(inplace=True)
        return frame

    @staticmethod
    def _parse_datetime(entry: dict[str, Any]) -> datetime:
        for key in ("snapshotTime", "snapshotTimeUTC"):
            value = entry.get(key)
            if value:
                return pd.to_datetime(value)
        raise RuntimeError("History entry missing timestamp.")
