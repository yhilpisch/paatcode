from dataclasses import dataclass  # trade record container
from pathlib import Path  # filesystem paths for trade logs
from typing import Iterable  # type hints for trade collections

import numpy as np  # numerical helpers for returns and risk
import pandas as pd  # tabular structures for trades and equity

"""
Python & AI for Algorithmic Trading
Chapter 19 -- Risk Management and Post-Trading Analysis

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Trade-level risk and post-trade analytics helpers.

This script assumes that trades are available in a CSV-style format
with one row per fill. It provides tools to load such data, compute a
simple marked-to-market equity curve, and derive risk summaries that
can feed into Chapter 19 style reports.
"""


@dataclass
class Trade:
    """Single fill with minimal information for risk analysis."""

    time: pd.Timestamp
    symbol: str
    side: int  # +1 for buy, -1 for sell
    quantity: float
    price: float
    fee: float=0.0


def load_trades(path: Path) -> pd.DataFrame:
    """Load trades from CSV or create a small synthetic sample."""
    if path.is_file():
        df = pd.read_csv(
            path,
            parse_dates=["time"],
        )
    else:
        df = synthetic_trades()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    return df


def synthetic_trades(n: int=250) -> pd.DataFrame:
    """Create a small synthetic trade sample for demonstrations."""
    rng = np.random.default_rng(seed=19)
    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=n,
        freq="B",
    )
    prices = 1.10 + 0.05 * np.cumsum(
        rng.normal(loc=0.0, scale=0.01, size=n),
    )
    side = rng.choice([1, -1], size=n)
    qty = rng.integers(1, 4, size=n) * 1_000
    fee = np.full(n, 0.25)

    df = pd.DataFrame(
        {
            "time": dates,
            "symbol": "EURUSD",
            "side": side,
            "quantity": qty,
            "price": prices,
            "fee": fee,
        }
    )
    return df


def trades_to_position_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday position and PnL series from trade data."""
    trades = trades.copy()
    trades["signed_qty"] = trades["side"] * trades["quantity"]
    trades["notional"] = trades["signed_qty"] * trades["price"]

    trades = trades.set_index("time").sort_index()

    pos = trades["signed_qty"].cumsum()
    cash = -(trades["notional"] + trades["fee"]).cumsum()

    df = pd.DataFrame(
        {
            "position": pos,
            "cash": cash,
            "price": trades["price"],
        }
    )
    df["equity"] = df["cash"] + df["position"] * df["price"]
    return df


def equity_to_risk_summary(equity: pd.Series) -> dict[str, float]:
    """Compute simple risk statistics from an equity curve."""
    equity = equity.sort_index()
    daily = equity.resample("B").last().ffill()
    rets = daily.pct_change(fill_method=None).dropna()

    wealth = (1.0 + rets).cumprod()

    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0

    max_dd = float(drawdown.min())
    vol = float(rets.std(ddof=1) * np.sqrt(252.0))
    worst_day = float(rets.min())
    best_day = float(rets.max())

    return {
        "max_drawdown": max_dd,
        "ann_vol": vol,
        "worst_day": worst_day,
        "best_day": best_day,
    }


def position_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise exposure by sign and rough size buckets."""
    last = df.iloc[-1]
    pos = float(last["position"])
    notional = pos * float(last["price"])

    sign = "long" if pos > 0.0 else "short" if pos < 0.0 else "flat"
    bucket = "small"
    if abs(notional) >= 100_000.0:
        bucket = "large"
    elif abs(notional) >= 25_000.0:
        bucket = "medium"

    summary = pd.DataFrame(
        [
            {
                "position": pos,
                "notional": notional,
                "sign": sign,
                "bucket": bucket,
            }
        ]
    )
    return summary


def main() -> None:
    """Run a small end-to-end post-trade analysis demonstration."""
    csv_path = Path("data") / "trades_eurusd_demo.csv"
    trades = load_trades(csv_path)

    df = trades_to_position_pnl(trades)
    summary = equity_to_risk_summary(df["equity"])
    bucket = position_bucket_summary(df)

    print("Risk summary from equity curve:")
    for key, value in summary.items():
        print(f"  {key}: {value: .4f}")

    print("\nCurrent position bucket:")
    print(bucket.to_string(index=False))


if __name__ == "__main__":
    main()

