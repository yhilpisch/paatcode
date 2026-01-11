from dataclasses import dataclass  # configuration container
from pathlib import Path  # filesystem paths for reports and figures
from typing import Iterable  # type hints for equity inputs

import numpy as np  # numerical helpers for drawdowns and volatility
import pandas as pd  # tabular time-series structures
import matplotlib.pyplot as plt  # plotting library for report figures

"""
Python & AI for Algorithmic Trading
Chapter 16 -- Reporting and Monitoring for Retail Algotraders

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Reporting helpers and tiny monitoring examples for Chapter 16.

This script focuses on two practical tasks.

1. Turn an equity curve into a small HTML report that shows recent
   values, drawdowns, and rolling volatility alongside a static PNG
   chart.
2. Provide simple portfolio and trade summaries that can be reused
   from schedulable jobs, interactive sessions, or broker-connected
   wrappers.

The functions are written to work with both backtest and live-style
data. They avoid external web frameworks and instead rely on familiar
``pandas`` and Matplotlib tools so that you can adapt them easily to
your own monitoring environment.
"""

plt.style.use("seaborn-v0_8")  # shared plotting style for figures


@dataclass
class EquityReportConfig:
    """Configuration options for HTML equity reports."""

    name: str  # short identifier, for example "eurusd_sma"
    out_dir: Path=Path("reports")  # base folder for all artefacts
    tail_rows: int=120  # number of rows to show in the HTML table
    vol_window: int=20  # window length for rolling volatility


def compute_drawdown(wealth: pd.Series) -> pd.Series:
    """Compute drawdown series from a wealth index.

    Parameters
    ----------
    wealth:
        Time series of cumulative wealth indexed by timestamp.

    Returns
    -------
    pd.Series
        Drawdown series, where zero denotes a running peak and more
        negative values represent deeper drawdowns.
    """
    running_max = wealth.cummax()  # running peak level
    drawdown = wealth / running_max - 1.0  # relative drop from peak
    return drawdown


def equity_table(
    wealth: pd.Series,
    vol_window: int=20,
) -> pd.DataFrame:
    """Build a small diagnostics table for an equity curve."""
    wealth = wealth.sort_index()  # enforce chronological order
    rets = wealth.pct_change(fill_method=None)  # daily net returns
    dd = compute_drawdown(wealth)  # drawdowns from running peak

    vol = rets.rolling(vol_window).std(ddof=1) * np.sqrt(252.0)
      # annualised rolling volatility over the chosen window

    table = pd.DataFrame(
        {
            "wealth": wealth,
            "return": rets,
            "drawdown": dd,
            "ann_vol": vol,
        }
    )
    return table


def plot_equity_and_drawdown(
    wealth: pd.Series,
    cfg: EquityReportConfig,
) -> Path:
    """Create a compact PNG figure for wealth and drawdowns."""
    wealth = wealth.sort_index()
    dd = compute_drawdown(wealth)

    fig, (ax_eq, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(7.0, 3.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    ax_eq.plot(wealth.index, wealth.values, color="tab:blue", lw=1.0)
    ax_eq.set_ylabel("wealth index")
    ax_eq.grid(alpha=0.3)

    ax_dd.fill_between(
        dd.index,
        dd.values,
        0.0,
        color="tab:red",
        alpha=0.3,
    )
    ax_dd.set_ylabel("drawdown")
    ax_dd.set_xlabel("date")
    ax_dd.grid(alpha=0.3)

    fig.tight_layout()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = cfg.out_dir / f"{cfg.name}_equity.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fig_path


def save_equity_report(
    wealth: pd.Series,
    cfg: EquityReportConfig,
) -> Path:
    """Write a minimal HTML report for an equity curve."""
    table = equity_table(
        wealth=wealth,
        vol_window=cfg.vol_window,
    )

    tail = table.tail(cfg.tail_rows)
    html_table = tail.to_html(
        float_format=lambda x: f"{x: .4f}",
        border=0,
        classes="equity-table",
    )

    fig_path = plot_equity_and_drawdown(wealth, cfg)

    html = [
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Equity report</title>",
        "<style>",
        "body { font-family: -apple-system, sans-serif; }",
        ".equity-table { border-collapse: collapse; }",
        ".equity-table th, .equity-table td {",
        "  padding: 4px 6px;",
        "  text-align: right;",
        "}",
        ".equity-table th { background-color: #f0f0f0; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h2>Equity report: {cfg.name}</h2>",
        "<p>Recent values, drawdowns, and rolling volatility.</p>",
        f"<img src='{fig_path.name}' width='640'>",
        "<h3>Last rows</h3>",
        html_table,
        "</body>",
        "</html>",
    ]

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    html_path = cfg.out_dir / f"{cfg.name}_equity_report.html"
    html_path.write_text("\n".join(html), encoding="utf8")

    return html_path


def summarise_positions(positions: pd.DataFrame) -> pd.DataFrame:
    """Summarise current positions by symbol and strategy bucket.

    The input is expected to contain at least the columns
    ``symbol``, ``strategy``, and ``notional``. The function returns
    a grouped table that is convenient for quick monitoring reports.
    """
    required = {"symbol", "strategy", "notional"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"missing columns in positions: {missing}")

    grouped = (
        positions.groupby(["strategy", "symbol"], as_index=False)
        .agg(
            notional=("notional", "sum"),
        )
        .sort_values(
            by=["strategy", "notional"],
            ascending=[True, False],
        )
    )
    return grouped


def build_demo_equity(
    n_days: int=750,
    drift: float=0.08,
    vol: float=0.18,
    seed: int=42,
) -> pd.Series:
    """Generate a synthetic equity curve as demonstration data."""
    rng = np.random.default_rng(seed=seed)
    steps = rng.normal(
        loc=drift / 252.0,
        scale=vol / np.sqrt(252.0),
        size=n_days,
    )
    wealth = np.exp(np.cumsum(steps))
    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=n_days,
        freq="B",
    )
    return pd.Series(wealth, index=dates, name="wealth")


def main() -> None:
    """Run a small demonstration report for synthetic data."""
    equity = build_demo_equity()
    cfg = EquityReportConfig(name="demo_equity")
    html_path = save_equity_report(equity, cfg)
    print(f"Equity report written to {html_path}")  # console summary


if __name__ == "__main__":
    main()

