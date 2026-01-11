from pathlib import Path  # filesystem paths for macro output

import numpy as np  # numerical arrays and statistics
import pandas as pd  # tabular time-series structures
import matplotlib.pyplot as plt  # plotting library for figures

"""
Python & AI for Algorithmic Trading
Chapter 7 -- Baseline Technical Strategies and Vectorised Backtesting

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Baseline technical strategy backtests and figures for Chapter 7.

This script implements a simple moving-average crossover strategy on SPY,
compares it with a buy-and-hold benchmark, and reports performance metrics
that are reused in the LaTeX chapter. The backtest incorporates both
transaction costs and daily financing charges to reflect leveraged trading
on retail platforms.

Running the script performs three main tasks.

1. Load the static end-of-day price panel from Chapter 5 and extract the
   SPY closing-price series.
2. Compute buy-and-hold and 20/100-day moving-average crossover strategies,
   including proportional transaction costs and a small daily financing
   charge on the absolute position.
3. Generate an equity-curve figure for strategy versus benchmark and write
   a small LaTeX macro file with summary statistics such as final wealth,
   annualised volatility, and Sharpe ratios.

The code is intentionally compact but fully commented so that it can serve
as a blueprint for more elaborate strategy backtests later in the book.
"""

from ch05_eod_engineering import load_eod_panel  # static EOD price panel

plt.style.use("seaborn-v0_8")  # consistent plotting style across figures


def max_drawdown(wealth: pd.Series) -> float:
    """Compute the maximum drawdown for a wealth index."""
    running_max = wealth.cummax()  # running peak of the equity curve
    drawdown = (wealth / running_max) - 1.0  # relative drop from the peak
    return float(drawdown.min())  # most negative drawdown over the sample


def compute_metrics(net_ret: pd.Series) -> dict[str, float]:
    """Compute basic performance metrics from daily net returns."""
    wealth = (1.0 + net_ret).cumprod()  # cumulative wealth from 1.0
    final_wealth = float(wealth.iloc[-1])  # terminal wealth at last date

    mean_daily = float(net_ret.mean())  # mean daily net return
    vol_daily = float(net_ret.std(ddof=1))  # daily return volatility
    ann_factor = np.sqrt(252.0)  # trading days per year for scaling

    n_days = float(net_ret.shape[0])  # number of trading days in sample
    if n_days > 0.0:
        ann_return = final_wealth ** (252.0 / n_days) - 1.0
          # path-consistent annualised return
    else:
        ann_return = 0.0
    ann_vol = vol_daily * ann_factor  # annualised volatility
    sharpe = ann_return / ann_vol if ann_vol > 0.0 else 0.0  # Sharpe ratio

    mdd = max_drawdown(wealth)  # maximum drawdown of the equity curve
    hit_ratio = float((net_ret > 0.0).mean()) if not net_ret.empty else 0.0

    return {
        "final_wealth": final_wealth,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "hit_ratio": hit_ratio,
    }


def backtest_strategy(
    prices: pd.Series,
    position: pd.Series,
    cost_rate: float=0.0005,
    fin_rate: float=0.0001,
) -> pd.Series:
    """Backtest a daily strategy with costs and financing."""
    prices = prices.astype(float)  # ensure numeric price series
    rets = prices.pct_change(fill_method=None).dropna()
      # simple daily returns without forward-filling missing prices

    pos = position.reindex(rets.index).fillna(0.0)  # align positions
    gross_ret = pos * rets  # strategy return before costs and financing

    delta_q = pos.diff().fillna(pos)  # position changes for turnover
    costs = cost_rate * delta_q.abs()  # proportional transaction costs

    financing = fin_rate * pos.abs()  # daily financing on exposure

    net_ret = gross_ret - costs - financing  # net daily strategy returns
    return net_ret


def build_positions_sma(prices: pd.Series) -> pd.Series:
    """Construct a 20/100-day moving-average crossover position."""
    fast = prices.rolling(20).mean()  # short lookback moving average
    slow = prices.rolling(100).mean()  # long lookback moving average

    signal = (fast > slow).astype(float)  # 1.0 when fast above slow
    position = signal.shift(1).fillna(0.0)  # trade on next day's return
    return position


def build_positions_buy_and_hold(prices: pd.Series) -> pd.Series:
    """Construct a constant long position for buy-and-hold."""
    rets_index = prices.pct_change(fill_method=None).dropna().index
      # index of returns without forward-filling missing prices
    position = pd.Series(1.0, index=rets_index)  # always fully invested
    return position


def count_trades(position: pd.Series) -> int:
    """Count the number of trades implied by a position series."""
    pos = position.fillna(0.0)
    delta_q = pos.diff().fillna(pos)
    return int((delta_q != 0.0).sum())


def format_pct(value: float) -> str:
    """Format a decimal fraction as a percentage string."""
    return f"{value * 100.0: .2f}\\%"  # percentage with two decimals


def format_float(value: float, decimals: int=2) -> str:
    """Format a floating-point number with fixed decimals."""
    fmt = f"{{: .{decimals}f}}"  # template for decimal formatting
    return fmt.format(value)  # resulting string (no thousands separator)


def write_tex_macros(
    metrics_bh: dict[str, float],
    metrics_sma: dict[str, float],
    trades_bh: int,
    trades_sma: int,
    outfile: str="figures/ch07_sma_stats.tex",
) -> None:
    """Write LaTeX macros with SMA vs buy-and-hold statistics."""
    lines: list[str]=[]  # collect macro definition lines

    lines.append(
        "\\newcommand{\\chSevenBhFinalWealth}{"
        f"{format_float(metrics_bh['final_wealth'])}"
        "}\n"
    )  # buy-and-hold terminal wealth
    lines.append(
        "\\newcommand{\\chSevenSmaFinalWealth}{"
        f"{format_float(metrics_sma['final_wealth'])}"
        "}\n"
    )  # SMA terminal wealth

    lines.append(
        "\\newcommand{\\chSevenBhAnnReturn}{"
        f"{format_pct(metrics_bh['ann_return'])}"
        "}\n"
    )  # annualised buy-and-hold return
    lines.append(
        "\\newcommand{\\chSevenSmaAnnReturn}{"
        f"{format_pct(metrics_sma['ann_return'])}"
        "}\n"
    )  # annualised SMA return

    lines.append(
        "\\newcommand{\\chSevenBhSharpe}{"
        f"{format_float(metrics_bh['sharpe'])}"
        "}\n"
    )  # buy-and-hold Sharpe ratio
    lines.append(
        "\\newcommand{\\chSevenSmaSharpe}{"
        f"{format_float(metrics_sma['sharpe'])}"
        "}\n"
    )  # SMA Sharpe ratio

    lines.append(
        "\\newcommand{\\chSevenBhMaxDD}{"
        f"{format_pct(metrics_bh['max_drawdown'])}"
        "}\n"
    )  # buy-and-hold maximum drawdown
    lines.append(
        "\\newcommand{\\chSevenSmaMaxDD}{"
        f"{format_pct(metrics_sma['max_drawdown'])}"
        "}\n"
    )  # SMA maximum drawdown

    lines.append(
        "\\newcommand{\\chSevenBhHitRatio}{"
        f"{format_pct(metrics_bh['hit_ratio'])}"
        "}\n"
    )  # buy-and-hold hit ratio
    lines.append(
        "\\newcommand{\\chSevenSmaHitRatio}{"
        f"{format_pct(metrics_sma['hit_ratio'])}"
        "}\n"
    )  # SMA hit ratio

    lines.append(
        "\\newcommand{\\chSevenBhNumTrades}{"
        f"{trades_bh}"
        "}\n"
    )  # number of trades for buy-and-hold
    lines.append(
        "\\newcommand{\\chSevenSmaNumTrades}{"
        f"{trades_sma}"
        "}\n"
    )  # number of trades for SMA strategy

    stats_path = Path(outfile)  # location of the macro file
    stats_path.write_text("".join(lines), encoding="utf8")  # save macros


def plot_equity_curves(
    net_ret_bh: pd.Series,
    net_ret_sma: pd.Series,
    outfile: str="figures/ch07_sma_vs_bh.pdf",
) -> None:
    """Plot equity curves for buy-and-hold and SMA strategies."""
    wealth_bh = (1.0 + net_ret_bh).cumprod()  # benchmark equity curve
    wealth_sma = (1.0 + net_ret_sma).cumprod()  # strategy equity curve

    fig, ax = plt.subplots(figsize=(7.0, 3.2))  # single axes for curves
    cmap = plt.cm.coolwarm  # shared colour map

    ax.plot(
        wealth_bh.index,
        wealth_bh.values,
        label="buy-and-hold (SPY)",
        color=cmap(0.10),
        lw=1.0,
    )
    ax.plot(
        wealth_sma.index,
        wealth_sma.values,
        label="20/100-day SMA",
        color=cmap(0.90),
        lw=1.0,
    )

    ax.set_ylabel("wealth (normalised)")  # y-axis label
    ax.set_xlabel("date")  # x-axis label
    ax.set_title("SPY: SMA crossover vs buy-and-hold")  # figure title
    ax.legend(loc="upper left")  # legend for curves
    ax.grid(True, alpha=0.3)  # light grid for readability

    fig.tight_layout()  # reduce whitespace
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # free Matplotlib resources


def main() -> None:
    """Run SMA and buy-and-hold backtests and create outputs."""
    panel = load_eod_panel()  # static price panel from Chapter 5
    prices_spy = panel["SPY"]  # SPY closing-price series

    pos_bh = build_positions_buy_and_hold(prices_spy)  # benchmark position
    pos_sma = build_positions_sma(prices_spy)  # SMA crossover position

    trades_bh = count_trades(pos_bh)  # number of trades for benchmark
    trades_sma = count_trades(pos_sma)  # number of trades for SMA strategy

    net_ret_bh = backtest_strategy(prices_spy, pos_bh)  # benchmark returns
    net_ret_sma = backtest_strategy(prices_spy, pos_sma)  # SMA returns

    # Align both series so that the comparison starts when the SMA strategy
    # first takes a non-zero position. Earlier days correspond to the warm-up
    # period where moving averages are still forming.
    non_flat = pos_sma[pos_sma != 0.0]  # days with active exposure
    if not non_flat.empty:  # guard against degenerate cases
        start_date = non_flat.index[0]  # first day with non-zero position
        net_ret_sma = net_ret_sma.loc[start_date:]  # trim strategy returns
        net_ret_bh = net_ret_bh.loc[start_date:]  # trim benchmark in sync

    metrics_bh = compute_metrics(net_ret_bh)  # stats for buy-and-hold
    metrics_sma = compute_metrics(net_ret_sma)  # stats for SMA strategy

    write_tex_macros(
        metrics_bh,
        metrics_sma,
        trades_bh,
        trades_sma,
    )  # LaTeX helper macros
    plot_equity_curves(net_ret_bh, net_ret_sma)  # equity-curve figure

    # Print a concise console summary for interactive inspection.
    print("Buy-and-hold (SPY) metrics:")  # header for benchmark stats
    print(metrics_bh)  # dictionary of benchmark metrics
    print("20/100-day SMA metrics:")  # header for SMA stats
    print(metrics_sma)  # dictionary of SMA metrics


if __name__ == "__main__":
    main()  # execute backtests when run as a script
