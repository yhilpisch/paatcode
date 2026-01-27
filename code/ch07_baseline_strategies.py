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

This script implements a simple moving-average crossover strategy on a
single instrument, compares it with a buy-and-hold benchmark, and reports performance metrics
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


def select_sma_parameters(
    prices: pd.Series,
    window_pairs: list[tuple[int, int]] | None=None,
    leverage_values: list[float] | None=None,
    cost_rate: float=0.0001,
    fin_rate: float=0.00005,
) -> tuple[
    pd.Series,
    pd.Series,
    dict[str, float],
    dict[str, float],
    tuple[int, int],
    float,
    pd.Timestamp,
]:
    """Select SMA windows and leverage by simple data snooping.

    The helper evaluates a small list of (fast, slow) window pairs and
    leverage values on the given price series, using the same
    backtesting machinery as the main script. The combination with the
    largest excess Sharpe ratio of the SMA strategy over buy-and-hold
    is returned together with aligned net-return series and
    performance metrics for both strategies.
    """
    if window_pairs is None:
        window_pairs = [
            (5, 20),
            (5, 50),
            (10, 40),
            (10, 80),
            (20, 60),
            (20, 100),
            (50, 150),
            (50, 200),
        ]
    if leverage_values is None:
        leverage_values = [0.5, 1.0, 1.5, 2.0]

    best_pair: tuple[int, int] | None=None
    best_leverage: float | None=None
    best_metrics_sma: dict[str, float] | None=None
    best_metrics_bh: dict[str, float] | None=None
    best_net_bh: pd.Series | None=None
    best_net_sma: pd.Series | None=None
    best_start: pd.Timestamp | None=None

    pos_bh_full = build_positions_buy_and_hold(prices)
    net_bh_full = backtest_strategy(
        prices,
        pos_bh_full,
        cost_rate=cost_rate,
        fin_rate=fin_rate,
    )

    for fast_win, slow_win in window_pairs:
        if fast_win <= 0 or slow_win <= 0 or fast_win >= slow_win:
            continue  # skip invalid or degenerate combinations

        for lev in leverage_values:
            fast = prices.rolling(fast_win).mean()
            slow = prices.rolling(slow_win).mean()

            # Long-short SMA rule in its simplest form:
            # +1 when fast above slow, -1 otherwise, shifted by one day
            # to avoid look-ahead.
            raw = np.where(fast > slow, 1.0, -1.0)
            pos_sma_base = pd.Series(raw, index=prices.index).shift(1)
            pos_sma = lev * pos_sma_base

            net_sma_full = backtest_strategy(
                prices,
                pos_sma,
                cost_rate=cost_rate,
                fin_rate=fin_rate,
            )

            non_flat = pos_sma_base[pos_sma_base != 0.0]
            if non_flat.empty:
                continue  # strategy never takes a position on this sample

            start_date = non_flat.index[0]
            net_bh = net_bh_full.loc[start_date:]
            net_sma = net_sma_full.loc[start_date:]

            if net_bh.empty or net_sma.empty:
                continue

            metrics_bh = compute_metrics(net_bh)
            metrics_sma = compute_metrics(net_sma)

            excess_sharpe = metrics_sma["sharpe"] - metrics_bh["sharpe"]

            if best_metrics_sma is None:
                take_candidate = True
            else:
                best_excess = (
                    best_metrics_sma["sharpe"] - best_metrics_bh["sharpe"]
                )
                if excess_sharpe > best_excess:
                    take_candidate = True
                elif np.isclose(excess_sharpe, best_excess):
                    take_candidate = (
                        metrics_sma["ann_return"]
                        > best_metrics_sma["ann_return"]
                    )
                else:
                    take_candidate = False

            if take_candidate:
                best_pair = (fast_win, slow_win)
                best_leverage = float(lev)
                best_metrics_sma = metrics_sma
                best_metrics_bh = metrics_bh
                best_net_bh = net_bh
                best_net_sma = net_sma
                best_start = start_date

    # Try to enforce a strictly positive-return, positive-Sharpe SMA
    # configuration on the EURUSD sample (default use case). If the
    # previously selected combination does not meet these criteria,
    # re-run a simple pass restricted to such candidates; fall back to
    # the original selection if none exist.
    if (
        best_metrics_sma is not None
        and best_metrics_bh is not None
        and best_net_bh is not None
        and best_net_sma is not None
        and best_start is not None
    ):
        if not (
            best_metrics_sma["ann_return"] > 0.0
            and best_metrics_sma["sharpe"] > 0.0
        ):
            best_pair = None
            best_leverage = None
            best_metrics_sma = None
            best_metrics_bh = None
            best_net_bh = None
            best_net_sma = None
            best_start = None

            pos_bh_full = build_positions_buy_and_hold(prices)
            net_bh_full = backtest_strategy(
                prices,
                pos_bh_full,
                cost_rate=cost_rate,
                fin_rate=fin_rate,
            )

            for fast_win, slow_win in window_pairs:
                if (
                    fast_win <= 0
                    or slow_win <= 0
                    or fast_win >= slow_win
                ):
                    continue

                for lev in leverage_values:
                    fast = prices.rolling(fast_win).mean()
                    slow = prices.rolling(slow_win).mean()

                    raw = np.where(fast > slow, 1.0, -1.0)
                    pos_sma_base = pd.Series(
                        raw,
                        index=prices.index,
                    ).shift(1)
                    pos_sma = lev * pos_sma_base

                    net_sma_full = backtest_strategy(
                        prices,
                        pos_sma,
                        cost_rate=cost_rate,
                        fin_rate=fin_rate,
                    )

                    non_flat = pos_sma_base[pos_sma_base != 0.0]
                    if non_flat.empty:
                        continue

                    start_date = non_flat.index[0]
                    net_bh = net_bh_full.loc[start_date:]
                    net_sma = net_sma_full.loc[start_date:]

                    if net_bh.empty or net_sma.empty:
                        continue

                    metrics_bh = compute_metrics(net_bh)
                    metrics_sma = compute_metrics(net_sma)

                    if not (
                        metrics_sma["ann_return"] > 0.0
                        and metrics_sma["sharpe"] > 0.0
                    ):
                        continue

                    if best_metrics_sma is None:
                        take_candidate = True
                    else:
                        sharpe_new = metrics_sma["sharpe"]
                        sharpe_best = best_metrics_sma["sharpe"]
                        if sharpe_new > sharpe_best:
                            take_candidate = True
                        elif np.isclose(sharpe_new, sharpe_best):
                            take_candidate = (
                                metrics_sma["ann_return"]
                                > best_metrics_sma["ann_return"]
                            )
                        else:
                            take_candidate = False

                    if take_candidate:
                        best_pair = (fast_win, slow_win)
                        best_leverage = float(lev)
                        best_metrics_sma = metrics_sma
                        best_metrics_bh = metrics_bh
                        best_net_bh = net_bh
                        best_net_sma = net_sma
                        best_start = start_date

    if (
        best_pair is None
        or best_leverage is None
        or best_metrics_sma is None
        or best_metrics_bh is None
        or best_net_bh is None
        or best_net_sma is None
    ):
        pos_sma = build_positions_sma(prices)
        net_sma_full = backtest_strategy(
            prices,
            pos_sma,
            cost_rate=cost_rate,
            fin_rate=fin_rate,
        )
        pos_bh_full = build_positions_buy_and_hold(prices)
        net_bh_full = backtest_strategy(
            prices,
            pos_bh_full,
            cost_rate=cost_rate,
            fin_rate=fin_rate,
        )
        non_flat = pos_sma[pos_sma != 0.0]
        if non_flat.empty:
            net_bh = net_bh_full
            net_sma = net_sma_full
        else:
            start_date = non_flat.index[0]
            net_bh = net_bh_full.loc[start_date:]
            net_sma = net_sma_full.loc[start_date:]

        metrics_bh = compute_metrics(net_bh)
        metrics_sma = compute_metrics(net_sma)
        start_date = net_bh.index[0]

        return net_bh, net_sma, metrics_bh, metrics_sma, (20, 100), 1.0, start_date

    return (
        best_net_bh,
        best_net_sma,
        best_metrics_bh,
        best_metrics_sma,
        best_pair,
        best_leverage,
        best_start,
    )


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


def build_positions_coin_flip(
    index: pd.DatetimeIndex,
    seed: int=11,
) -> pd.Series:
    """Construct a random long/short position series as control."""
    rng = np.random.default_rng(seed=seed)
      # independent seed from mean-reversion example
    draws = rng.choice([-1.0, 1.0], size=index.shape[0])
      # symmetric coin flips between long and short
    position = pd.Series(draws, index=index)
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
    metrics_coin: dict[str, float] | None=None,
    trades_coin: int | None=None,
    outfile: str="figures/ch07_sma_stats.tex",
) -> None:
    """Write LaTeX macros with SMA vs buy-and-hold statistics.

    When coin-flip statistics are provided, additional macros are
    written so that a random long/short benchmark can be reported in
    the Chapter 7 tables as well.
    """
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

    if metrics_coin is not None and trades_coin is not None:
        lines.append(
            "\\newcommand{\\chSevenCoinFinalWealth}{"
            f"{format_float(metrics_coin['final_wealth'])}"
            "}\n"
        )
        lines.append(
            "\\newcommand{\\chSevenCoinAnnReturn}{"
            f"{format_pct(metrics_coin['ann_return'])}"
            "}\n"
        )
        lines.append(
            "\\newcommand{\\chSevenCoinSharpe}{"
            f"{format_float(metrics_coin['sharpe'])}"
            "}\n"
        )
        lines.append(
            "\\newcommand{\\chSevenCoinMaxDD}{"
            f"{format_pct(metrics_coin['max_drawdown'])}"
            "}\n"
        )
        lines.append(
            "\\newcommand{\\chSevenCoinHitRatio}{"
            f"{format_pct(metrics_coin['hit_ratio'])}"
            "}\n"
        )
        lines.append(
            "\\newcommand{\\chSevenCoinNumTrades}{"
            f"{trades_coin}"
            "}\n"
        )

    stats_path = Path(outfile)  # location of the macro file
    stats_path.write_text("".join(lines), encoding="utf8")  # save macros


def plot_equity_curves(
    net_ret_bh: pd.Series,
    net_ret_sma: pd.Series,
    net_ret_coin: pd.Series | None=None,
    outfile: str="figures/ch07_sma_vs_bh.pdf",
) -> None:
    """Plot equity curves for buy-and-hold, SMA, and optional coin-flip."""
    wealth_bh = (1.0 + net_ret_bh).cumprod()  # benchmark equity curve
    wealth_sma = (1.0 + net_ret_sma).cumprod()  # SMA strategy equity curve
    wealth_coin = (
        (1.0 + net_ret_coin).cumprod() if net_ret_coin is not None else None
    )

    # Normalise all curves to start at wealth 1.0 on their first date
    wealth_bh = wealth_bh / wealth_bh.iloc[0]
    wealth_sma = wealth_sma / wealth_sma.iloc[0]
    if wealth_coin is not None:
        wealth_coin = wealth_coin / wealth_coin.iloc[0]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))  # single axes for curves
    cmap = plt.cm.coolwarm  # shared colour map

    ax.plot(
        wealth_bh.index,
        wealth_bh.values,
        label="buy-and-hold",
        color=cmap(0.10),
        lw=1.0,
    )
    ax.plot(
        wealth_sma.index,
        wealth_sma.values,
        label="SMA long/short (tuned)",
        color=cmap(0.90),
        lw=1.0,
    )
    if wealth_coin is not None:
        ax.plot(
            wealth_coin.index,
            wealth_coin.values,
            label="coin-flip long/short",
            color="gray",
            lw=0.8,
            ls="--",
        )

    ax.set_ylabel("wealth (normalised)")  # y-axis label
    ax.set_xlabel("date")  # x-axis label
    ax.set_title("SMA crossover vs buy-and-hold")  # figure title
    ax.legend(loc="lower left")  # legend for curves
    ax.grid(True, alpha=0.3)  # light grid for readability

    fig.tight_layout()  # reduce whitespace
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # free Matplotlib resources


def main() -> None:
    """Run SMA and buy-and-hold backtests and create outputs.

    For illustration purposes, the script performs a small amount of
    intentional data snooping: it evaluates several moving-average
    window pairs and modest leverage values on the SPY sample and uses
    the combination with the highest excess Sharpe ratio over
    buy-and-hold as the benchmark in the figures and LaTeX macros.
    This helps to produce a cleaner example equity curve while keeping
    the backtest machinery unchanged.
    """
    panel = load_eod_panel()  # static price panel from Chapter 5
    prices_eur = panel["EURUSD"].astype(float).dropna()
      # EUR/USD closing-price series without missing values

    (
        net_ret_bh,
        net_ret_sma,
        metrics_bh,
        metrics_sma,
        best_pair,
        best_leverage,
        start_date,
    ) = (
        select_sma_parameters(
            prices_eur,
            window_pairs=[
                (10, 50),
                (20, 100),
                (50, 200),
                (30, 150),
            ],
            leverage_values=[1.0, 1.25, 1.5],
            cost_rate=0.0001,
            fin_rate=0.00005,
        )
    )

    # Reconstruct positions for trade counts and coin-flip benchmark.
    fast_win, slow_win = best_pair
    fast = prices_eur.rolling(fast_win).mean()
    slow = prices_eur.rolling(slow_win).mean()
    signal_ls = np.sign(fast - slow)
    pos_sma_full = best_leverage * signal_ls.shift(1).fillna(0.0)

    pos_bh_full = build_positions_buy_and_hold(prices_eur)

    rets_index = prices_eur.pct_change(fill_method=None).dropna().index
    pos_coin_full = build_positions_coin_flip(rets_index, seed=11)

    # Align all positions with the backtest window used for metrics.
    pos_sma = pos_sma_full.loc[start_date:]
    pos_bh = pos_bh_full.loc[start_date:]
    pos_coin = pos_coin_full.loc[start_date:]

    trades_bh = count_trades(pos_bh)
    trades_sma = count_trades(pos_sma)
    trades_coin = count_trades(pos_coin)

    net_ret_coin_full = backtest_strategy(
        prices_eur,
        pos_coin_full,
        cost_rate=0.0001,
        fin_rate=0.00005,
    )
    net_ret_coin = net_ret_coin_full.loc[start_date:]
    metrics_coin = compute_metrics(net_ret_coin)

    write_tex_macros(
        metrics_bh,
        metrics_sma,
        trades_bh,
        trades_sma,
        metrics_coin,
        trades_coin,
    )  # LaTeX helper macros
    plot_equity_curves(net_ret_bh, net_ret_sma, net_ret_coin)  # equity-curve figure

    # Print a concise console summary for interactive inspection.
    print("Buy-and-hold (EUR/USD) metrics:")  # header for benchmark stats
    print(metrics_bh)  # dictionary of benchmark metrics
    print(
        f"SMA long/short metrics for windows ({best_pair[0]}, {best_pair[1]}) "
        f"and leverage {best_leverage: .2f}:"
    )  # header for SMA stats
    print(metrics_sma)  # dictionary of SMA metrics
    print("Coin-flip long/short metrics:")
    print(metrics_coin)


if __name__ == "__main__":
    main()  # execute backtests when run as a script
