from pathlib import Path  # filesystem paths for macro and figure output

from pprint import pprint  # pretty printing
import numpy as np  # numerical arrays and statistics
import pandas as pd  # tabular time-series structures
import matplotlib.pyplot as plt  # plotting library for figures
import matplotlib.dates as mdates  # date locators and formatters for axes

"""
Python & AI for Algorithmic Trading
Chapter 7 -- Baseline Technical Strategies and Vectorised Backtesting

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Mean-reversion backtest on EURUSD using the static EOD data set.

This script implements a simple contrarian strategy on daily EURUSD
log-returns. The rule uses a seven-day cumulative return as a short-horizon
time-series momentum signal and takes the opposite side when the move
exceeds a fixed threshold. The backtest is fully vectorised and includes
transaction costs and daily financing charges, matching the structure used
elsewhere in the book.

Running the script performs four main tasks.

1. Load the static end-of-day price panel and extract the EURUSD series.
2. Compute daily log-returns and a seven-day cumulative return feature.
3. Construct three position series: a mean-reversion strategy, a passive
   buy-and-hold benchmark, and a random coin-flip long/short control.
4. Backtest all three with costs and financing, generate equity-curve
   figures, and write LaTeX macros with summary performance statistics.

The aim is to provide a concrete numerical reference for Chapter 7, using
exactly the same data set and cost structure as the earlier diagnostics.
"""

from ch05_eod_engineering import load_eod_panel  # static EOD price panel
from ch07_baseline_strategies import (  # backtest helpers for Chapter 7
    backtest_strategy,
    compute_metrics,
)

plt.style.use("seaborn-v0_8")  # consistent plotting style across figures


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log-returns from a price series."""
    prices = prices.astype(float)  # ensure numeric dtype
    log_ret = np.log(prices / prices.shift(1))  # log-return from prices
    return log_ret.dropna()  # drop first NaN observation


def build_mean_reversion_positions(
    log_ret: pd.Series,
    window: int=9,
    entry_threshold: float=0.006,
    exit_threshold: float=0.002,
    require_flat_between: bool=True,
) -> pd.Series:
    """Construct a contrarian position based on recent momentum.

    The strategy looks at the cumulative log-return over the last ``window``
    trading days. When the move exceeds ``entry_threshold`` in either
    direction it takes the opposite side, betting on a partial reversal.
    Once the cumulative move has reverted back into the band
    ``[-exit_threshold, exit_threshold]`` the position returns to flat.
    Positions are aligned so that signals decided at the end of day ``t-1``
    apply to returns on day ``t``.
    """
    roll_sum = log_ret.rolling(window).sum()  # recent cumulative move

    signal = pd.Series(0.0, index=log_ret.index)  # default flat signal
    signal[roll_sum > entry_threshold] = -1.0  # fade sustained appreciation
    signal[roll_sum < -entry_threshold] = 1.0  # fade sustained depreciation

    neutral = (roll_sum >= -exit_threshold) & (roll_sum <= exit_threshold)
    signal[neutral] = 0.0  # flat after sufficient reversal

    if require_flat_between:
        prev_signal = signal.shift(1).fillna(0.0)  # previous day's signal
        direct_flip = (signal * prev_signal) == -1.0  # +1 to -1 or -1 to +1
        signal = signal.where(~direct_flip, 0.0)  # enforce flat on flip day

    position = signal.shift(1).fillna(0.0)  # apply signal one day later
    return position


def build_buy_and_hold_positions(index: pd.DatetimeIndex) -> pd.Series:
    """Construct a constant long position as benchmark."""
    position = pd.Series(1.0, index=index)  # always fully invested
    return position


def build_coin_flip_positions(
    index: pd.DatetimeIndex,
    seed: int=7,
) -> pd.Series:
    """Construct a random long/short position series as control."""
    rng = np.random.default_rng(seed=seed)  # reproducible randomness
    draws = rng.choice([-1.0, 1.0], size=index.shape[0])  # coin flips
    position = pd.Series(draws, index=index)  # ±1 each day
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
    metrics_mr: dict[str, float],
    metrics_bh: dict[str, float],
    metrics_coin: dict[str, float],
    trades_mr: int,
    trades_bh: int,
    trades_coin: int,
    window: int,
    entry: float,
    exit: float,
    leverage: float,
    cost: float,
    fin: float,
    start_date: str,
    end_date: str,
    outfile: str="figures/ch07_eurusd_mr_stats.tex",
) -> None:
    """Write LaTeX macros with EURUSD backtest statistics."""
    lines: list[str]=[]  # collect macro definition lines

    # Final wealth
    lines.append(
        "\\newcommand{\\chSevenEurMrFinalWealth}{"
        f"{format_float(metrics_mr['final_wealth'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurBhFinalWealth}{"
        f"{format_float(metrics_bh['final_wealth'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurCoinFinalWealth}{"
        f"{format_float(metrics_coin['final_wealth'])}"
        "}\n"
    )

    # Annualised returns
    lines.append(
        "\\newcommand{\\chSevenEurMrAnnReturn}{"
        f"{format_pct(metrics_mr['ann_return'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurBhAnnReturn}{"
        f"{format_pct(metrics_bh['ann_return'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurCoinAnnReturn}{"
        f"{format_pct(metrics_coin['ann_return'])}"
        "}\n"
    )

    # Sharpe ratios
    lines.append(
        "\\newcommand{\\chSevenEurMrSharpe}{"
        f"{format_float(metrics_mr['sharpe'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurBhSharpe}{"
        f"{format_float(metrics_bh['sharpe'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurCoinSharpe}{"
        f"{format_float(metrics_coin['sharpe'])}"
        "}\n"
    )

    # Maximum drawdowns
    lines.append(
        "\\newcommand{\\chSevenEurMrMaxDD}{"
        f"{format_pct(metrics_mr['max_drawdown'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurBhMaxDD}{"
        f"{format_pct(metrics_bh['max_drawdown'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurCoinMaxDD}{"
        f"{format_pct(metrics_coin['max_drawdown'])}"
        "}\n"
    )

    # Hit ratios
    lines.append(
        "\\newcommand{\\chSevenEurMrHitRatio}{"
        f"{format_pct(metrics_mr['hit_ratio'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurBhHitRatio}{"
        f"{format_pct(metrics_bh['hit_ratio'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurCoinHitRatio}{"
        f"{format_pct(metrics_coin['hit_ratio'])}"
        "}\n"
    )

    # Number of trades
    lines.append(
        "\\newcommand{\\chSevenEurMrNumTrades}{"
        f"{trades_mr}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurBhNumTrades}{"
        f"{trades_bh}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurCoinNumTrades}{"
        f"{trades_coin}"
        "}\n"
    )

    # Parameter overview and sample period
    lines.append(
        "\\newcommand{\\chSevenEurWindow}{"
        f"{window}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurEntry}{"
        f"{format_float(entry, decimals=4)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurExit}{"
        f"{format_float(exit, decimals=4)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurLeverage}{"
        f"{format_float(leverage, decimals=1)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurCostRate}{"
        f"{format_float(cost, decimals=5)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurFinRate}{"
        f"{format_float(fin, decimals=5)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurStartDate}{"
        f"{start_date}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chSevenEurEndDate}{"
        f"{end_date}"
        "}\n"
    )

    stats_path = Path(outfile)  # macro file location
    stats_path.write_text("".join(lines), encoding="utf8")  # save macros


def plot_equity_curves(
    net_ret_mr: pd.Series,
    net_ret_bh: pd.Series,
    net_ret_coin: pd.Series,
    outfile: str="figures/ch07_eurusd_mean_reversion_equity.pdf",
) -> None:
    """Plot equity curves for mean-reversion, benchmark, and coin flip."""
    wealth_mr = (1.0 + net_ret_mr).cumprod()  # strategy equity curve
    wealth_bh = (1.0 + net_ret_bh).cumprod()  # buy-and-hold equity curve
    wealth_coin = (1.0 + net_ret_coin).cumprod()  # coin-flip equity

    fig, ax = plt.subplots(figsize=(7.0, 3.2))  # single axes for curves
    cmap = plt.cm.coolwarm  # shared colour map

    ax.plot(
        wealth_mr.index,
        wealth_mr.values,
        label="mean-reversion strategy",
        color=cmap(0.95),
        lw=1.0,
    )
    ax.plot(
        wealth_bh.index,
        wealth_bh.values,
        label="buy-and-hold (EURUSD)",
        color=cmap(0.05),
        lw=1.0,
        ls=':'
    )
    ax.plot(
        wealth_coin.index,
        wealth_coin.values,
        label="coin-flip long/short",
        color=cmap(0.3),
        lw=1.0,
        ls="--",
    )

    ax.set_ylabel("wealth (normalised)")  # y-axis label
    ax.set_xlabel("date")  # x-axis label
    ax.set_title("EURUSD: mean-reversion vs benchmarks")  # figure title
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)  # adaptive date ticks
    ax.xaxis.set_major_formatter(formatter)  # concise labels
    fig.autofmt_xdate()  # rotate and align tick labels when needed
    ax.legend(loc=0)  # legend for curves
    ax.grid(True, alpha=0.3)  # light grid for readability

    fig.tight_layout()  # reduce whitespace
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # free Matplotlib resources


def sweep_mean_reversion_parameters(
    prices: pd.Series,
    log_ret: pd.Series,
    leverage: float=1.0,
    cost_rate: float=0.0001,
    fin_rate: float=0.00001,
    require_flat_between: bool=True,
) -> pd.DataFrame:
    """Explore a small grid of mean-reversion parameters.

    The sweep varies the lookback window and the entry and exit thresholds
    for the cumulative log-return. For each combination it runs a full
    backtest using the shared cost and financing model and records key
    performance statistics.
    """
    windows = [3, 5, 9, 15]  # lookback windows in trading days
    entry_vals = [0.0015, 0.0025, 0.004, 0.006]  # entry thresholds
    exit_vals = [0.0005, 0.001, 0.002]  # exit thresholds

    records: list[dict[str, float | int]] = []  # list of result dictionaries

    for win in windows:
        for entry in entry_vals:
            for exit_ in exit_vals:
                if exit_ >= entry:  # require tighter exit band than entry
                    continue
                pos = build_mean_reversion_positions(
                    log_ret,
                    window=win,
                    entry_threshold=entry,
                    exit_threshold=exit_,
                    require_flat_between=require_flat_between,
                )
                pos = leverage * pos  # apply leverage only to strategy
                net_ret = backtest_strategy(
                    prices,
                    pos,
                    cost_rate=cost_rate,
                    fin_rate=fin_rate,
                )  # net strategy returns with aligned costs
                metrics = compute_metrics(net_ret)  # performance statistics
                records.append(
                    {
                        "window": int(win),
                        "entry": float(entry),
                        "exit": float(exit_),
                        "leverage": float(leverage),
                        "final_wealth": float(metrics["final_wealth"]),
                        "ann_return": float(metrics["ann_return"]),
                        "sharpe": float(metrics["sharpe"]),
                        "max_drawdown": float(metrics["max_drawdown"]),
                    }
                )

    results = pd.DataFrame.from_records(records)  # table of all parameter sets
    if not results.empty:
        results = results.sort_values(
            by=["sharpe", "ann_return"],
            ascending=[False, False],
        )
    return results


def main(
    window: int=9,
    entry: float=0.006,
    exit: float=0.0005,
    leverage: float=1.0,
    cost: float=0.0001,
    fin: float=0.00001,
    do_sweep: bool=False,
    start: str | None=None,
    end: str | None=None,
    require_flat_between: bool=True,
) -> None:
    """Run mean-reversion backtest on EURUSD and create outputs.

    Parameters
    ----------
    window:
        Lookback window in trading days for the cumulative log-return.
    entry:
        Entry threshold for the cumulative move; the strategy takes a
        contrarian position once the absolute cumulative return exceeds
        this level.
    exit:
        Exit threshold; positions return to flat once the cumulative move
        has reverted into the band [-exit, exit].
    leverage:
        Leverage factor applied only to the mean-reversion strategy
        positions, leaving the benchmarks unlevered.
    cost:
        Proportional transaction-cost rate per unit turnover, passed as
        ``cost_rate`` to ``backtest_strategy``.
    fin:
        Daily financing rate applied to the absolute position, passed as
        ``fin_rate`` to ``backtest_strategy``.
    require_flat_between:
        When ``True``, enforce that the signal goes through a flat position
        (zero exposure) before switching directly from long to short or from
        short to long; this avoids instantaneous flips between full leverage
        in opposite directions.
    """
    panel = load_eod_panel()  # static price panel
    prices_eur = panel["EURUSD"].dropna()  # EURUSD closing prices
    if start is not None or end is not None:
        start_ts = pd.to_datetime(start) if start is not None else None
        end_ts = pd.to_datetime(end) if end is not None else None
        prices_eur = prices_eur.loc[start_ts:end_ts]  # cut to requested span

    log_ret = compute_log_returns(prices_eur)  # daily log-returns

    pos_mr = build_mean_reversion_positions(  # contrarian rule
        log_ret,
        window=window,
        entry_threshold=entry,
        exit_threshold=exit,
        require_flat_between=require_flat_between,
    )
    pos_mr = leverage * pos_mr  # scaled exposure for the strategy only
    pos_bh = build_buy_and_hold_positions(log_ret.index)  # benchmark
    pos_coin = build_coin_flip_positions(log_ret.index)  # random control

    trades_mr = count_trades(pos_mr)  # number of strategy trades
    trades_bh = count_trades(pos_bh)  # number of benchmark trades
    trades_coin = count_trades(pos_coin)  # number of coin-flip trades

    net_ret_mr = backtest_strategy(  # strategy returns
        prices_eur,
        pos_mr,
        cost_rate=cost,
        fin_rate=fin,
    )
    net_ret_bh = backtest_strategy(  # benchmark returns
        prices_eur,
        pos_bh,
        cost_rate=cost,
        fin_rate=fin,
    )
    net_ret_coin = backtest_strategy(  # coin-flip returns
        prices_eur,
        pos_coin,
        cost_rate=cost,
        fin_rate=fin,
    )

    metrics_mr = compute_metrics(net_ret_mr)  # stats for strategy
    metrics_bh = compute_metrics(net_ret_bh)  # stats for benchmark
    metrics_coin = compute_metrics(net_ret_coin)  # stats for control

    start_date = prices_eur.index.min().date().isoformat()
    end_date = prices_eur.index.max().date().isoformat()

    write_tex_macros(
        metrics_mr,
        metrics_bh,
        metrics_coin,
        trades_mr,
        trades_bh,
        trades_coin,
        window,
        entry,
        exit,
        leverage,
        cost,
        fin,
        start_date,
        end_date,
    )  # LaTeX macros
    plot_equity_curves(net_ret_mr, net_ret_bh, net_ret_coin)  # equity figure

    # Print a concise console summary for interactive inspection.
    print("EURUSD mean-reversion backtest metrics:")  # header for summary
    print("mean-reversion:")
    pprint(metrics_mr)
    print("buy-and-hold:  ")
    pprint(metrics_bh)
    print("coin flip:     ")
    pprint(metrics_coin)

    if do_sweep:
        # Optional parameter sweep to identify more promising mean-reversion
        # settings. The results are printed sorted by Sharpe ratio so that
        # potential candidates for further study stand out.
        sweep_results = sweep_mean_reversion_parameters(
            prices_eur,
            log_ret,
            leverage=leverage,
            cost_rate=cost,
            fin_rate=fin,
            require_flat_between=require_flat_between,
        )
        if not sweep_results.empty:
            csv_path = Path("figures/ch07_eurusd_mr_parameter_sweep.csv")
            sweep_results.to_csv(csv_path, index=False)  # save full grid results
            print("Top EURUSD mean-reversion parameter sets (by Sharpe):")
            print(
                sweep_results.head(5).to_string(
                    index=False,
                    float_format=lambda v: f"{v: .4f}",
                )
            )


if __name__ == "__main__":
    import argparse  # command-line argument parsing

    parser = argparse.ArgumentParser(
        description="EURUSD mean-reversion backtest with tunable parameters.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=9,
        help="lookback window in trading days (default: 9)",
    )
    parser.add_argument(
        "--entry",
        type=float,
        default=0.006,
        help="entry threshold for cumulative log-return (default: 0.006)",
    )
    parser.add_argument(
        "--exit",
        type=float,
        default=0.0005,
        help="exit threshold for cumulative log-return (default: 0.0005)",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="leverage factor applied only to the strategy (default: 1.0)",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.0001,
        help="transaction-cost rate per unit turnover (default: 0.0001)",
    )
    parser.add_argument(
        "--fin",
        type=float,
        default=0.00001,
        help="daily financing rate on absolute position (default: 0.00001)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="run parameter sweep in addition to the main backtest",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help='start date for the backtest, for example "2020-01-01"',
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help='end date for the backtest, for example "2025-01-01"',
    )
    parser.add_argument(
        "--no-flat-between",
        action="store_false",
        dest="require_flat_between",
        help="allow direct flips between long and short without a flat day",
    )
    parser.set_defaults(sweep=False)

    args = parser.parse_args()
    main(
        window=args.window,
        entry=args.entry,
        exit=args.exit,
        leverage=args.leverage,
        cost=args.cost,
        fin=args.fin,
        do_sweep=args.sweep,
        start=args.start,
        end=args.end,
        require_flat_between=args.require_flat_between,
    )  # execute backtest with CLI or default parameters
