from pathlib import Path  # filesystem paths for macro and figure output

from pprint import pprint  # pretty printing of metric dictionaries
import numpy as np  # numerical arrays and linear algebra
import pandas as pd  # tabular time-series structures
import matplotlib.pyplot as plt  # plotting library for figures

"""
Python & AI for Algorithmic Trading
Chapter 10 -- Object-Oriented Backtesting and Trading System Design

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Object-oriented OLS backtest for daily returns.

This script illustrates a small object-oriented trading system built
around the ordinary-least-squares (OLS) prediction example from
Chapter 8. The goal is to show how data handling, strategy logic, and
portfolio evaluation can be organised in classes without losing the
clarity of the vectorised approach.

The workflow is:

1. Load the static end-of-day price panel and prepare an OLS data set
   with lagged features, reusing helpers from the Chapter 8 script.
2. Wrap the OLS signal-generation logic in a Strategy-style class that
   exposes a ``generate_positions`` method.
3. Use a Portfolio-style class to run the backtest via the shared
   ``backtest_strategy`` helper from Chapter 7 and compute metrics.
4. Export LaTeX macros and a figure comparing buy-and-hold with the
   OLS-based strategy so that the main text can report concrete
   numbers while keeping the code reusable.

The emphasis is on separation of concerns rather than on recreating a
full event-based engine. For an example that simulates events and
orders explicitly, see the Chapter 11 script.
"""

from ch05_eod_engineering import load_eod_panel  # static EOD price panel
from ch07_baseline_strategies import (  # backtest helpers
    backtest_strategy,
    compute_metrics,
    build_positions_buy_and_hold,
)
from ch08_ols_baseline import (  # OLS utilities from Chapter 8
    prepare_ml_dataset,
    standardise_features,
    fit_ols,
    predict_ols,
    sweep_ols_parameters,
)

plt.style.use("seaborn-v0_8")  # consistent plotting style across figures


class EODDataHandler:
    """Minimal data handler for end-of-day prices."""

    def __init__(
        self,
        symbol: str="EURUSD",
        start: str | None=None,
        end: str | None=None,
    ) -> None:
        self.symbol = symbol
        self.panel = load_eod_panel().astype(float)
        prices = self.panel[self.symbol].dropna()

        start_ts = pd.to_datetime(start) if start is not None else None
        end_ts = pd.to_datetime(end) if end is not None else None
        if start_ts is not None or end_ts is not None:
            prices = prices.loc[start_ts:end_ts]

        self.prices = prices

    def get_prices(self) -> pd.Series:
        """Return the cleaned price series for the configured symbol."""
        return self.prices


class OLSStrategy:
    """OLS-based prediction strategy wrapped as a class."""

    def __init__(
        self,
        symbol: str="EURUSD",
        n_lags: int=5,
        entry_threshold: float=0.002,
        leverage: float=1.0,
        train_fraction: float=0.7,
        start: str | None=None,
        end: str | None=None,
    ) -> None:
        self.symbol = symbol
        self.n_lags = n_lags
        self.entry_threshold = entry_threshold
        self.leverage = leverage
        self.train_fraction = train_fraction
        self.start = start
        self.end = end

        self.beta: np.ndarray | None=None
        self.train_index: pd.DatetimeIndex | None=None
        self.test_index: pd.DatetimeIndex | None=None

    def fit(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit OLS coefficients on a training window."""
        features_lagged, targets = prepare_ml_dataset(
            n_lags=self.n_lags,
            symbol=self.symbol,
        )
          # same feature construction as in Chapter 8

        start_ts = (
            pd.to_datetime(self.start) if self.start is not None else None
        )
        end_ts = (
            pd.to_datetime(self.end) if self.end is not None else None
        )
        if start_ts is not None or end_ts is not None:
            features_lagged = features_lagged.loc[start_ts:end_ts]
            targets = targets.loc[features_lagged.index]

        features_scaled = standardise_features(features_lagged)

        n_obs = features_scaled.shape[0]
        split_idx = int(self.train_fraction * n_obs)

        X_train = features_scaled.iloc[:split_idx].to_numpy()
        y_train = targets.iloc[:split_idx].to_numpy()

        X_test = features_scaled.iloc[split_idx:].to_numpy()
        y_test = targets.iloc[split_idx:].to_numpy()

        self.train_index = features_scaled.index[:split_idx]
        self.test_index = features_scaled.index[split_idx:]

        beta = fit_ols(X_train, y_train)
        self.beta = beta

        return X_train, X_test, y_test

    def generate_positions(self) -> pd.Series:
        """Generate leveraged positions from fitted OLS predictions."""
        if self.beta is None or self.test_index is None:
            raise RuntimeError("fit() must be called before positions.")

        features_lagged, targets = prepare_ml_dataset(
            n_lags=self.n_lags,
            symbol=self.symbol,
        )

        start_ts = (
            pd.to_datetime(self.start) if self.start is not None else None
        )
        end_ts = (
            pd.to_datetime(self.end) if self.end is not None else None
        )
        if start_ts is not None or end_ts is not None:
            features_lagged = features_lagged.loc[start_ts:end_ts]
            targets = targets.loc[features_lagged.index]

        features_scaled = standardise_features(features_lagged)

        n_obs = features_scaled.shape[0]
        split_idx = int(self.train_fraction * n_obs)

        X_test = features_scaled.iloc[split_idx:].to_numpy()

        y_hat_test = predict_ols(self.beta, X_test)

        signal = pd.Series(0.0, index=features_scaled.index)
        base_signal = np.sign(y_hat_test)
        if self.entry_threshold > 0.0:
            active = np.abs(y_hat_test) > self.entry_threshold
            base_signal = np.where(active, base_signal, 0.0)

        signal.iloc[split_idx:] = base_signal

        position = self.leverage * signal.shift(1).fillna(0.0)
          # trade on next day's return
        position = position.loc[self.test_index]
          # restrict to test window for evaluation
        return position


class PortfolioBacktester:
    """Run OLS strategy and benchmark through common backtester."""

    def __init__(
        self,
        data_handler: EODDataHandler,
        strategy: OLSStrategy,
        cost_rate: float=0.0002,
        fin_rate: float=0.00005,
    ) -> None:
        self.data_handler = data_handler
        self.strategy = strategy
        self.cost_rate = cost_rate
        self.fin_rate = fin_rate

    def run(self) -> tuple[pd.Series, pd.Series]:
        """Compute net returns for buy-and-hold and OLS strategy."""
        prices = self.data_handler.get_prices()

        _, _, _ = self.strategy.fit()
        pos_ols = self.strategy.generate_positions()

        prices_test = prices.loc[pos_ols.index]

        pos_bh = build_positions_buy_and_hold(prices_test)

        net_bh = backtest_strategy(
            prices_test,
            pos_bh,
            cost_rate=self.cost_rate,
            fin_rate=self.fin_rate,
        )
        net_ols = backtest_strategy(
            prices_test,
            pos_ols,
            cost_rate=self.cost_rate,
            fin_rate=self.fin_rate,
        )

        return net_bh, net_ols


def format_pct(value: float) -> str:
    """Format a decimal fraction as a percentage string."""
    return f"{value * 100.0: .2f}\\%"  # percentage with two decimals


def format_float(value: float, decimals: int=2) -> str:
    """Format a floating-point number with fixed decimals."""
    fmt = f"{{: .{decimals}f}}"
    return fmt.format(value)


def write_tex_macros(
    metrics_bh: dict[str, float],
    metrics_ols: dict[str, float],
    trades_bh: int,
    trades_ols: int,
    symbol: str,
    n_lags: int,
    entry_threshold: float,
    leverage: float,
    cost_rate: float,
    fin_rate: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    outfile: str="figures/ch10_oop_stats.tex",
) -> None:
    """Write LaTeX macros with OOP OLS vs buy-and-hold statistics."""
    lines: list[str]=[]

    # Buy-and-hold metrics
    lines.append(
        "\\newcommand{\\chTenBhFinalWealth}{"
        f"{format_float(metrics_bh['final_wealth'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenBhAnnReturn}{"
        f"{format_pct(metrics_bh['ann_return'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenBhAnnVol}{"
        f"{format_pct(metrics_bh['ann_vol'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenBhSharpe}{"
        f"{format_float(metrics_bh['sharpe'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenBhMaxDD}{"
        f"{format_pct(metrics_bh['max_drawdown'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenBhHitRatio}{"
        f"{format_pct(metrics_bh['hit_ratio'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenBhNumTrades}{"
        f"{trades_bh}"
        "}\n"
    )

    # OLS strategy metrics
    lines.append(
        "\\newcommand{\\chTenOlsFinalWealth}{"
        f"{format_float(metrics_ols['final_wealth'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenOlsAnnReturn}{"
        f"{format_pct(metrics_ols['ann_return'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenOlsAnnVol}{"
        f"{format_pct(metrics_ols['ann_vol'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenOlsSharpe}{"
        f"{format_float(metrics_ols['sharpe'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenOlsMaxDD}{"
        f"{format_pct(metrics_ols['max_drawdown'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenOlsHitRatio}{"
        f"{format_pct(metrics_ols['hit_ratio'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenOlsNumTrades}{"
        f"{trades_ols}"
        "}\n"
    )

    # Configuration macros
    lines.append(
        "\\newcommand{\\chTenSymbol}{"
        f"{symbol}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenNumLags}{"
        f"{n_lags}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenOlsEntryThreshold}{"
        f"{format_float(entry_threshold, decimals=4)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenOlsLeverage}{"
        f"{format_float(leverage, decimals=1)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenOlsCostRate}{"
        f"{format_float(cost_rate, decimals=5)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenOlsFinRate}{"
        f"{format_float(fin_rate, decimals=5)}"
        "}\n"
    )

    lines.append(
        "\\newcommand{\\chTenTrainStart}{"
        f"{train_start}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenTrainEnd}{"
        f"{train_end}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenTestStart}{"
        f"{test_start}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chTenTestEnd}{"
        f"{test_end}"
        "}\n"
    )

    Path(outfile).write_text("".join(lines), encoding="utf8")


def plot_equity_curves(
    net_bh: pd.Series,
    net_ols: pd.Series,
    symbol: str,
    outfile: str="figures/ch10_oop_vs_bh.pdf",
) -> None:
    """Plot equity curves for buy-and-hold and the OLS strategy."""
    wealth_bh = (1.0 + net_bh).cumprod()
    wealth_ols = (1.0 + net_ols).cumprod()

    wealth_bh = wealth_bh / wealth_bh.iloc[0]
    wealth_ols = wealth_ols / wealth_ols.iloc[0]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    cmap = plt.cm.coolwarm

    w_min = float(min(wealth_bh.min(), wealth_ols.min()))
    w_max = float(max(wealth_bh.max(), wealth_ols.max()))
    padding = 0.05 * (w_max - w_min) if w_max > w_min else 0.02

    ax.plot(
        wealth_bh.index,
        wealth_bh.values,
        label=f"buy-and-hold ({symbol})",
        color=cmap(0.10),
        lw=1.0,
        ls=":",
    )
    ax.plot(
        wealth_ols.index,
        wealth_ols.values,
        label="OOP OLS strategy",
        color=cmap(0.90),
        lw=1.0,
    )

    ax.set_ylabel("wealth (normalised)")
    ax.set_xlabel("date")
    ax.set_title(f"{symbol}: OOP OLS strategy vs buy-and-hold")
    ax.legend(loc="upper left")
    ax.set_ylim(
        max(0.0, w_min - padding),
        w_max + padding,
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def count_trades(position: pd.Series) -> int:
    """Count the number of trades implied by a position series."""
    pos = position.fillna(0.0)
    delta_q = pos.diff().fillna(pos)
    return int((delta_q != 0.0).sum())


def main(
    symbol: str="EURUSD",
    n_lags: int=5,
    entry_threshold: float=0.002,
    leverage: float=1.0,
    train_fraction: float=0.7,
    cost: float=0.0002,
    fin: float=0.00005,
    start: str | None=None,
    end: str | None=None,
    do_sweep: bool=False,
) -> None:
    """Run object-oriented OLS backtest and create figures and macros."""
    sweep_results = sweep_ols_parameters(
        n_lags_list=[1, 3, 5, 10],
        entry_values=[0.0, 0.0001, 0.0002, 0.0005, 0.0010],
        leverage_values=[0.5, 1.0, 1.5, 2.0],
        cost=cost,
        fin=fin,
        train_fraction=train_fraction,
        symbol=symbol,
        start=start,
        end=end,
    )

    if not sweep_results.empty:
        candidates = sweep_results[
            (sweep_results["sharpe"] > 0.0)
            & (sweep_results["final_wealth"] > 1.0)
        ]
        if candidates.empty:
            candidates = sweep_results[sweep_results["sharpe"] > 0.0]
        if candidates.empty:
            candidates = sweep_results
        best = candidates.sort_values(
            by=["sharpe", "ann_return"],
            ascending=[False, False],
        ).iloc[0]
        n_lags_selected = int(best["lags"])
        entry_selected = float(best["entry"])
        leverage_selected = float(best["leverage"])
    else:
        n_lags_selected = n_lags
        entry_selected = entry_threshold
        leverage_selected = leverage

    data_handler = EODDataHandler(
        symbol=symbol,
        start=start,
        end=end,
    )
    strategy = OLSStrategy(
        symbol=symbol,
        n_lags=n_lags_selected,
        entry_threshold=entry_selected,
        leverage=leverage_selected,
        train_fraction=train_fraction,
        start=start,
        end=end,
    )
    backtester = PortfolioBacktester(
        data_handler=data_handler,
        strategy=strategy,
        cost_rate=cost,
        fin_rate=fin,
    )

    net_bh, net_ols = backtester.run()

    metrics_bh = compute_metrics(net_bh)
    metrics_ols = compute_metrics(net_ols)

    trades_bh = count_trades(
        build_positions_buy_and_hold(
            data_handler.get_prices().loc[net_bh.index],
        )
    )
    trades_ols = count_trades(
        strategy.generate_positions(),
    )

    train_start = strategy.train_index[0].date().isoformat()  # type: ignore[index]
    train_end = strategy.train_index[-1].date().isoformat()  # type: ignore[index]
    test_start = strategy.test_index[0].date().isoformat()  # type: ignore[index]
    test_end = strategy.test_index[-1].date().isoformat()  # type: ignore[index]

    write_tex_macros(
        metrics_bh,
        metrics_ols,
        trades_bh,
        trades_ols,
        symbol,
        n_lags_selected,
        entry_selected,
        leverage_selected,
        cost,
        fin,
        train_start,
        train_end,
        test_start,
        test_end,
    )

    plot_equity_curves(net_bh, net_ols, symbol)

    print(f"{symbol} OOP OLS backtest metrics (test window):")
    print("buy-and-hold:")
    pprint(metrics_bh)
    print("OOP OLS strategy:")
    pprint(metrics_ols)

    if do_sweep and not sweep_results.empty:
        csv_path = Path("figures/ch10_oop_parameter_sweep.csv")
        sweep_results.to_csv(csv_path, index=False)
        print("Top OOP OLS parameter sets (by Sharpe):")
        print(
            sweep_results[
                [
                    "lags",
                    "entry",
                    "leverage",
                    "final_wealth",
                    "ann_return",
                    "ann_vol",
                    "sharpe",
                    "max_drawdown",
                    "hit_ratio",
                    "num_trades",
                ]
            ]
            .head(5)
            .to_string(
                index=False,
                float_format=lambda v: f"{v: .4f}",
            )
        )


if __name__ == "__main__":
    import argparse  # command-line argument parsing

    parser = argparse.ArgumentParser(
        description=(
            "Object-oriented OLS backtest for a single symbol."
        ),
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="symbol to backtest (default: EURUSD)",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=5,
        help="number of daily feature lags (default: 5)",
    )
    parser.add_argument(
        "--entry",
        type=float,
        default=0.002,
        help="entry threshold for predicted returns (default: 0.002)",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="leverage factor applied to the strategy (default: 1.0)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="fraction of sample used for training (default: 0.7)",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.0002,
        help="transaction-cost rate per unit turnover (default: 0.0002)",
    )
    parser.add_argument(
        "--fin",
        type=float,
        default=0.00005,
        help="daily financing rate on absolute position (default: 0.00005)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help='optional start date for the sample, for example "2020-01-01"',
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help='optional end date for the sample, for example "2025-01-01"',
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="export parameter sweep as CSV",
    )

    args = parser.parse_args()
    main(
        symbol=args.symbol,
        n_lags=args.lags,
        entry_threshold=args.entry,
        leverage=args.leverage,
        train_fraction=args.train_fraction,
        cost=args.cost,
        fin=args.fin,
        start=args.start,
        end=args.end,
        do_sweep=args.sweep,
    )
