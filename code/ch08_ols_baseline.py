from pathlib import Path  # filesystem paths for macro and figure output

from pprint import pprint  # pretty printing of metric dictionaries
import numpy as np  # numerical arrays and linear algebra
import pandas as pd  # tabular time-series structures
import matplotlib.pyplot as plt  # plotting library for figures

"""
Python & AI for Algorithmic Trading
Chapter 8 -- Predictive Modelling: OLS, Machine & Deep Learning

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Baseline predictive modelling example for Chapter 8.

This script implements a simple ordinary-least-squares (OLS) regression
on engineered features for a single instrument and turns the resulting predictions into a
trading strategy. The set-up is deliberately minimal:

1. Build a daily feature panel for a chosen symbol using helper functions from
   Chapter 5 (log-returns, rolling means, rolling volatility, and a
   z-score of surprises).
2. Construct a supervised-learning data set by pairing lagged features
   with same-day log-returns as targets; the lag ensures that only
   information available up to the previous close is used.
3. Fit an OLS model on an initial training window and evaluate its
   predictive performance and trading P&L on a later test window.
4. Compare the OLS-based strategy with a buy-and-hold benchmark using
   the backtesting helpers from Chapter 7 and export a small set of
   LaTeX macros and an equity-curve figure for use in the text.

The code follows the same conventions as earlier scripts: fully
vectorised computations, explicit transaction costs and financing
charges, and small helper functions that can be reused interactively.
"""

from ch05_eod_engineering import load_eod_panel, build_feature_panel
from ch07_baseline_strategies import (  # backtest helpers
    backtest_strategy,
    compute_metrics,
    build_positions_buy_and_hold,
)

plt.style.use("seaborn-v0_8")  # consistent plotting style across figures


def prepare_ml_dataset(
    n_lags: int=1,
    symbol: str="EURUSD",
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare lagged-feature data set for daily log-returns.

    Returns
    -------
    features_lagged:
        DataFrame of lagged features (one row per trading day). Each row
        uses only information available up to the previous close.
    targets:
        Series of daily log-returns aligned with the feature rows.
    """
    panel = load_eod_panel()  # full price panel from Chapter 5
    prices = panel[symbol].astype(float).dropna()
      # selecting time-series data for `symbol` without missing values
    log_ret = np.log(prices / prices.shift(1)).dropna()  # daily log-returns

    features = build_feature_panel(log_ret)  # feature panel from Chapter 5

    lagged_frames: list[pd.DataFrame]=[]  # collect lagged feature blocks
    for lag in range(1, n_lags + 1):
        block = features.shift(lag)  # features lagged by given number of days
        block.columns = [f"{c}_lag{lag}" for c in features.columns]
          # make lag explicit in column names
        lagged_frames.append(block)

    features_lagged = pd.concat(lagged_frames, axis=1)  # stacked lagged block
    features_lagged = features_lagged.dropna()  # drop initial NaNs

    targets = log_ret.reindex(features_lagged.index)  # align with features
    targets = targets.dropna()  # drop any residual missing targets

    # Align indices once more in case log-returns had extra NaNs
    common_index = features_lagged.index.intersection(targets.index)
    features_lagged = features_lagged.loc[common_index]
    targets = targets.loc[common_index]

    return features_lagged, targets


def standardise_features(features: pd.DataFrame) -> pd.DataFrame:
    """Standardise each feature column to zero mean and unit variance."""
    mean = features.mean()  # column means
    std = features.std(ddof=0).replace(0.0, 1.0)  # avoid division by zero
    scaled = (features - mean) / std  # z-scored features
    return scaled


def fit_ols(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """Fit an OLS regression with intercept using NumPy linear algebra."""
    X_design = np.column_stack([np.ones_like(y_train), X_train])
    xtx = X_design.T @ X_design  # normal equations matrix
    xty = X_design.T @ y_train  # right-hand side
    beta = np.linalg.solve(xtx, xty)  # OLS coefficients
    return beta


def predict_ols(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute OLS predictions for a design matrix."""
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    return X_design @ beta


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination R^2."""
    resid = y_true - y_pred  # residuals
    ss_res = float(np.sum(resid ** 2))  # residual sum of squares
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))  # total variance
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0


def format_pct(value: float) -> str:
    """Format a decimal fraction as a percentage string."""
    return f"{value * 100.0: .2f}\\%"  # percentage with two decimals


def format_float(value: float, decimals: int=2) -> str:
    """Format a floating-point number with fixed decimals."""
    fmt = f"{{: .{decimals}f}}"  # template for decimal formatting
    return fmt.format(value)  # resulting string (no thousands separator)


def write_tex_macros(
    metrics_bh: dict[str, float],
    metrics_ols: dict[str, float],
    n_trades_bh: int,
    n_trades_ols: int,
    r2_train: float,
    r2_test: float,
    corr_test: float,
    n_lags: int,
    entry_threshold: float,
    leverage: float,
    cost_rate: float,
    fin_rate: float,
    train_fraction: float,
    symbol: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    outfile: str="figures/ch08_predictive_stats.tex",
) -> None:
    """Write LaTeX macros with OLS vs buy-and-hold statistics."""
    lines: list[str]=[]  # collect macro definition lines

    # Buy-and-hold metrics
    lines.append(
        "\\newcommand{\\chEightBhFinalWealth}{"
        f"{format_float(metrics_bh['final_wealth'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightBhAnnReturn}{"
        f"{format_pct(metrics_bh['ann_return'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightBhAnnVol}{"
        f"{format_pct(metrics_bh['ann_vol'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightBhSharpe}{"
        f"{format_float(metrics_bh['sharpe'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightBhMaxDD}{"
        f"{format_pct(metrics_bh['max_drawdown'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightBhHitRatio}{"
        f"{format_pct(metrics_bh['hit_ratio'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightBhNumTrades}{"
        f"{n_trades_bh}"
        "}\n"
    )

    # OLS strategy metrics
    lines.append(
        "\\newcommand{\\chEightOlsFinalWealth}{"
        f"{format_float(metrics_ols['final_wealth'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsAnnReturn}{"
        f"{format_pct(metrics_ols['ann_return'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsAnnVol}{"
        f"{format_pct(metrics_ols['ann_vol'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsSharpe}{"
        f"{format_float(metrics_ols['sharpe'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsMaxDD}{"
        f"{format_pct(metrics_ols['max_drawdown'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsHitRatio}{"
        f"{format_pct(metrics_ols['hit_ratio'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsNumTrades}{"
        f"{n_trades_ols}"
        "}\n"
    )

    # Predictive performance diagnostics
    lines.append(
        "\\newcommand{\\chEightOlsRTwoTrain}{"
        f"{format_float(r2_train, decimals=3)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsRTwoTest}{"
        f"{format_float(r2_test, decimals=3)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsCorrTest}{"
        f"{format_float(corr_test, decimals=3)}"
        "}\n"
    )

    # Configuration macros
    lines.append(
        "\\newcommand{\\chEightSymbol}{"
        f"{symbol}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightNumLags}{"
        f"{n_lags}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsEntryThreshold}{"
        f"{format_float(entry_threshold, decimals=4)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsLeverage}{"
        f"{format_float(leverage, decimals=1)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsCostRate}{"
        f"{format_float(cost_rate, decimals=5)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsFinRate}{"
        f"{format_float(fin_rate, decimals=5)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightOlsTrainFraction}{"
        f"{format_pct(train_fraction)}"
        "}\n"
    )

    # Sample splits
    lines.append(
        "\\newcommand{\\chEightTrainStart}{"
        f"{train_start}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightTrainEnd}{"
        f"{train_end}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightTestStart}{"
        f"{test_start}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chEightTestEnd}{"
        f"{test_end}"
        "}\n"
    )

    Path(outfile).write_text("".join(lines), encoding="utf8")


def plot_equity_curves(
    net_bh: pd.Series,
    net_ols: pd.Series,
    symbol: str,
    outfile: str="figures/ch08_spy_ols_vs_bh.pdf",
) -> None:
    """Plot equity curves for buy-and-hold and OLS-based strategy."""
    wealth_bh = (1.0 + net_bh).cumprod()  # benchmark equity curve
    wealth_ols = (1.0 + net_ols).cumprod()  # OLS-based strategy

    # Normalise both curves to start at wealth 1.0 on the first test date
    wealth_bh = wealth_bh / wealth_bh.iloc[0]
    wealth_ols = wealth_ols / wealth_ols.iloc[0]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))  # single axes for curves
    cmap = plt.cm.coolwarm  # shared colour map

    w_min = float(min(wealth_bh.min(), wealth_ols.min()))
      # smallest wealth across both strategies
    w_max = float(max(wealth_bh.max(), wealth_ols.max()))
      # largest wealth across both strategies
    padding = 0.05 * (w_max - w_min) if w_max > w_min else 0.02
      # vertical padding around the extrema

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
        label="OLS prediction strategy",
        color=cmap(0.90),
        lw=1.0,
    )

    ax.set_ylabel("wealth (normalised)")  # y-axis label
    ax.set_xlabel("date")  # x-axis label
    ax.set_title(f"{symbol}: OLS-based prediction vs buy-and-hold")  # title
    ax.legend(loc="upper left")  # legend for curves
    ax.set_ylim(
        max(0.0, w_min - padding * 2),
        w_max + padding * 2,
    )  # slightly wider y-range for readability
    ax.grid(True, alpha=0.3)  # light grid for readability

    fig.tight_layout()  # reduce unused space
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # free Matplotlib resources


def count_trades(position: pd.Series) -> int:
    """Count the number of trades implied by a position series."""
    pos = position.fillna(0.0)
    delta_q = pos.diff().fillna(pos)
    return int((delta_q != 0.0).sum())


def sweep_ols_parameters(
    n_lags_list: list[int],
    entry_values: list[float],
    leverage_values: list[float],
    cost: float,
    fin: float,
    train_fraction: float,
    symbol: str,
    start: str | None=None,
    end: str | None=None,
) -> pd.DataFrame:
    """Explore a small grid of OLS backtest parameters.

    The sweep varies the number of lagged feature blocks and the entry
    threshold applied to predicted returns. For each combination it
    fits an OLS model, constructs prediction-based positions on the
    test window, and records key performance statistics.
    """
    panel = load_eod_panel()  # full price panel from Chapter 5
    prices = panel[symbol].astype(float)  # selecting time-series data for `symbol`

    start_ts = pd.to_datetime(start) if start is not None else None
    end_ts = pd.to_datetime(end) if end is not None else None
    if start_ts is not None or end_ts is not None:
        prices = prices.loc[start_ts:end_ts]

    records: list[dict[str, float | int]]=[]  # collect parameter results

    for lags in n_lags_list:
        features_lagged, targets = prepare_ml_dataset(
            n_lags=lags,
            symbol=symbol,
        )

        if start_ts is not None or end_ts is not None:
            features_lagged = features_lagged.loc[start_ts:end_ts]
            targets = targets.loc[start_ts:end_ts]

        features_scaled = standardise_features(features_lagged)

        n_obs = features_scaled.shape[0]
        if n_obs < 40:  # skip very short samples
            continue

        split_idx = int(train_fraction * n_obs)
        X_train = features_scaled.iloc[:split_idx].to_numpy()
        y_train = targets.iloc[:split_idx].to_numpy()
        X_test = features_scaled.iloc[split_idx:].to_numpy()
        y_test = targets.iloc[split_idx:].to_numpy()

        beta = fit_ols(X_train, y_train)
        y_hat_test = predict_ols(beta, X_test)

        idx_test = features_scaled.index[split_idx:]
        base_signal = np.sign(y_hat_test)  # directional signal

        for entry in entry_values:
            for lev in leverage_values:
                signal = pd.Series(0.0, index=features_scaled.index)
                raw_signal = base_signal.copy()
                if entry > 0.0:
                    active = np.abs(y_hat_test) > entry
                    raw_signal = np.where(active, raw_signal, 0.0)
                signal.loc[idx_test] = raw_signal

                position_ols = lev * signal.shift(1).fillna(0.0)

                pos_bh = build_positions_buy_and_hold(prices)

                net_ols = backtest_strategy(
                    prices,
                    position_ols,
                    cost_rate=cost,
                    fin_rate=fin,
                )

                test_start_ts = features_scaled.index[split_idx]
                net_ols_test = net_ols.loc[test_start_ts:]
                metrics_ols = compute_metrics(net_ols_test)
                trades_ols = count_trades(position_ols)

                records.append(
                    {
                        "lags": int(lags),
                        "entry": float(entry),
                        "leverage": float(lev),
                        "final_wealth": float(metrics_ols["final_wealth"]),
                        "ann_return": float(metrics_ols["ann_return"]),
                        "ann_vol": float(metrics_ols["ann_vol"]),
                        "sharpe": float(metrics_ols["sharpe"]),
                        "max_drawdown": float(
                            metrics_ols["max_drawdown"]
                        ),
                        "hit_ratio": float(metrics_ols["hit_ratio"]),
                        "num_trades": int(trades_ols),
                    }
                )

    results = pd.DataFrame.from_records(records)
    if not results.empty:
        results = results.sort_values(
            by=["sharpe", "ann_return"],
            ascending=[False, False],
        )
    return results


def main(
    n_lags: int=5,
    entry_threshold: float=0.002,
    leverage: float=1.0,
    cost: float=0.0001,
    fin: float=0.00001,
    train_fraction: float=0.7,
    do_sweep: bool=False,
    symbol: str="EURUSD",
    start: str | None=None,
    end: str | None=None,
) -> None:
    """Run OLS backtest and create figures and macro file.

    Parameters
    ----------
    n_lags:
        Number of daily lags of the feature panel to include in the
        regression design matrix. A value of one uses only the previous
        day's features; larger values stack multiple lagged copies side
        by side. For example, a three-lag model uses features from the
        previous three trading days when predicting the return for the
        current day.
    entry_threshold:
        Minimum absolute predicted return required before taking a
        position. Predictions with magnitude below this level lead to
        a flat position on that day.
    leverage:
        Scalar applied to the prediction-based signal before
        backtesting. A value of one corresponds to unlevered long or
        short positions; larger values scale exposure proportionally.
    cost:
        Proportional transaction-cost rate per unit turnover passed as
        ``cost_rate`` to ``backtest_strategy``. The defaults are tuned
        for a liquid FX pair such as EUR/USD.
    fin:
        Daily financing rate applied to the absolute position, passed
        as ``fin_rate`` to ``backtest_strategy``.
    train_fraction:
        Fraction of the available sample used for training, with the
        remainder reserved for out-of-sample testing.
    do_sweep:
        When ``True``, export a small parameter sweep over different
        lags and entry thresholds for inspection.
    start, end:
        Optional date strings (for example "2020-01-01") used to
        restrict the sample before splitting into training and test
        windows.
    The implementation performs a modest amount of intentional data
    snooping: it evaluates a grid of OLS configurations on the chosen
    sample and uses the combination with the best risk-adjusted
    performance as the basis for figures and LaTeX macros.
    This keeps the backtesting and evaluation code unchanged while
    producing a cleaner example equity curve.
    """
    start_ts = pd.to_datetime(start) if start is not None else None
    end_ts = pd.to_datetime(end) if end is not None else None

    # Parameter sweep for OLS configuration (data snooping).
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
        # Prefer positive Sharpe and wealth above 1.0 if available.
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

    features_lagged, targets = prepare_ml_dataset(
        n_lags=n_lags_selected,
        symbol=symbol,
    )
      # supervised data set with chosen number of lags

    if start_ts is not None or end_ts is not None:
        features_lagged = features_lagged.loc[start_ts:end_ts]
        targets = targets.loc[start_ts:end_ts]

    features_scaled = standardise_features(features_lagged)  # z-scored

    # Train/test split that respects time order
    n_obs = features_scaled.shape[0]
    split_idx = int(train_fraction * n_obs)  # training fraction

    X_train = features_scaled.iloc[:split_idx].to_numpy()
    y_train = targets.iloc[:split_idx].to_numpy()
    X_test = features_scaled.iloc[split_idx:].to_numpy()
    y_test = targets.iloc[split_idx:].to_numpy()

    beta = fit_ols(X_train, y_train)  # OLS coefficients
    y_hat_train = predict_ols(beta, X_train)  # in-sample predictions
    y_hat_test = predict_ols(beta, X_test)  # out-of-sample predictions

    r2_train = r2_score(y_train, y_hat_train)  # in-sample R^2
    r2_test = r2_score(y_test, y_hat_test)  # out-of-sample R^2
    corr_test = float(np.corrcoef(y_test, y_hat_test)[0, 1])  # test corr

    # Strategy positions based on the sign and magnitude of predictions
    idx_test = features_scaled.index[split_idx:]

    signal = pd.Series(0.0, index=features_scaled.index)  # default flat
    raw_signal = np.sign(y_hat_test)  # directional signal
    if entry_selected > 0.0:
        active = np.abs(y_hat_test) > entry_selected
        raw_signal = np.where(active, raw_signal, 0.0)
    signal.loc[idx_test] = raw_signal  # prediction-based signal on test

    # Convert prediction-based signal to positions for backtesting;
    # shifting by one day ensures that decisions made at the end of
    # day t affect returns on day t+1 and that the equity curve is
    # flat until the first non-zero position.
    position_ols = leverage_selected * signal.shift(1).fillna(0.0)
      # scaled exposure

    # Price series and buy-and-hold benchmark
    panel = load_eod_panel()  # full price panel from Chapter 5
    prices = panel[symbol].astype(float)  # selecting time-series data for `symbol`
    if start_ts is not None or end_ts is not None:
        prices = prices.loc[start_ts:end_ts]

    pos_bh = build_positions_buy_and_hold(prices)  # benchmark position

    trades_bh = count_trades(pos_bh)  # benchmark trades
    trades_ols = count_trades(position_ols)  # OLS-strategy trades

    train_start = features_scaled.index[0].date().isoformat()
    train_end = features_scaled.index[split_idx - 1].date().isoformat()
    test_start = features_scaled.index[split_idx].date().isoformat()
    test_end = features_scaled.index[-1].date().isoformat()

    net_bh = backtest_strategy(  # benchmark returns
        prices,
        pos_bh,
        cost_rate=cost,
        fin_rate=fin,
    )
    net_ols = backtest_strategy(  # strategy returns
        prices,
        position_ols,
        cost_rate=cost,
        fin_rate=fin,
    )

    # For a fair comparison in figures and tables, restrict both return
    # series to the test window where the OLS strategy is active and
    # normalise wealth from that point onwards.
    test_start_ts = features_scaled.index[split_idx]
    net_bh_test = net_bh.loc[test_start_ts:]
    net_ols_test = net_ols.loc[test_start_ts:]

    metrics_bh = compute_metrics(net_bh_test)  # benchmark metrics on test
    metrics_ols = compute_metrics(net_ols_test)  # strategy metrics on test

    write_tex_macros(
        metrics_bh,
        metrics_ols,
        trades_bh,
        trades_ols,
        r2_train,
        r2_test,
        corr_test,
        n_lags_selected,
        entry_selected,
        leverage_selected,
        cost,
        fin,
        train_fraction,
        symbol,
        train_start,
        train_end,
        test_start,
        test_end,
    )

    plot_equity_curves(net_bh_test, net_ols_test, symbol)  # equity curves

    # Print concise console summary for quick inspection.
    print(f"{symbol} OLS prediction backtest metrics (tuned parameters):")
    print("buy-and-hold:")
    pprint(metrics_bh)
    print("OLS strategy:")
    pprint(metrics_ols)
    print("OLS predictive diagnostics:")
    print(
        {
            "r2_train": r2_train,
            "r2_test": r2_test,
            "corr_test": corr_test,
        }
    )

    if do_sweep and not sweep_results.empty:
        csv_path = Path("figures/ch08_ols_parameter_sweep.csv")
        sweep_results.to_csv(csv_path, index=False)
        print("Top OLS parameter sets (by Sharpe):")
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
            "OLS-based prediction backtest with tunable parameters."
        ),
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
        "--train-fraction",
        type=float,
        default=0.7,
        help="fraction of sample used for training (default: 0.7)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="symbol to backtest (default: EURUSD)",
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
        help='optional start date, for example "2020-01-01"',
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help='optional end date, for example "2025-01-01"',
    )

    args = parser.parse_args()
    main(
        n_lags=args.lags,
        entry_threshold=args.entry,
        leverage=args.leverage,
        cost=args.cost,
        fin=args.fin,
        train_fraction=args.train_fraction,
        do_sweep=args.sweep,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
    )
