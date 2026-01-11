from pathlib import Path  # filesystem paths for optional local data

import numpy as np  # numerical arrays and statistics
import pandas as pd  # data loading and time-series handling
import matplotlib.pyplot as plt  # plotting library for figures
from scipy import stats  # normal densities for reference curves

"""
Python & AI for Algorithmic Trading
Chapter 5 -- Static EOD Data: CSV Baseline and Data Engineering

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Static end-of-day data engineering utilities and figures for Chapter 5.

This script turns the baseline end-of-day price data set into a cleaned
multi-asset panel, computes return-based statistics, builds a small feature
panel for one representative series, and generates figures and LaTeX helper
macros used throughout Chapter 5.

The functions are written to be reusable from interactive sessions and from
the LaTeX examples. Running the script as a program refreshes all figures and
the macro file in one go.
"""

plt.style.use("seaborn-v0_8")  # consistent plotting style across figures


def load_eod_panel(
    url: str="https://hilpisch.com/nov25eod.csv",
    local_path: str="data/nov25eod.csv",
) -> pd.DataFrame:
    """Load the baseline end-of-day price panel as a DataFrame.

    The CSV file contains a first column of trading dates and one column per
    instrument. The function parses the dates, sorts the index, and returns a
    DataFrame of closing prices with a DatetimeIndex.
    """
    path = Path(local_path)  # potential local copy of the data set
    if path.is_file():  # prefer local file if it exists
        data = pd.read_csv(  # read CSV from local path
            path,
            index_col=0,
            parse_dates=True,
        )
    else:
        data = pd.read_csv(  # read CSV with dates as index from URL
            url,
            index_col=0,
            parse_dates=True,
        )  # resulting DataFrame has one column per instrument
    data = data.sort_index()  # ensure chronological order of the index
    return data.astype(float)  # standardise to float for downstream code


def load_spy_log_returns() -> pd.Series:
    """Return daily log-returns for one representative series (SPY)."""
    panel = load_eod_panel()  # load the full price panel
    prices = panel["SPY"]  # closing prices for the chosen series
    log_ret = np.log(prices / prices.shift(1))  # log-returns from prices
    return log_ret.dropna()  # drop initial NaN


def describe_panel(panel: pd.DataFrame) -> dict[str, float | int | str]:
    """Compute basic descriptive statistics for the price panel."""
    n_obs = int(panel.shape[0])  # number of trading days
    n_assets = int(panel.shape[1])  # number of instruments
    start_date = panel.index.min().date().isoformat()  # first trading date
    end_date = panel.index.max().date().isoformat()  # last trading date

    missing_by_col = panel.isna().sum()  # missing prices per instrument
    max_missing = int(missing_by_col.max())  # worst-case missing count

    return {
        "n_obs": n_obs,
        "n_assets": n_assets,
        "start_date": start_date,
        "end_date": end_date,
        "max_missing": max_missing,
    }


def summarise_return_scales(log_ret: pd.Series) -> dict[str, float]:
    """Summarise daily, weekly, and monthly log-return volatilities."""
    daily = log_ret  # already cleaned daily log-returns
    weekly = daily.resample("W-FRI").sum()  # aggregated weekly log-returns
    monthly = daily.resample("ME").sum()  # aggregated month-end log-returns

    daily_vol = float(daily.std(ddof=1))  # daily volatility estimate
    weekly_vol = float(weekly.std(ddof=1))  # weekly volatility estimate
    monthly_vol = float(monthly.std(ddof=1))  # monthly volatility estimate

    return {
        "daily_vol": daily_vol,
        "weekly_vol": weekly_vol,
        "monthly_vol": monthly_vol,
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
    }


def build_feature_panel(
    log_ret: pd.Series,
    window_short: int=20,
    window_long: int=60,
) -> pd.DataFrame:
    """Construct a small daily feature panel from log-returns."""
    features = pd.DataFrame(index=log_ret.index)  # aligned feature frame

    features["log_return"] = log_ret  # raw daily log-return
    features["ma_short"] = log_ret.rolling(window_short).mean()  # trend
    features["vol_short"] = log_ret.rolling(window_short).std(ddof=1)  # risk
    features["momentum_long"] = log_ret.rolling(window_long).sum()  # momentum

    deviation = features["log_return"] - features["ma_short"]  # surprise term
    features["z_score"] = deviation / features["vol_short"]  # standardised move

    return features.dropna()  # drop initial rows with incomplete windows


def count_large_moves(
    log_ret: pd.Series,
    threshold: float=0.045,
) -> int:
    """Count days with large absolute log-returns."""
    mask = np.abs(log_ret) > threshold  # boolean mask for large moves
    return int(mask.sum())  # number of extreme-move days


def format_float(value: float, decimals: int=4) -> str:
    """Format a floating-point number with fixed decimals."""
    fmt = f"{{:.{decimals}f}}"  # template for decimal formatting
    return fmt.format(value)  # formatted string, no thousands separator


def write_tex_macros(
    panel_stats: dict[str, float | int | str],
    scale_stats: dict[str, float | pd.Series],
    feature_panel: pd.DataFrame,
    n_large_moves: int,
    outfile: str="figures/ch05_eod_stats.tex",
) -> None:
    """Write LaTeX macros with Chapter 5 statistics."""
    lines: list[str]=[]  # collect macro definition strings

    lines.append(
        "\\newcommand{\\chFiveNumObs}{"
        f"{panel_stats['n_obs']}"
        "}\n"
    )  # number of trading days
    lines.append(
        "\\newcommand{\\chFiveStartDate}{"
        f"{panel_stats['start_date']}"
        "}\n"
    )  # first trading date in ISO format
    lines.append(
        "\\newcommand{\\chFiveEndDate}{"
        f"{panel_stats['end_date']}"
        "}\n"
    )  # last trading date in ISO format
    lines.append(
        "\\newcommand{\\chFiveNumAssets}{"
        f"{panel_stats['n_assets']}"
        "}\n"
    )  # number of instruments in the panel
    lines.append(
        "\\newcommand{\\chFiveMaxMissing}{"
        f"{panel_stats['max_missing']}"
        "}\n"
    )  # maximum number of missing prices per column

    lines.append(
        "\\newcommand{\\chFiveSpyDailyVol}{"
        f"{format_float(scale_stats['daily_vol'])}"
        "}\n"
    )  # daily log-return volatility
    lines.append(
        "\\newcommand{\\chFiveSpyWeeklyVol}{"
        f"{format_float(scale_stats['weekly_vol'])}"
        "}\n"
    )  # weekly log-return volatility
    lines.append(
        "\\newcommand{\\chFiveSpyMonthlyVol}{"
        f"{format_float(scale_stats['monthly_vol'])}"
        "}\n"
    )  # monthly log-return volatility

    lines.append(
        "\\newcommand{\\chFiveNumFeatureRows}{"
        f"{feature_panel.shape[0]}"
        "}\n"
    )  # number of usable feature rows

    lines.append(
        "\\newcommand{\\chFiveNumLargeMoves}{"
        f"{n_large_moves}"
        "}\n"
    )  # number of days with large absolute log-returns

    with open(outfile, "w", encoding="utf8") as f:  # write macros to file
        f.writelines(lines)  # save all macro lines at once


def plot_eod_panel_overview(
    panel: pd.DataFrame,
    outfile: str="figures/ch05_eod_panel_overview.pdf",
) -> None:
    """Plot normalised prices for a selection of instruments."""
    cols = ["SPY", "AAPL", "GLD"]  # representative subset of series
    available = [c for c in cols if c in panel.columns]  # keep existing ones

    data = panel[available].copy()  # subset for plotting
    data = data.ffill()  # forward-fill small gaps in prices

    normed = data / data.iloc[0]  # normalise all series to start at 1.0

    fig, ax = plt.subplots(figsize=(7.0, 3.2))  # single axes for all lines

    cmap = plt.cm.coolwarm  # colour map for line colours
    colour_lookup = {  # avoid the grey midpoint of the map
        "SPY": cmap(0.10),
        "AAPL": cmap(0.30),
        "GLD": cmap(0.90),
    }

    for col in available:  # plot each normalised series
        color = colour_lookup.get(col, cmap(0.50))  # fallback colour
        ax.plot(
            normed.index,
            normed[col].values,
            label=col,
            color=color,
            lw=1.0,
        )

    ax.set_ylabel("normalised price")  # y-axis label
    ax.set_xlabel("date")  # x-axis label
    ax.legend(loc="upper left")  # legend for instrument names
    ax.grid(True, alpha=0.3)  # light grid for readability

    fig.tight_layout()  # reduce unused margins
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # free Matplotlib resources


def plot_return_scales(
    daily: pd.Series,
    weekly: pd.Series,
    monthly: pd.Series,
    outfile: str="figures/ch05_returns_resampling.pdf",
) -> None:
    """Plot histograms and normal densities for different horizons."""
    fig, (ax_hist, ax_pdf) = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.2),
        sharey=True,
    )  # side-by-side panels

    series = [daily, weekly, monthly]  # list of Series objects
    labels = ["daily", "weekly", "monthly"]  # labels for horizons
    cmap = plt.cm.coolwarm  # colour map shared across figures
    colors = [cmap(0.01), cmap(0.15), cmap(0.95)]  # distinct horizon colours

    # Left panel: histograms at different horizons
    for data, label, color in zip(series, labels, colors):
        ax_hist.hist(
            data.values,
            bins=40,
            density=True,
            alpha=0.45,
            label=label,
            color=color,
        )

    ax_hist.set_xlabel("log-return")  # x-axis label
    ax_hist.set_ylabel("frequency")  # y-axis label
    ax_hist.legend(loc="upper left")  # legend for horizons

    # Right panel: fitted normal densities with matching mean and std
    xmin = min(s.min() for s in series)  # smallest observed return
    xmax = max(s.max() for s in series)  # largest observed return
    x_grid = np.linspace(xmin, xmax, 400)  # grid for density curves

    for data, label, color in zip(series, labels, colors):
        mu = float(data.mean())  # sample mean for this horizon
        sigma = float(data.std(ddof=1))  # sample standard deviation
        y_grid = stats.norm.pdf(x_grid, loc=mu, scale=sigma)  # normal pdf
        ax_pdf.plot(x_grid, y_grid, label=label, color=color, lw=1.0)

    ax_pdf.set_xlabel("log-return")  # x-axis label for density panel
    ax_pdf.set_ylabel("normal density")  # y-axis label
    ax_pdf.legend(loc="upper left")  # legend for fitted curves
    ax_pdf.grid(True, alpha=0.3)  # light grid for clarity

    fig.tight_layout()  # reduce padding between subplots
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # close figure to free resources


def plot_feature_panel(
    prices: pd.Series,
    features: pd.DataFrame,
    outfile: str="figures/ch05_spy_feature_panel.pdf",
    symbol: str="SPY",
) -> None:
    """Plot price, log-return, volatility, and z-score diagnostics."""
    fig, (ax_price, ax_ret, ax_vol, ax_z) = plt.subplots(
        4,
        1,
        figsize=(7.0, 6.0),
        sharex=True,
    )  # stacked panels sharing the date axis

    normed_price = prices / prices.iloc[0]  # normalise to start at 1.0

    # Top panel: normalised price series
    ax_price.plot(
        normed_price.index,
        normed_price.values,
        color=plt.cm.coolwarm(0.15),
        lw=1.0,
        label="normalised price",
    )
    title = f"{symbol}: price, returns, volatility, z-score"  # figure title
    ax_price.set_title(title)  # add symbol-specific title
    ax_price.set_ylabel("price (normalised)")  # y-axis label
    ax_price.legend(loc="upper left")  # legend for the price line
    ax_price.grid(True, alpha=0.3)  # light grid

    # Middle panel: log-return with short moving average
    ax_ret.plot(
        features.index,
        features["log_return"].values,
        color=plt.cm.coolwarm(0.80),
        lw=1.0,
        label="log-return",
    )
    ax_ret.plot(
        features.index,
        features["ma_short"].values,
        color=plt.cm.coolwarm(0.25),
        lw=1.0,
        label="20-day moving average",
    )
    ax_ret.set_ylabel("log-return")  # y-axis label
    ax_ret.legend(loc="upper left")  # legend for return lines
    ax_ret.grid(True, alpha=0.3)  # light grid

    # Third panel: rolling 20-day volatility
    ax_vol.plot(
        features.index,
        features["vol_short"].values,
        color=plt.cm.coolwarm(0.30),
        lw=1.0,
        label="20-day volatility",
    )
    ax_vol.set_ylabel("20-day volatility")  # y-axis label
    ax_vol.legend(loc="upper left")  # legend for volatility
    ax_vol.grid(True, alpha=0.3)  # light grid

    # Bottom panel: z-score time series
    ax_z.plot(
        features.index,
        features["z_score"].values,
        color=plt.cm.coolwarm(0.90),
        lw=1.0,
        label="z-score",
    )
    ax_z.axhline(0.0, color="black", lw=0.8, alpha=0.6)  # zero line
    ax_z.set_ylabel("z-score")  # y-axis label
    ax_z.set_xlabel("date")  # x-axis label
    ax_z.legend(loc="upper left")  # legend for z-score panel
    ax_z.grid(True, alpha=0.3)  # light grid

    fig.tight_layout()  # remove excess whitespace
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # free Matplotlib resources


def plot_missing_and_extremes(
    panel: pd.DataFrame,
    log_ret: pd.Series,
    threshold: float=0.045,
    outfile: str="figures/ch05_missing_and_extremes.pdf",
) -> None:
    """Visualise missing values and large absolute returns."""
    fig, (ax_missing, ax_moves) = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.6),
        sharex=False,
    )  # stacked panels for diagnostics

    # Top panel: missing-value heatmap
    missing = panel.isna()  # boolean DataFrame of missing prices
    img = ax_missing.imshow(
        missing.T,
        aspect="auto",
        interpolation="nearest",
        cmap=plt.cm.coolwarm,
    )  # instruments on y-axis, time on x-axis

    ax_missing.set_yticks(range(panel.shape[1]))  # ticks for instruments
    ax_missing.set_yticklabels(panel.columns)  # label with column names
    ax_missing.set_xlabel("time index")  # generic x-axis label
    ax_missing.set_title("missing-price pattern across instruments")  # title

    fig.colorbar(img, ax=ax_missing, fraction=0.046, pad=0.02)  # legend

    # Bottom panel: log-returns with markers for large moves
    ax_moves.plot(
        log_ret.index,
        log_ret.values,
        color=plt.cm.coolwarm(0.20),
        lw=1.0,
        label="daily log-return",
    )

    large_mask = np.abs(log_ret) > threshold  # boolean mask for extremes
    ax_moves.scatter(
        log_ret.index[large_mask],
        log_ret.values[large_mask],
        color=plt.cm.coolwarm(0.85),
        s=12.0,
        label=f"|return| > {threshold*100:.1f}%",
    )

    ax_moves.axhline(0.0, color="black", lw=0.8, alpha=0.6)  # zero line
    ax_moves.set_ylabel("log-return")  # y-axis label
    ax_moves.set_xlabel("date")  # x-axis label
    ax_moves.legend(loc="upper left")  # legend describing markers
    ax_moves.grid(True, alpha=0.3)  # light grid

    fig.tight_layout()  # adjust layout for readability
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # free Matplotlib resources


def main() -> None:
    """Run the full Chapter 5 data-engineering workflow."""
    panel = load_eod_panel()  # price panel
    log_ret = load_spy_log_returns()  # daily log-returns for one series

    panel_stats = describe_panel(panel)  # basic size and coverage stats
    scale_stats = summarise_return_scales(log_ret)  # volatility by horizon

    feature_panel = build_feature_panel(log_ret)  # feature engineering
    n_large_moves = count_large_moves(log_ret)  # large-move count

    write_tex_macros(  # keep manuscript numbers in sync with data
        panel_stats=panel_stats,
        scale_stats=scale_stats,
        feature_panel=feature_panel,
        n_large_moves=n_large_moves,
    )

    # Generate figures used in the chapter.
    plot_eod_panel_overview(panel=panel)
    plot_return_scales(
        daily=scale_stats["daily"],
        weekly=scale_stats["weekly"],
        monthly=scale_stats["monthly"],
    )
    # Price series aligned with the feature panel index
    aligned_prices = panel["SPY"].reindex(feature_panel.index).ffill()
    plot_feature_panel(
        prices=aligned_prices,
        features=feature_panel,
        symbol="SPY",
    )
    plot_missing_and_extremes(panel=panel, log_ret=log_ret)


if __name__ == "__main__":
    main()  # execute workflow when script is run as a program
