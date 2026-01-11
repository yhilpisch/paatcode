import numpy as np  # numerical arrays and statistics
import pandas as pd  # data loading and time-series handling
import matplotlib.pyplot as plt  # plotting library for equity curves

"""
Python & AI for Algorithmic Trading
Chapter 2 -- Practical Python Tests of Market Efficiency

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Extreme-day performance analysis for SPY daily returns.

This script uses the same end-of-day data set as the other Chapter 2 examples
to compare long-run performance for four scenarios:

1. Staying fully invested in SPY over the whole sample.
2. Missing the best 5, 10, and 20 daily returns (being out of the market
   whenever those days occur).
3. Avoiding the worst 5, 10, and 20 daily returns (being flat on those days).

For each case, the script computes the cumulative percentage return and creates
equity-curve plots that visualise how a small number of extreme days dominate
long-run performance. It also writes a LaTeX helper file with macros so that
the cumulative returns reported in the manuscript stay in sync with the data.
"""

plt.style.use("seaborn-v0_8")  # consistent plotting style


def load_spy_simple_returns(
    url: str="https://hilpisch.com/nov25eod.csv",
) -> pd.Series:
    """Load SPY daily simple returns from the end-of-day data set."""
    data = pd.read_csv(  # read CSV with trading dates as index
        url,
        index_col=0,
        parse_dates=True,
    )  # resulting DataFrame has one column per instrument
    data = data.sort_index()  # ensure chronological order of the index
    spy_prices = data["SPY"].astype(float)  # extract SPY closing prices

    simple_ret = spy_prices.pct_change()  # simple returns R_t
    simple_ret = simple_ret.dropna()  # drop the first NaN observation
    return simple_ret  # Pandas Series of daily simple returns


def cumulative_return(returns: pd.Series) -> float:
    """Compute cumulative simple return over the full sample."""
    gross = float((1.0 + returns).prod())  # total gross return multiplier
    return gross - 1.0  # convert to cumulative simple return


def equity_curve(returns: pd.Series) -> pd.Series:
    """Construct an equity curve from a series of simple returns."""
    return (1.0 + returns).cumprod()  # path of wealth over time


def zero_on_extreme_days(
    returns: pd.Series,
    n_days: int,
    mode: str,
) -> pd.Series:
    """Set returns to zero on the best or worst n_days of the sample.

    Parameters
    ----------
    returns:
        Series of daily simple returns indexed by date.
    n_days:
        Number of extreme days to neutralise.
    mode:
        Either \"best\" (zero the highest returns) or \"worst\" (zero the
        lowest returns).
    """
    modified = returns.copy()  # work on a copy to preserve the original

    if n_days <= 0:  # no modification requested
        return modified

    if mode == "best":  # identify the days with largest positive returns
        extreme_index = modified.nlargest(n_days).index
    elif mode == "worst":  # identify the days with most negative returns
        extreme_index = modified.nsmallest(n_days).index
    else:
        msg = "mode must be either 'best' or 'worst'"  # helpful error message
        raise ValueError(msg)  # raise an exception for invalid input

    modified.loc[extreme_index] = 0.0  # assume flat positions on those days
    return modified


def write_tex_macros(
    full_ret: float,
    miss_best: dict[int, float],
    avoid_worst: dict[int, float],
    outfile: str="figures/ch02_spy_extreme_days_stats.tex",
) -> None:
    """Write LaTeX macros with cumulative returns, wealth, and differences."""
    lines = []  # collect macro definitions as strings

    full_wealth = 1.0 + full_ret  # final wealth for full investment
    lines.append(
        "\\newcommand{\\chTwoSpyFullRet}{"
        f"{full_ret * 100.0:.1f}"
        "}\n"
    )  # full-sample cumulative return in percent
    lines.append(
        "\\newcommand{\\chTwoSpyFullWealth}{"
        f"{full_wealth:.2f}"
        "}\n"
    )  # final wealth for full investment

    miss_best_wealth = {  # final wealth for miss-best cases
        n: 1.0 + miss_best[n] for n in miss_best
    }
    avoid_worst_wealth = {  # final wealth for avoid-worst cases
        n: 1.0 + avoid_worst[n] for n in avoid_worst
    }

    miss_best_diff = {  # percentage difference vs full investment
        n: (miss_best_wealth[n] / full_wealth - 1.0) * 100.0
        for n in miss_best_wealth
    }
    avoid_worst_diff = {
        n: (avoid_worst_wealth[n] / full_wealth - 1.0) * 100.0
        for n in avoid_worst_wealth
    }

    lines.append(
        "\\newcommand{\\chTwoSpyMissBestFive}{"
        f"{miss_best[5] * 100.0:.1f}"
        "}\n"
    )  # cumulative return when missing best 5 days
    lines.append(
        "\\newcommand{\\chTwoSpyMissBestFiveWealth}{"
        f"{miss_best_wealth[5]:.2f}"
        "}\n"
    )  # final wealth when missing best 5 days
    lines.append(
        "\\newcommand{\\chTwoSpyMissBestFiveDiffPct}{"
        f"{miss_best_diff[5]:+.1f}"
        "}\n"
    )  # percentage difference vs full investment
    lines.append(
        "\\newcommand{\\chTwoSpyMissBestTen}{"
        f"{miss_best[10] * 100.0:.1f}"
        "}\n"
    )  # cumulative return when missing best 10 days
    lines.append(
        "\\newcommand{\\chTwoSpyMissBestTenWealth}{"
        f"{miss_best_wealth[10]:.2f}"
        "}\n"
    )  # final wealth when missing best 10 days
    lines.append(
        "\\newcommand{\\chTwoSpyMissBestTenDiffPct}{"
        f"{miss_best_diff[10]:+.1f}"
        "}\n"
    )  # percentage difference vs full investment
    lines.append(
        "\\newcommand{\\chTwoSpyMissBestTwenty}{"
        f"{miss_best[20] * 100.0:.1f}"
        "}\n"
    )  # cumulative return when missing best 20 days
    lines.append(
        "\\newcommand{\\chTwoSpyMissBestTwentyWealth}{"
        f"{miss_best_wealth[20]:.2f}"
        "}\n"
    )  # final wealth when missing best 20 days
    lines.append(
        "\\newcommand{\\chTwoSpyMissBestTwentyDiffPct}{"
        f"{miss_best_diff[20]:+.1f}"
        "}\n"
    )  # percentage difference vs full investment

    lines.append(
        "\\newcommand{\\chTwoSpyAvoidWorstFive}{"
        f"{avoid_worst[5] * 100.0:.1f}"
        "}\n"
    )  # cumulative return when avoiding worst 5 days
    lines.append(
        "\\newcommand{\\chTwoSpyAvoidWorstFiveWealth}{"
        f"{avoid_worst_wealth[5]:.2f}"
        "}\n"
    )  # final wealth when avoiding worst 5 days
    lines.append(
        "\\newcommand{\\chTwoSpyAvoidWorstFiveDiffPct}{"
        f"{avoid_worst_diff[5]:+.1f}"
        "}\n"
    )  # percentage difference vs full investment
    lines.append(
        "\\newcommand{\\chTwoSpyAvoidWorstTen}{"
        f"{avoid_worst[10] * 100.0:.1f}"
        "}\n"
    )  # cumulative return when avoiding worst 10 days
    lines.append(
        "\\newcommand{\\chTwoSpyAvoidWorstTenWealth}{"
        f"{avoid_worst_wealth[10]:.2f}"
        "}\n"
    )  # final wealth when avoiding worst 10 days
    lines.append(
        "\\newcommand{\\chTwoSpyAvoidWorstTenDiffPct}{"
        f"{avoid_worst_diff[10]:+.1f}"
        "}\n"
    )  # percentage difference vs full investment
    lines.append(
        "\\newcommand{\\chTwoSpyAvoidWorstTwenty}{"
        f"{avoid_worst[20] * 100.0:.1f}"
        "}\n"
    )  # cumulative return when avoiding worst 20 days
    lines.append(
        "\\newcommand{\\chTwoSpyAvoidWorstTwentyWealth}{"
        f"{avoid_worst_wealth[20]:.2f}"
        "}\n"
    )  # final wealth when avoiding worst 20 days
    lines.append(
        "\\newcommand{\\chTwoSpyAvoidWorstTwentyDiffPct}{"
        f"{avoid_worst_diff[20]:+.1f}"
        "}\n"
    )  # percentage difference vs full investment

    with open(outfile, "w", encoding="utf8") as f:  # write macros to file
        f.writelines(lines)  # save all macro lines at once


def plot_extreme_day_equity_curves(
    base_returns: pd.Series,
    miss_best: dict[int, pd.Series],
    avoid_worst: dict[int, pd.Series],
    outfile: str="figures/ch02_spy_extreme_days.pdf",
) -> None:
    """Plot equity curves for full, miss-best, and avoid-worst scenarios."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.8), sharex=True)

    equity_full = equity_curve(base_returns)  # full-investment path

    # Color maps for multiple lines in each panel (avoid grey midpoint)
    miss_ns = sorted(miss_best.keys())  # ensure deterministic ordering
    avoid_ns = sorted(avoid_worst.keys())
    full_color = plt.cm.coolwarm(0.10)  # dark blue from coolwarm map
    miss_colors = plt.cm.coolwarm(
        np.linspace(0.65, 0.95, len(miss_ns))
    )  # warmer colours for miss-best
    avoid_colors = plt.cm.coolwarm(
        np.linspace(0.05, 0.45, len(avoid_ns))
    )  # cooler colours for avoid-worst

    # Top panel: full investment vs missing the best days
    ax1.plot(
        equity_full.index,
        equity_full.values,
        label="full investment",
        color=full_color,
        lw=1.0,
    )

    for color, n_days in zip(miss_colors, miss_ns):  # curves missing best days
        eq = equity_curve(miss_best[n_days])
        label = f"missing best {n_days} days"
        ax1.plot(eq.index, eq.values, label=label, color=color, lw=1.0)

    ax1.set_ylabel("wealth (normalised)")
    ax1.set_title("SPY equity curves: impact of missing best days")
    ax1.legend(loc="upper left", fontsize="x-small")

    # Bottom panel: full investment vs avoiding the worst days
    ax2.plot(
        equity_full.index,
        equity_full.values,
        label="full investment",
        color=full_color,
        lw=1.0,
    )

    for color, n_days in zip(avoid_colors, avoid_ns):  # curves avoiding worst
        eq = equity_curve(avoid_worst[n_days])
        label = f"avoiding worst {n_days} days"
        ax2.plot(eq.index, eq.values, label=label, color=color, lw=1.0)

    ax2.set_xlabel("date")
    ax2.set_ylabel("wealth (normalised)")
    ax2.set_title("SPY equity curves: impact of avoiding worst days")
    ax2.legend(loc="upper left", fontsize="x-small")

    fig.autofmt_xdate()  # nicer x-axis labels for dates
    fig.tight_layout()  # reduce wasted space in the layout
    fig.savefig(outfile, bbox_inches="tight")  # save figure as a single PDF
    plt.close(fig)  # free Matplotlib resources


if __name__ == "__main__":
    spy_ret = load_spy_simple_returns()  # daily SPY simple returns

    full = cumulative_return(spy_ret)  # full-sample cumulative return

    miss_best_returns: dict[int, pd.Series]={}  # scenarios missing best days
    avoid_worst_returns: dict[int, pd.Series]={}  # avoiding worst days

    for n in (5, 10, 20):  # number of extreme days to neutralise
        miss_best_returns[n] = zero_on_extreme_days(spy_ret, n, mode="best")
        avoid_worst_returns[n] = zero_on_extreme_days(spy_ret, n, mode="worst")

    miss_best_perf = {  # cumulative returns for miss-best cases
        n: cumulative_return(series) for n, series in miss_best_returns.items()
    }
    avoid_worst_perf = {  # cumulative returns for avoid-worst cases
        n: cumulative_return(series) for n, series in avoid_worst_returns.items()
    }

    print("SPY extreme-day performance analysis")  # console summary
    print(f"  full investment     : {full * 100.0:6.2f}%")
    for n in (5, 10, 20):
        value = miss_best_perf[n] * 100.0  # cumulative return in percent
        print(f"  missing best {n:2d} days : {value:6.2f}%")
    for n in (5, 10, 20):
        value = avoid_worst_perf[n] * 100.0  # cumulative return in percent
        print(f"  avoiding worst {n:2d} days: {value:6.2f}%")

    write_tex_macros(full, miss_best_perf, avoid_worst_perf)  # LaTeX macros
    plot_extreme_day_equity_curves(  # equity-curve visualisation
        spy_ret,
        miss_best_returns,
        avoid_worst_returns,
    )
