import numpy as np  # numerical arrays and math
import pandas as pd  # time-series and tabular data
import matplotlib.pyplot as plt  # plotting library for figures

"""
Python & AI for Algorithmic Trading
Chapter 2 -- Practical Python Tests of Market Efficiency

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

End-of-day SPY efficiency diagnostics for Chapter 2.

This script loads the nov25eod.csv end-of-day price data set, extracts
the SPY column, computes log-returns and simple summary statistics, and
generates a two-panel figure with the price path and log-return histogram.
It also writes a small LaTeX helper file with macros so that the numbers
quoted in the manuscript stay in sync with the data.
"""

plt.style.use("seaborn-v0_8")  # consistent plotting style


def load_spy_prices(
    url: str="https://hilpisch.com/nov25eod.csv",
) -> pd.Series:
    """Load SPY closing prices from the EOD panel CSV."""
    data = pd.read_csv(  # read CSV with dates as index
        url,
        index_col=0,
        parse_dates=True,
    )  # DataFrame with one column per instrument
    data = data.sort_index()  # ensure chronological order
    spy = data["SPY"].astype(float)  # extract SPY closing prices
    return spy


def compute_spy_diagnostics(
    spy_prices: pd.Series,
    max_lag: int=5,
) -> tuple[pd.Series, dict[str, float]]:
    """Compute log-returns and summary diagnostics for SPY."""
    spy_log_ret = np.log(spy_prices / spy_prices.shift(1))  # log-returns from prices
    spy_log_ret = spy_log_ret.dropna()  # drop first NaN

    n_obs = spy_log_ret.shape[0]  # number of observations
    mean_lr = float(spy_log_ret.mean())  # average log-return
    std_lr = float(spy_log_ret.std(ddof=1))  # volatility estimate
    t_stat = mean_lr / (std_lr / np.sqrt(n_obs))  # t-stat vs zero mean

    acf_lag1 = float(spy_log_ret.autocorr(lag=1))  # lag-1 autocorrelation

    diagnostics = {  # gather key diagnostics in a dictionary
        "mean": mean_lr,
        "std": std_lr,
        "t_stat": t_stat,
        "acf_1": acf_lag1,
    }
    return spy_log_ret, diagnostics


def write_tex_macros(
    diagnostics: dict[str, float],
    outfile: str="figures/ch02_spy_stats.tex",
) -> None:
    """Write LaTeX macros for SPY statistics used in the manuscript."""
    lines = []  # collect macro definitions
    lines.append(
        "\\newcommand{\\chTwoSpyMean}{"
        f"{diagnostics['mean']:.5f}"
        "}\n"
    )  # macro for mean daily log-return
    lines.append(
        "\\newcommand{\\chTwoSpyTstat}{"
        f"{diagnostics['t_stat']:.2f}"
        "}\n"
    )  # macro for t-statistic vs zero mean
    lines.append(
        "\\newcommand{\\chTwoSpyAcfOne}{"
        f"{diagnostics['acf_1']:.3f}"
        "}\n"
    )  # macro for lag-1 autocorrelation

    with open(outfile, "w", encoding="utf8") as f:  # write macros to file
        f.writelines(lines)  # save all macro lines


def plot_spy_diagnostics(
    spy_prices: pd.Series,
    spy_log_ret: pd.Series,
    outfile: str="figures/ch02_spy_diagnostics.pdf",
) -> None:
    """Plot SPY price path and log-return histogram and save as a PDF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0))  # two panels

    ax1.plot(spy_prices.index, spy_prices.values)  # SPY price path
    ax1.set_title("SPY end-of-day price")
    ax1.set_xlabel("date")
    ax1.set_ylabel("price level")

    ax2.hist(spy_log_ret.values, bins=30, density=False, alpha=0.7)  # histogram
    ax2.set_title("SPY daily log-returns")
    ax2.set_xlabel("log-return")
    ax2.set_ylabel("frequency")

    fig.autofmt_xdate()  # nicer x-axis date labels
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)


if __name__ == "__main__":
    spy_prices = load_spy_prices()  # load SPY prices from the panel CSV
    spy_log_ret, diagnostics = compute_spy_diagnostics(spy_prices)  # stats

    print("SPY efficiency diagnostics")  # console summary
    print(f"  mean log-return   = {diagnostics['mean']:.6f}")
    print(f"  std  log-return   = {diagnostics['std']:.6f}")
    print(f"  t-stat vs 0       = {diagnostics['t_stat']:.2f}")
    print(f"  lag-1 autocorr    = {diagnostics['acf_1']:.3f}")

    write_tex_macros(diagnostics)  # update LaTeX helper macros
    plot_spy_diagnostics(spy_prices, spy_log_ret)  # create diagnostic figure
