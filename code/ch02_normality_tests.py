import numpy as np  # numerical arrays and statistics
import pandas as pd  # data loading and time-series handling
import matplotlib.pyplot as plt  # plotting library for figures
from scipy import stats  # statistical moments and normality tests

"""
Python & AI for Algorithmic Trading
Chapter 2 -- Practical Python Tests of Market Efficiency

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Normality diagnostics for simulated and empirical daily log-returns.

This script compares a simple random-walk benchmark to daily SPY log-returns
from the end-of-day data set used in Chapter 2. For both series it computes
sample skewness, excess kurtosis, and the Jarque–Bera statistic and creates
histograms with normal probability density overlays and Q–Q style plots.

The script also writes a small LaTeX helper file with macros so that the
numeric results reported in the manuscript stay in sync with the code.
"""

plt.style.use("seaborn-v0_8")  # consistent plotting style


def simulate_random_walk_log_returns(
    steps: int=252 * 5,  # number of trading days (about five years)
    mu: float=0.0,  # daily drift under the random-walk baseline
    sigma: float=0.01,  # daily volatility of the log-returns
) -> np.ndarray:
    """Simulate i.i.d. normal log-returns for a geometric random walk."""
    rng = np.random.default_rng(seed=123)  # random number generator
    eps = rng.normal(mu, sigma, size=steps)  # simulated log-returns
    return eps  # array of simulated daily log-returns


def load_spy_log_returns(
    url: str="https://hilpisch.com/nov25eod.csv",
) -> pd.Series:
    """Load SPY daily log-returns from the end-of-day data set."""
    data = pd.read_csv(  # read CSV with date index and price columns
        url,
        index_col=0,
        parse_dates=True,
    )  # resulting DataFrame has one column per instrument
    data = data.sort_index()  # ensure chronological order of the index
    spy_prices = data["SPY"].astype(float)  # extract SPY closing prices

    log_ret = np.log(spy_prices / spy_prices.shift(1))  # log-returns from prices
    log_ret = log_ret.dropna()  # drop the first NaN observation
    return log_ret  # Pandas Series with DatetimeIndex


def compute_moments_and_jb(
    returns: np.ndarray,
) -> dict[str, float]:
    """Compute skewness, excess kurtosis, and Jarque–Bera statistic."""
    n_obs = returns.shape[0]  # number of observations
    mean_lr = float(returns.mean())  # sample mean of returns
    std_lr = float(returns.std(ddof=1))  # sample standard deviation

    skew = float(stats.skew(returns, bias=False))  # sample skewness
    kurt_excess = float(  # sample excess kurtosis (kurtosis minus 3)
        stats.kurtosis(returns, fisher=True, bias=False)
    )

    jb_stat, jb_pvalue = stats.jarque_bera(returns)  # Jarque–Bera test

    return {
        "n_obs": float(n_obs),  # cast to float for consistent formatting
        "mean": mean_lr,
        "std": std_lr,
        "skew": skew,
        "kurt_excess": kurt_excess,
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pvalue),
    }


def write_tex_macros(
    sim_stats: dict[str, float],
    spy_stats: dict[str, float],
    outfile: str="figures/ch02_normality_stats.tex",
) -> None:
    """Write LaTeX macros for normality diagnostics used in Chapter 2."""
    lines = []  # collect macro definitions as strings

    lines.append(
        "\\newcommand{\\chTwoSimSkew}{"
        f"{sim_stats['skew']:.3f}"
        "}\n"
    )  # skewness of simulated returns
    lines.append(
        "\\newcommand{\\chTwoSimKurt}{"
        f"{sim_stats['kurt_excess']:.3f}"
        "}\n"
    )  # excess kurtosis of simulated returns
    sim_jb = f"{sim_stats['jb_stat']:,.2f}".replace(",", "\\,")
    lines.append(
        "\\newcommand{\\chTwoSimJB}{"
        f"{sim_jb}"
        "}\n"
    )  # Jarque--Bera statistic for simulated returns

    lines.append(
        "\\newcommand{\\chTwoSpySkew}{"
        f"{spy_stats['skew']:.3f}"
        "}\n"
    )  # skewness of SPY returns
    lines.append(
        "\\newcommand{\\chTwoSpyKurt}{"
        f"{spy_stats['kurt_excess']:.3f}"
        "}\n"
    )  # excess kurtosis of SPY returns
    spy_jb = f"{spy_stats['jb_stat']:,.2f}".replace(",", "\\,")
    lines.append(
        "\\newcommand{\\chTwoSpyJB}{"
        f"{spy_jb}"
        "}\n"
    )  # Jarque--Bera statistic for SPY returns

    with open(outfile, "w", encoding="utf8") as f:  # write macros to file
        f.writelines(lines)  # save all macro lines at once


def _hist_with_normal_overlay(
    ax: plt.Axes,
    returns: np.ndarray,
    title: str,
) -> None:
    """Plot histogram of returns with normal density overlay."""
    mean_lr = float(returns.mean())  # sample mean of returns
    std_lr = float(returns.std(ddof=1))  # sample standard deviation

    ax.hist(  # histogram of returns scaled to a density
        returns,
        bins=40,
        density=True,
        alpha=0.7,
        color="C0",
    )

    grid = np.linspace(  # grid of points for the normal density
        mean_lr - 4.0 * std_lr,
        mean_lr + 4.0 * std_lr,
        200,
    )
    pdf = stats.norm.pdf(grid, loc=mean_lr, scale=std_lr)  # normal density

    ax.plot(grid, pdf, "C3", lw=2.0)  # overlay normal curve
    ax.set_title(title)
    ax.set_xlabel("log-return")
    ax.set_ylabel("frequency")


def _qq_plot_standardised(
    ax: plt.Axes,
    returns: np.ndarray,
    title: str,
) -> None:
    """Create a Q–Q style plot against the standard normal distribution."""
    mean_lr = float(returns.mean())  # sample mean
    std_lr = float(returns.std(ddof=1))  # sample standard deviation
    standardised = (returns - mean_lr) / std_lr  # z-scores of returns

    osm, osr = stats.probplot(  # theoretical and empirical quantiles
        standardised,
        dist="norm",
    )[0]

    ax.scatter(osm, osr, s=10, alpha=0.6, color="C0")  # Q–Q points

    lower = min(osm.min(), osr.min())  # lower bound for 45-degree line
    upper = max(osm.max(), osr.max())  # upper bound for 45-degree line
    ax.plot([lower, upper], [lower, upper], "C3--", lw=1.5)  # reference line

    ax.set_title(title)
    ax.set_xlabel("theoretical quantiles (normal)")
    ax.set_ylabel("sample quantiles")


def plot_normality_diagnostics(
    sim_returns: np.ndarray,
    spy_returns: pd.Series,
    outfile: str="figures/ch02_normality_diagnostics.pdf",
) -> None:
    """Create side-by-side normality diagnostics for both series."""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.0))  # 2x2 grid of plots

    _hist_with_normal_overlay(
        axes[0, 0],
        sim_returns,
        "Simulated log-returns: histogram vs normal",
    )
    _qq_plot_standardised(
        axes[0, 1],
        sim_returns,
        "Simulated log-returns: Q–Q against normal",
    )

    _hist_with_normal_overlay(
        axes[1, 0],
        spy_returns.values,
        "SPY log-returns: histogram vs normal",
    )
    _qq_plot_standardised(
        axes[1, 1],
        spy_returns.values,
        "SPY log-returns: Q–Q against normal",
    )

    fig.tight_layout()  # reduce wasted space in the figure layout
    fig.savefig(outfile, bbox_inches="tight")  # save plots as a single PDF
    plt.close(fig)  # free Matplotlib resources


if __name__ == "__main__":
    sim_returns = simulate_random_walk_log_returns()  # simulated benchmark
    spy_log_ret = load_spy_log_returns()  # empirical SPY log-returns

    sim_stats = compute_moments_and_jb(sim_returns)  # simulated diagnostics
    spy_stats = compute_moments_and_jb(spy_log_ret.values)  # SPY diagnostics

    print("Normality diagnostics for daily log-returns")  # console summary
    print("Simulated random-walk returns")
    print(f"  skewness          = {sim_stats['skew']:.3f}")
    print(f"  excess kurtosis   = {sim_stats['kurt_excess']:.3f}")
    print(f"  Jarque-Bera stat  = {sim_stats['jb_stat']:.2f}")
    print("SPY daily log-returns")
    print(f"  skewness          = {spy_stats['skew']:.3f}")
    print(f"  excess kurtosis   = {spy_stats['kurt_excess']:.3f}")
    print(f"  Jarque-Bera stat  = {spy_stats['jb_stat']:.2f}")

    write_tex_macros(sim_stats, spy_stats)  # update LaTeX helper macros
    plot_normality_diagnostics(sim_returns, spy_log_ret)  # create figure
