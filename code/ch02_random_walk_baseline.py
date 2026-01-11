import numpy as np  # numerical library for arrays and random numbers
import matplotlib.pyplot as plt  # plotting library for figures

"""
Python & AI for Algorithmic Trading
Chapter 2 -- Practical Python Tests of Market Efficiency

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Random-walk baseline simulation and diagnostics for Chapter 2.

This script simulates a simple log-price random walk under an EMH-style
assumption of unpredictable shocks, computes basic summary statistics and
autocorrelations for the log-returns, and produces a compact diagnostic
figure. It also writes a small LaTeX helper file with macros used in the
manuscript so that numeric results stay in sync with the code.
"""

plt.style.use("seaborn-v0_8")  # consistent plotting style


def simulate_log_random_walk(
    steps: int=252 * 5,  # number of trading days
    mu: float=0.0,  # drift of the log-returns per step
    sigma: float=0.01,  # volatility of the log-returns per step
    s0: float=100.0,  # starting price level
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a log-price random walk and corresponding log-returns."""
    rng = np.random.default_rng(seed=42)  # random number generator
    eps = rng.normal(mu, sigma, size=steps)  # log-return shocks ε_t
    log_price0 = np.log(s0)  # starting log-price
    log_prices = log_price0 + eps.cumsum()  # log-price path over time
    prices = np.exp(log_prices)  # price levels from log-prices
    return prices, eps  # simulated prices and log-returns


def compute_diagnostics(
    log_returns: np.ndarray, max_lag: int=5
) -> dict[str, float]:
    """Compute mean, standard deviation, t-statistic, and short acf."""
    n_obs = log_returns.shape[0]  # number of observations
    mean_lr = float(log_returns.mean())  # sample mean
    std_lr = float(log_returns.std(ddof=1))  # sample standard deviation
    t_stat = mean_lr / (std_lr / np.sqrt(n_obs))  # t-statistic vs zero mean

    acf = []  # list to collect autocorrelations
    for lag in range(1, max_lag + 1):  # lags 1, 2, ..., max_lag
        x = log_returns[:-lag]  # aligned series without last lag entries
        y = log_returns[lag:]  # series shifted by the lag
        num = float(np.dot(x - x.mean(), y - y.mean()))  # covariance-like term
        den = float(np.dot(log_returns - log_returns.mean(),
                           log_returns - log_returns.mean()))  # variance term
        acf.append(num / den)  # lag-k autocorrelation estimate

    diagnostics = {  # collect results in a dictionary
        "mean": mean_lr,
        "std": std_lr,
        "t_stat": t_stat,
    }
    for idx, value in enumerate(acf, start=1):  # enumerate lags
        diagnostics[f"acf_{idx}"] = float(value)  # store acf_k under key
    return diagnostics


def write_tex_macros(
    diagnostics: dict[str, float],
    outfile: str="figures/ch02_random_walk_stats.tex",
) -> None:
    """Write LaTeX macros with summary statistics for inclusion in the book."""
    lines = []  # collect macro definitions as strings
    lines.append(
        "\\newcommand{\\chTwoRWMean}{"
        f"{diagnostics['mean']:.5f}"
        "}\n"
    )  # macro for mean log-return
    lines.append(
        "\\newcommand{\\chTwoRWTstat}{"
        f"{diagnostics['t_stat']:.2f}"
        "}\n"
    )  # macro for t-statistic vs zero
    lines.append(
        "\\newcommand{\\chTwoRWAcfOne}{"
        f"{diagnostics['acf_1']:.3f}"
        "}\n"
    )  # macro for lag-1 autocorrelation

    with open(outfile, "w", encoding="utf8") as f:  # write macros to file
        f.writelines(lines)  # save all macro lines at once


def plot_random_walk_diagnostics(
    prices: np.ndarray,
    log_returns: np.ndarray,
    outfile: str="figures/ch02_random_walk_diagnostics.pdf",
) -> None:
    """Plot price path and histogram of log-returns and save as a PDF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0))  # two subplots

    ax1.plot(prices)  # simulated price path
    ax1.set_title("Random walk price path")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("price level")

    ax2.hist(log_returns, bins=30, density=False, alpha=0.7)  # histogram
    ax2.set_title("Histogram of log-returns")
    ax2.set_xlabel("log-return")
    ax2.set_ylabel("frequency")

    fig.tight_layout()  # reduce wasted space
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # free resources


if __name__ == "__main__":
    prices, log_returns = simulate_log_random_walk()  # simulate baseline
    diagnostics = compute_diagnostics(log_returns)  # compute summary numbers

    print("Random-walk baseline diagnostics")  # short console summary
    print(f"  mean log-return   = {diagnostics['mean']:.6f}")
    print(f"  std  log-return   = {diagnostics['std']:.6f}")
    print(f"  t-stat vs 0       = {diagnostics['t_stat']:.2f}")
    print(f"  lag-1 autocorr    = {diagnostics['acf_1']:.3f}")

    write_tex_macros(diagnostics)  # update LaTeX helper macros
    plot_random_walk_diagnostics(prices, log_returns)  # create diagnostic plot
