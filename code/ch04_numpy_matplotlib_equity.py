import numpy as np  # numerical arrays for simulations and returns
import matplotlib.pyplot as plt  # plotting library for figures

"""
Python & AI for Algorithmic Trading
Chapter 4 -- The Scientific Python Stack for Algorithmic Trading

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Toy NumPy and Matplotlib equity-curve figure for Chapter 4.

This script generates a small vector of synthetic daily returns, constructs
the corresponding equity curve, and plots it using Matplotlib. The figure is
saved as a PDF and included in Chapter 4 to illustrate the combination of
NumPy arrays and Matplotlib plotting in a compact example.
"""

plt.style.use("seaborn-v0_8")  # consistent plotting style for all figures


def build_equity_curve() -> tuple[np.ndarray, np.ndarray]:
    """Create a simple equity curve from a tiny return vector."""
    daily_ret = np.array(  # small sequence of daily simple returns
        [0.01, -0.005, 0.007, -0.003, 0.004],
        dtype=float,
    )
    days = np.arange(daily_ret.shape[0])  # integer time index for plotting
    equity = (1.0 + daily_ret).cumprod()  # equity curve from simple returns
    return days, equity  # x-axis index and corresponding equity values


def plot_equity_curve(
    days: np.ndarray,
    equity: np.ndarray,
    outfile: str="figures/ch04_numpy_matplotlib_equity.pdf",
) -> None:
    """Plot the toy equity curve and save it as a PDF."""
    fig, ax = plt.subplots(figsize=(6.0, 3.0))  # single-axis figure

    ax.plot(days, equity, color="C0", lw=1.0)  # equity curve line
    ax.set_xlabel("time step")  # x-axis label
    ax.set_ylabel("wealth (normalised)")  # y-axis label
    ax.set_title("Toy equity curve from NumPy returns")  # figure title

    ax.grid(True, alpha=0.3)  # light grid for readability
    fig.tight_layout()  # minimise wasted space around axes
    fig.savefig(outfile, bbox_inches="tight")  # save figure as PDF
    plt.close(fig)  # free Matplotlib resources


if __name__ == "__main__":
    x_days, y_equity = build_equity_curve()  # construct toy example
    plot_equity_curve(x_days, y_equity)  # create and save the figure
