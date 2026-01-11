from dataclasses import dataclass  # configuration container
from pathlib import Path  # filesystem paths for figures and stats

import numpy as np  # numerical arrays and random numbers
import pandas as pd  # tabular containers for simulation output
import matplotlib.pyplot as plt  # plotting library for figures

"""
Python & AI for Algorithmic Trading
Chapter 21 -- EMH in the Age of AI and Retail Algotrading

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Monte Carlo experiment for EMH-style baselines with many strategies.

The idea is to simulate a large number of ``random`` trading rules
that, by construction, have no true edge. Even so, a fraction of them
will display attractive Sharpe ratios over a finite sample purely by
chance. This connects the Efficient Market Hypothesis to multiple-
testing concerns in AI-assisted research workflows.
"""

plt.style.use("seaborn-v0_8")  # shared plotting style


@dataclass
class EmhSimulationConfig:
    """Configuration for the EMH-style Monte Carlo experiment."""

    n_strategies: int=5_000
    n_days: int=750
    mu: float=0.0
    sigma: float=0.02
    seed: int=21


def simulate_random_sharpes(cfg: EmhSimulationConfig) -> np.ndarray:
    """Simulate Sharpe ratios for many random strategies."""
    rng = np.random.default_rng(seed=cfg.seed)
    rets = rng.normal(
        loc=cfg.mu,
        scale=cfg.sigma,
        size=(cfg.n_strategies, cfg.n_days),
    )

    mean = rets.mean(axis=1)
    vol = rets.std(axis=1, ddof=1)
    sharpe_daily = np.divide(
        mean,
        vol,
        out=np.zeros_like(mean),
        where=vol > 0.0,
    )
    sharpe_ann = sharpe_daily * np.sqrt(252.0)
    return sharpe_ann


def sharpe_summary(sharpes: np.ndarray) -> pd.Series:
    """Turn Sharpe samples into a small summary table."""
    s = pd.Series(sharpes)
    return pd.Series(
        {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }
    )


def plot_sharpe_histogram(
    sharpes: np.ndarray,
    path: Path,
) -> Path:
    """Plot a histogram of simulated Sharpe ratios."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.hist(
        sharpes,
        bins=40,
        color="tab:blue",
        alpha=0.7,
        edgecolor="white",
    )
    ax.set_xlabel("annualised Sharpe ratio")
    ax.set_ylabel("number of strategies")
    ax.set_title("Random-strategy Sharpe distribution under EMH baseline")
    ax.grid(alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Run the EMH-style Monte Carlo and print summary stats."""
    cfg = EmhSimulationConfig()
    sharpes = simulate_random_sharpes(cfg)
    summary = sharpe_summary(sharpes)

    print("EMH-style Monte Carlo for random strategies:")
    for key, value in summary.items():
        print(f"  {key}: {value: .3f}")

    fig_path = Path("figures") / "ch21_random_sharpes_hist.pdf"
    plot_sharpe_histogram(sharpes, fig_path)
    print(f"\nHistogram written to {fig_path}")


if __name__ == "__main__":
    main()

