import numpy as np  # numerical routines for arrays

"""
Python & AI for Algorithmic Trading
Chapter 2 -- Practical Python Tests of Market Efficiency

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Toy Granger-style predictability example for Chapter 2.

This script constructs a pair of series (X_t, Y_t) where X_t is white noise
and Y_t depends on lagged X_{t-1} plus its own noise. It reports the sample
correlation between Y_t and X_{t-1} and writes a LaTeX macro that can be
used in the manuscript to quote the current value.
"""


def simulate_toy_system(
    steps: int=500,  # number of time steps
    beta: float=0.5,  # strength of dependence on lagged X_t
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the toy system X_t, Y_t with Y_t depending on X_{t-1}."""
    rng = np.random.default_rng(seed=7)  # random number generator
    eps_x = rng.normal(scale=1.0, size=steps)  # noise driving X_t
    eps_y = rng.normal(scale=1.0, size=steps)  # noise driving Y_t

    x = eps_x  # X_t is pure white noise by construction
    y = np.empty_like(x)  # allocate array for Y_t

    y[0] = eps_y[0]  # first Y_t has no lagged X_t
    for t in range(1, steps):  # build Y_t recursively
        y[t] = beta * x[t - 1] + eps_y[t]  # dependence on lagged X_t
    return x, y  # series X_t and Y_t


def compute_lagged_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute correlation between Y_t and lagged X_{t-1}."""
    x_lag = x[:-1]  # X_{t-1} aligned with Y_t except last element
    y_trim = y[1:]  # Y_t from t=1 onward
    corr = float(np.corrcoef(x_lag, y_trim)[0, 1])  # Pearson correlation
    return corr


def write_tex_macro(
    corr: float,
    outfile: str="figures/ch02_toy_granger_stats.tex",
) -> None:
    """Write LaTeX macro with the toy correlation for inclusion in the book."""
    line = (
        "\\newcommand{\\chTwoToyLagCorr}{"
        f"{corr:.3f}"
        "}\n"
    )  # macro definition line
    with open(outfile, "w", encoding="utf8") as f:  # write macro to file
        f.write(line)  # save the single macro line


if __name__ == "__main__":
    x, y = simulate_toy_system()  # simulate X_t and Y_t
    corr = compute_lagged_correlation(x, y)  # correlation with lagged X_t

    print("Toy Granger-style example")  # console summary
    print(f"  corr(Y_t, X_(t-1)) = {corr:.3f}")

    write_tex_macro(corr)  # update LaTeX helper macro
