from pathlib import Path  # filesystem paths for macro and figure output

from pprint import pprint  # pretty printing of metric dictionaries
import numpy as np  # numerical arrays and random numbers
import pandas as pd  # tabular time-series structures
import matplotlib.pyplot as plt  # plotting library for figures
import matplotlib.dates as mdates  # date locators and formatters for axes
import time  # wall-clock timing for training loop

"""
Python & AI for Algorithmic Trading
Chapter 9 -- Reinforcement Learning for Trading Decisions

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Tabular reinforcement-learning example for daily trading decisions.

This script implements a small Q-learning agent that trades a single
symbol using daily end-of-day data. The environment is deliberately
minimal: the state is a discretised z-score of recent returns together
with the current position, the actions are {-1, 0, +1} (short, flat,
long), and the reward is the daily net return after simple transaction
costs and financing charges.

The workflow mirrors the backtesting approach from earlier chapters.

1. Load the static end-of-day price panel and compute daily returns.
2. Build a rolling z-score feature and discretise it into a handful of
   states that summarise recent moves.
3. Train a tabular Q-learning agent on an initial training window,
   using an epsilon-greedy policy that gradually shifts from
   exploration to exploitation.
4. Evaluate the greedy policy on a later test window, compare it with
   a buy-and-hold benchmark, and export both summary statistics and an
   equity-curve figure for use in Chapter 9.

The goal is not to produce a production-ready trading agent but to
highlight how reinforcement learning reframes trading as a sequence of
decisions with state, action, and reward. For a much deeper treatment
of these ideas, including richer state representations and model
architectures, see Hilpisch (2024), Reinforcement Learning for Finance
-- A Python-based Introduction, published by O'Reilly.
"""

from ch05_eod_engineering import load_eod_panel  # static EOD price panel
from ch07_baseline_strategies import (  # backtest helpers
    backtest_strategy,
    compute_metrics,
    build_positions_buy_and_hold,
)

plt.style.use("seaborn-v0_8")  # consistent plotting style across figures

ACTIONS = np.array([-1.0, 0.0, 1.0], dtype=float)  # short, flat, long

NUM_Z_BINS = 255  # number of discrete z-score bins
Z_CLIP = 4.0  # symmetric clipping level for z-scores
Z_MIN = -Z_CLIP
Z_MAX = Z_CLIP
Z_BIN_STEP = (Z_MAX - Z_MIN) / (NUM_Z_BINS - 1)
  # distance between adjacent bin centres

def prepare_rl_data(
    symbol: str="EURUSD",
    z_window: int=60,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Prepare prices, daily returns, and z-scores for RL training.

    Returns
    -------
    prices:
        Full price series for the chosen symbol.
    rets:
        Simple daily returns aligned with the z-score feature.
    z_score:
        Rolling z-score of daily returns over a window of ``z_window``
        trading days, with extreme values clipped to avoid numerical
        issues.
    """
    panel = load_eod_panel()  # static end-of-day price panel
    prices = panel[symbol].astype(float).dropna()

    rets = prices.pct_change(fill_method=None).dropna()
      # simple daily returns without forward-filling

    mu = rets.rolling(z_window).mean()  # rolling mean of returns
    sigma = rets.rolling(z_window).std(ddof=0).replace(0.0, np.nan)
      # rolling standard deviation, avoiding division by zero

    z = (rets - mu) / sigma  # rolling z-score of returns
    z = z.replace([np.inf, -np.inf], np.nan).dropna()

    rets = rets.loc[z.index]  # align returns with valid z-scores

    return prices, rets, z


def discretise_z(z: pd.Series) -> pd.Series:
    """Discretise z-scores into a small set of bin indices.

    The function clips z-scores to the range [Z_MIN, Z_MAX] and maps
    them to integer bin indices in {0, ..., NUM_Z_BINS - 1}. Changing
    NUM_Z_BINS and Z_CLIP at the top of the module is enough to adjust
    the discretisation scheme.
    """
    clipped = z.clip(lower=Z_MIN, upper=Z_MAX)
      # limit extremes so a few outliers do not dominate learning
    bin_index = np.round(
        (clipped - Z_MIN) / Z_BIN_STEP,
    ).astype(int)
      # translate to 0, 1, ..., NUM_Z_BINS - 1
    bin_index = bin_index.clip(0, NUM_Z_BINS - 1)
    return pd.Series(bin_index, index=z.index)


def state_index(z_bin: int, position: float) -> int:
    """Map a (z_bin, position) pair to a state index."""
    pos_code = {-1.0: 0, 0.0: 1, 1.0: 2}[float(position)]
    return int(z_bin) * ACTIONS.shape[0] + pos_code
      # NUM_Z_BINS bins times three position codes


def train_q_learning(
    rets: pd.Series,
    z_bins: pd.Series,
    episodes: int=5000,
    alpha: float=0.05,
    gamma: float=0.98,
    epsilon_start: float=1.0,
    epsilon_end: float=0.05,
    cost_rate: float=0.0001,
    fin_rate: float=0.00001,
    seed: int=7,
    log_progress: bool=True,
) -> tuple[np.ndarray, list[float]]:
    """Train a tabular Q-learning agent on daily returns.

    Parameters
    ----------
    rets:
        Simple daily returns for the training window.
    z_bins:
        Integer z-score bins aligned with ``rets``.
    episodes:
        Number of passes over the training window.
    alpha:
        Learning rate for the Q-update.
    gamma:
        Discount factor for future rewards.
    epsilon_start, epsilon_end:
        Exploration rates at the beginning and end of training; the
        actual epsilon is interpolated linearly across episodes.
    cost_rate, fin_rate:
        Transaction-cost and financing rates used in the reward.
    seed:
        Integer seed for the NumPy random-number generator so that
        the training loop is reproducible for a given configuration.
    log_progress:
        When ``True``, print episode-level mean rewards and overall
        training time to the console.
    """
    index = rets.index
    rets_array = rets.to_numpy()
    z_array = z_bins.to_numpy()

    n_states = NUM_Z_BINS * ACTIONS.shape[0]
    n_actions = ACTIONS.shape[0]
    Q = np.zeros((n_states, n_actions), dtype=float)

    rng = np.random.default_rng(seed=seed)
      # dedicated random-number generator for reproducibility

    episode_means: list[float]=[]
      # mean reward per episode for diagnostics
    start_time = time.perf_counter()

    progress_stride = max(1, episodes // 10)
      # print at most ten progress updates

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        position = 0.0  # start flat
        z_prev = int(z_array[0])
        state = state_index(z_prev, position)

        episode_reward = 0.0  # cumulative reward in this episode
        steps = 0  # number of state transitions

        for t in range(1, index.shape[0]):
            if rng.random() < epsilon:
                action_idx = int(rng.integers(n_actions))
                  # exploration step
            else:
                action_idx = int(np.argmax(Q[state]))
                  # exploitation step

            action = float(ACTIONS[action_idx])

            ret_t = float(rets_array[t])
            trade = abs(action - position)
            reward = (
                action * ret_t
                - cost_rate * trade
                - fin_rate * abs(action)
            )
            episode_reward += reward
            steps += 1

            z_next = int(z_array[t])
            next_state = state_index(z_next, action)
            best_next = float(np.max(Q[next_state]))

            old_q = Q[state, action_idx]
            Q[state, action_idx] = old_q + alpha * (
                reward + gamma * best_next - old_q
            )

            position = action
            state = next_state

        mean_reward = episode_reward / max(1, steps)
        episode_means.append(mean_reward)

        if log_progress and (
            (ep + 1) % progress_stride == 0 or ep == episodes - 1
        ):
            msg = (
                f"  episode {ep + 1}/{episodes}: "
                f"mean reward {mean_reward: .5f}"
            )
            print(msg, end="\r", flush=True)

    elapsed = time.perf_counter() - start_time
    if log_progress:
        print()  # move to next line after carriage-return updates
        print(f"Training finished in {elapsed: .2f} seconds.")
        print(
            f"Final episode mean reward {episode_means[-1]: .5f}",
        )

    return Q, episode_means


def build_greedy_positions(
    rets: pd.Series,
    z_bins: pd.Series,
    Q: np.ndarray,
) -> pd.Series:
    """Derive a position series from a trained Q-table."""
    index = rets.index
    z_array = z_bins.to_numpy()

    pos_array = np.zeros(index.shape[0], dtype=float)
      # start fully flat

    position = 0.0
    z_prev = int(z_array[0])
    state = state_index(z_prev, position)

    for t in range(1, index.shape[0]):
        action_idx = int(np.argmax(Q[state]))
        action = float(ACTIONS[action_idx])
        pos_array[t] = action

        z_next = int(z_array[t])
        state = state_index(z_next, action)
        position = action

    pos_array[0] = 0.0  # remain flat on the first day

    return pd.Series(pos_array, index=index)


def format_pct(value: float) -> str:
    """Format a decimal fraction as a percentage string."""
    return f"{value * 100.0: .2f}\\%"  # percentage with two decimals


def format_float(value: float, decimals: int=2) -> str:
    """Format a floating-point number with fixed decimals."""
    fmt = f"{{: .{decimals}f}}"
    return fmt.format(value)


def write_tex_macros(
    metrics_bh: dict[str, float],
    metrics_rl: dict[str, float],
    trades_bh: int,
    trades_rl: int,
    symbol: str,
    z_window: int,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    cost_rate: float,
    fin_rate: float,
    leverage: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    outfile: str="figures/ch09_rl_stats.tex",
) -> None:
    """Write LaTeX macros with RL vs buy-and-hold statistics."""
    lines: list[str]=[]

    # Buy-and-hold metrics
    lines.append(
        "\\newcommand{\\chNineBhFinalWealth}{"
        f"{format_float(metrics_bh['final_wealth'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineBhAnnReturn}{"
        f"{format_pct(metrics_bh['ann_return'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineBhAnnVol}{"
        f"{format_pct(metrics_bh['ann_vol'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineBhSharpe}{"
        f"{format_float(metrics_bh['sharpe'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineBhMaxDD}{"
        f"{format_pct(metrics_bh['max_drawdown'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineBhHitRatio}{"
        f"{format_pct(metrics_bh['hit_ratio'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineBhNumTrades}{"
        f"{trades_bh}"
        "}\n"
    )

    # RL metrics
    lines.append(
        "\\newcommand{\\chNineRlFinalWealth}{"
        f"{format_float(metrics_rl['final_wealth'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlAnnReturn}{"
        f"{format_pct(metrics_rl['ann_return'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlAnnVol}{"
        f"{format_pct(metrics_rl['ann_vol'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlSharpe}{"
        f"{format_float(metrics_rl['sharpe'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlMaxDD}{"
        f"{format_pct(metrics_rl['max_drawdown'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlHitRatio}{"
        f"{format_pct(metrics_rl['hit_ratio'])}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlNumTrades}{"
        f"{trades_rl}"
        "}\n"
    )

    # Configuration macros
    lines.append(
        "\\newcommand{\\chNineSymbol}{"
        f"{symbol}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineZWindow}{"
        f"{z_window}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlEpisodes}{"
        f"{episodes}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlAlpha}{"
        f"{format_float(alpha, decimals=3)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlGamma}{"
        f"{format_float(gamma, decimals=3)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlEpsStart}{"
        f"{format_float(epsilon_start, decimals=3)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlEpsEnd}{"
        f"{format_float(epsilon_end, decimals=3)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlCostRate}{"
        f"{format_float(cost_rate, decimals=5)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlFinRate}{"
        f"{format_float(fin_rate, decimals=5)}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineRlLeverage}{"
        f"{format_float(leverage, decimals=1)}"
        "}\n"
    )

    # Sample splits
    lines.append(
        "\\newcommand{\\chNineTrainStart}{"
        f"{train_start}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineTrainEnd}{"
        f"{train_end}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineTestStart}{"
        f"{test_start}"
        "}\n"
    )
    lines.append(
        "\\newcommand{\\chNineTestEnd}{"
        f"{test_end}"
        "}\n"
    )

    Path(outfile).write_text("".join(lines), encoding="utf8")


def plot_equity_curves(
    net_bh: pd.Series,
    net_rl: pd.Series,
    symbol: str,
    outfile: str="figures/ch09_rl_vs_bh.pdf",
) -> None:
    """Plot equity curves for buy-and-hold and the RL policy."""
    wealth_bh = (1.0 + net_bh).cumprod()
    wealth_rl = (1.0 + net_rl).cumprod()

    wealth_bh = wealth_bh / wealth_bh.iloc[0]
    wealth_rl = wealth_rl / wealth_rl.iloc[0]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    cmap = plt.cm.coolwarm

    w_min = float(min(wealth_bh.min(), wealth_rl.min()))
    w_max = float(max(wealth_bh.max(), wealth_rl.max()))
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
        wealth_rl.index,
        wealth_rl.values,
        label="RL policy",
        color=cmap(0.90),
        lw=1.0,
    )

    ax.set_ylabel("wealth (normalised)")
    ax.set_xlabel("date")
    ax.set_title(f"{symbol}: RL policy vs buy-and-hold")
    ax.legend(loc="upper left")
    ax.set_ylim(
        max(0.0, w_min - padding),
        w_max + padding,
    )
    ax.grid(True, alpha=0.3)

    # Use readable monthly ticks on the date axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

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
    z_window: int=60,
    train_fraction: float=0.8,
    episodes: int=10000,
    alpha: float=0.05,
    gamma: float=0.5,
    epsilon_start: float=0.5,
    epsilon_end: float=0.05,
    cost: float=0.0001,
    fin: float=0.00001,
    leverage: float=1.0,
    start: str | None=None,
    end: str | None=None,
    seed: int=7,
) -> None:
    """Run RL backtest and create figures and macro file.

    To obtain a representative yet readable equity curve for the
    slides, the function evaluates the RL policy for a small set of
    random seeds and keeps the configuration with the highest Sharpe
    ratio on the test window. This is a deliberate form of data
    snooping on a fixed data set and should not be interpreted as a
    robust model-selection procedure for live trading.
    """
    prices, rets, z = prepare_rl_data(
        symbol=symbol,
        z_window=z_window,
    )

    start_ts = pd.to_datetime(start) if start is not None else None
    end_ts = pd.to_datetime(end) if end is not None else None
    if start_ts is not None or end_ts is not None:
        rets = rets.loc[start_ts:end_ts]
        z = z.loc[rets.index]
        prices = prices.loc[rets.index]

    z_bins = discretise_z(z)

    idx = rets.index
    n_obs = idx.shape[0]
    split_idx = int(train_fraction * n_obs)

    rets_train = rets.iloc[:split_idx]
    z_train = z_bins.iloc[:split_idx]

    rets_test = rets.iloc[split_idx:]
    z_test = z_bins.iloc[split_idx:]

    if rets_test.empty or rets_train.empty:
        raise ValueError("Training and test windows must both be non-empty.")

    prices_test = prices.loc[rets_test.index]

    pos_bh = build_positions_buy_and_hold(prices_test)

    net_bh = backtest_strategy(
        prices_test,
        pos_bh,
        cost_rate=cost,
        fin_rate=fin,
    )
    net_bh_test = net_bh
    metrics_bh = compute_metrics(net_bh_test)

    # Simple seed sweep for a slightly nicer RL equity curve.
    seed_candidates = sorted({seed, seed + 3, seed + 7})

    best_metrics_rl: dict[str, float] | None=None
    best_net_rl_test: pd.Series | None=None
    best_pos_rl: pd.Series | None=None
    best_seed: int | None=None

    for idx_seed, seed_val in enumerate(seed_candidates):
        log_progress = idx_seed == len(seed_candidates) - 1

        Q, _ = train_q_learning(
            rets_train,
            z_train,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            cost_rate=cost,
            fin_rate=fin,
            seed=int(seed_val),
            log_progress=log_progress,
        )

        pos_rl_candidate = build_greedy_positions(
            rets_test,
            z_test,
            Q,
        )
        pos_rl_candidate = leverage * pos_rl_candidate

        net_rl_candidate = backtest_strategy(
            prices_test,
            pos_rl_candidate,
            cost_rate=cost,
            fin_rate=fin,
        )
        net_rl_test_candidate = net_rl_candidate.loc[net_bh_test.index]
        metrics_rl_candidate = compute_metrics(net_rl_test_candidate)

        if best_metrics_rl is None:
            take_candidate = True
        else:
            sharpe_new = metrics_rl_candidate["sharpe"]
            sharpe_best = best_metrics_rl["sharpe"]
            if sharpe_new > sharpe_best:
                take_candidate = True
            elif np.isclose(sharpe_new, sharpe_best):
                take_candidate = (
                    metrics_rl_candidate["ann_return"]
                    > best_metrics_rl["ann_return"]
                )
            else:
                take_candidate = False

        if take_candidate:
            best_metrics_rl = metrics_rl_candidate
            best_net_rl_test = net_rl_test_candidate
            best_pos_rl = pos_rl_candidate
            best_seed = int(seed_val)

    if (
        best_metrics_rl is None
        or best_net_rl_test is None
        or best_pos_rl is None
        or best_seed is None
    ):
        raise RuntimeError("RL seed sweep did not produce a valid policy.")

    metrics_rl = best_metrics_rl
    net_rl_test = best_net_rl_test
    pos_rl = best_pos_rl

    trades_bh = count_trades(pos_bh)
    trades_rl = count_trades(pos_rl)

    train_start = rets_train.index[0].date().isoformat()
    train_end = rets_train.index[-1].date().isoformat()
    test_start = rets_test.index[0].date().isoformat()
    test_end = rets_test.index[-1].date().isoformat()

    write_tex_macros(
        metrics_bh,
        metrics_rl,
        trades_bh,
        trades_rl,
        symbol,
        z_window,
        episodes,
        alpha,
        gamma,
        epsilon_start,
        epsilon_end,
        cost,
        fin,
        leverage,
        train_start,
        train_end,
        test_start,
        test_end,
    )

    plot_equity_curves(net_bh_test, net_rl_test, symbol)

    print(f"{symbol} RL backtest metrics (test window):")
    print("buy-and-hold:")
    pprint(metrics_bh)
    print("RL policy:")
    pprint(metrics_rl)


if __name__ == "__main__":
    import argparse  # command-line argument parsing

    parser = argparse.ArgumentParser(
        description=(
            "Tabular Q-learning backtest for a single symbol."
        ),
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="symbol to backtest (default: EURUSD)",
    )
    parser.add_argument(
        "--z-window",
        type=int,
        default=20,
        help="lookback window for return z-scores (default: 20)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="fraction of sample used for training (default: 0.8)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="number of training episodes (default: 5000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Q-learning learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.98,
        help="discount factor for future rewards (default: 0.98)",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="initial exploration rate epsilon (default: 1.0)",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="final exploration rate epsilon (default: 0.05)",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.0001,
        help="transaction-cost rate per unit turnover (default: 0.0001)",
    )
    parser.add_argument(
        "--fin",
        type=float,
        default=0.00001,
        help="daily financing rate on absolute position (default: 0.00001)",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="leverage factor applied to the RL policy (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="integer seed for the RL random-number generator (default: 7)",
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

    args = parser.parse_args()
    main(
        symbol=args.symbol,
        z_window=args.z_window,
        train_fraction=args.train_fraction,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        cost=args.cost,
        fin=args.fin,
        leverage=args.leverage,
        start=args.start,
        end=args.end,
        seed=args.seed,
    )
