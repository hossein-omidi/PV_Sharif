"""Plot DQN training results and evaluation trajectory summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG_DIR = "logs/dqn_cartpole"
DEFAULT_TRAJ_DIR = "eval_trajectories"
DEFAULT_PLOT_DIR = "plots"


def load_monitor_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load episode returns and lengths from an SB3 Monitor CSV file."""
    try:
        # SB3 Monitor CSV format:
        # line 1: # metadata
        # line 2: r,l,t
        # line 3+: data
        data = np.loadtxt(path, delimiter=",", skiprows=2)
    except Exception:
        return np.array([]), np.array([])

    if data.size == 0:
        return np.array([]), np.array([])

    if data.ndim == 1:
        data = data.reshape(1, -1)

    returns = data[:, 0]
    lengths = data[:, 1] if data.shape[1] > 1 else np.ones_like(returns)

    return returns, lengths


def load_all_seed_logs(
    log_dir: Path,
    seeds: list[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load Monitor logs for each seed."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for seed in seeds:
        seed_dir = log_dir / f"seed_{seed}"

        if not seed_dir.exists():
            print(f"Warning: log directory not found for seed {seed}: {seed_dir}")
            continue

        monitor_files = sorted(seed_dir.glob("*.monitor.csv"))

        if not monitor_files:
            fallback = seed_dir / "monitor.csv"
            if fallback.exists():
                monitor_files = [fallback]

        if not monitor_files:
            print(f"Warning: no monitor CSV files found for seed {seed} in {seed_dir}")
            continue

        all_returns = []
        all_lengths = []

        for monitor_file in monitor_files:
            returns, lengths = load_monitor_csv(monitor_file)

            if len(returns) > 0:
                all_returns.append(returns)
                all_lengths.append(lengths)

        if all_returns:
            out[seed] = (
                np.concatenate(all_returns),
                np.concatenate(all_lengths),
            )

    return out


def smooth_curve(y: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average smoothing."""
    if window <= 1 or len(y) < 2:
        return y

    window = min(window, len(y))

    kernel = np.ones(window) / window

    return np.convolve(y, kernel, mode="same")


def plot_learning_curve(
    log_dir: Path,
    out_dir: Path,
    seeds: list[int],
    smooth: int = 10,
) -> None:
    """Plot DQN training episode return curve."""
    data = load_all_seed_logs(log_dir, seeds)

    if not data:
        print(f"No training monitor data found in {log_dir}")
        return

    valid_data = {
        seed: (returns, lengths)
        for seed, (returns, lengths) in data.items()
        if len(returns) > 0 and len(lengths) > 0
    }

    if not valid_data:
        print("No valid training episodes found.")
        return

    max_common_steps = min(float(np.sum(lengths)) for _, lengths in valid_data.values())

    if max_common_steps <= 0:
        print("No positive training steps found.")
        return

    step_grid = np.linspace(0.0, max_common_steps, num=200)
    interpolated_returns = []

    fig, ax = plt.subplots(figsize=(9, 5))

    for seed in sorted(valid_data.keys()):
        returns, lengths = valid_data[seed]

        steps = np.concatenate([[0.0], np.cumsum(lengths)])
        returns_extended = np.concatenate([[returns[0]], returns])

        seed_curve = np.interp(step_grid, steps, returns_extended)
        seed_curve_smooth = smooth_curve(seed_curve, smooth)

        interpolated_returns.append(seed_curve_smooth)

        ax.plot(
            step_grid,
            seed_curve_smooth,
            linewidth=1.0,
            alpha=0.5,
            label=f"seed {seed}",
        )

    returns_matrix = np.asarray(interpolated_returns)

    mean_return = np.mean(returns_matrix, axis=0)
    std_return = np.std(returns_matrix, axis=0)

    ax.plot(
        step_grid,
        mean_return,
        linewidth=2.5,
        label="mean",
    )

    ax.fill_between(
        step_grid,
        mean_return - std_return,
        mean_return + std_return,
        alpha=0.25,
        label="mean ± std",
    )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title("DQN training: episode return")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out_path = out_dir / "learning_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


def load_trajectories(
    traj_dir: Path,
    seeds: list[int],
) -> dict[int, list[dict]]:
    """Load evaluation trajectory JSON files."""
    out: dict[int, list[dict]] = {}

    for seed in seeds:
        path = traj_dir / f"trajectories_seed{seed}.json"

        if not path.exists():
            print(f"Warning: trajectory file not found for seed {seed}: {path}")
            continue

        with open(path, "r") as f:
            out[seed] = json.load(f)

    return out


def plot_trajectory_states(
    traj_dir: Path,
    out_dir: Path,
    seeds: list[int],
) -> None:
    """Plot mean ± std of evaluation states over time."""
    data = load_trajectories(traj_dir, seeds)

    if not data:
        print(f"No trajectory data found in {traj_dir}")
        return

    trajectories = []

    for seed in sorted(data.keys()):
        for episode in data[seed]:
            states = np.asarray(episode["states"], dtype=np.float64)

            if states.size == 0:
                continue

            if states.ndim != 2 or states.shape[1] != 4:
                print(
                    f"Skipping invalid states for seed {seed}: "
                    f"shape={states.shape}"
                )
                continue

            trajectories.append(states)

    if not trajectories:
        print("No valid state trajectories found.")
        return

    max_t = max(states.shape[0] for states in trajectories)
    n_traj = len(trajectories)

    S = np.full((n_traj, max_t, 4), np.nan, dtype=np.float64)

    for i, states in enumerate(trajectories):
        T = states.shape[0]
        S[i, :T, :] = states

    t_grid = np.arange(max_t)

    labels = [
        "x: cart position",
        "x_dot: cart velocity",
        "theta: pole angle",
        "theta_dot: pole angular velocity",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for dim, ax in enumerate(axes):
        mean_state = np.nanmean(S[:, :, dim], axis=0)
        std_state = np.nanstd(S[:, :, dim], axis=0)

        valid = ~np.isnan(mean_state)

        ax.plot(
            t_grid[valid],
            mean_state[valid],
            linewidth=1.8,
        )

        ax.fill_between(
            t_grid[valid],
            mean_state[valid] - std_state[valid],
            mean_state[valid] + std_state[valid],
            alpha=0.25,
        )

        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Step")
    axes[3].set_xlabel("Step")

    fig.suptitle("DQN evaluation trajectories: states")
    fig.tight_layout()

    out_path = out_dir / "trajectory_states.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


def plot_trajectory_actions(
    traj_dir: Path,
    out_dir: Path,
    seeds: list[int],
) -> None:
    """Plot discrete DQN actions over evaluation trajectories."""
    data = load_trajectories(traj_dir, seeds)

    if not data:
        print(f"No trajectory data found in {traj_dir}")
        return

    actions_all = []

    for seed in sorted(data.keys()):
        for episode in data[seed]:
            actions = np.asarray(episode["actions"], dtype=np.float64).reshape(-1)

            if actions.size == 0:
                continue

            actions_all.append(actions)

    if not actions_all:
        print("No valid action trajectories found.")
        return

    max_t = max(actions.shape[0] for actions in actions_all)
    n_traj = len(actions_all)

    A = np.full((n_traj, max_t), np.nan, dtype=np.float64)

    for i, actions in enumerate(actions_all):
        T = actions.shape[0]
        A[i, :T] = actions

    t_grid = np.arange(max_t)

    mean_action = np.nanmean(A, axis=0)
    std_action = np.nanstd(A, axis=0)

    valid = ~np.isnan(mean_action)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        t_grid[valid],
        mean_action[valid],
        linewidth=1.8,
        label="mean action",
    )

    ax.fill_between(
        t_grid[valid],
        mean_action[valid] - std_action[valid],
        mean_action[valid] + std_action[valid],
        alpha=0.25,
        label="mean ± std",
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Discrete action")
    ax.set_title("DQN evaluation trajectories: actions")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0: left", "1: right"])
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out_path = out_dir / "trajectory_actions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")


def plot_evaluation_returns(
    traj_dir: Path,
    out_dir: Path,
    seeds: list[int],
) -> None:
    """Plot evaluation return distribution."""
    data = load_trajectories(traj_dir, seeds)

    if not data:
        print(f"No trajectory data found in {traj_dir}")
        return

    returns = []
    lengths = []

    for seed in sorted(data.keys()):
        for episode in data[seed]:
            returns.append(float(episode["return"]))
            lengths.append(float(episode["length"]))

    if not returns:
        print("No evaluation returns found.")
        return

    returns_arr = np.asarray(returns, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4))

    bins = min(10, max(len(returns_arr), 1))

    ax.hist(returns_arr, bins=bins)

    ax.set_xlabel("Episode return")
    ax.set_ylabel("Count")
    ax.set_title(
        "DQN evaluation return distribution\n"
        f"mean={np.mean(returns_arr):.2f}, "
        f"std={np.std(returns_arr):.2f}, "
        f"min={np.min(returns_arr):.2f}, "
        f"max={np.max(returns_arr):.2f}"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = out_dir / "evaluation_returns.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved {out_path}")

    print("\nEvaluation summary:")
    print(f"  Episodes:      {len(returns_arr)}")
    print(f"  Return mean:   {np.mean(returns_arr):.2f}")
    print(f"  Return std:    {np.std(returns_arr):.2f}")
    print(f"  Return min:    {np.min(returns_arr):.2f}")
    print(f"  Return max:    {np.max(returns_arr):.2f}")
    print(f"  Length mean:   {np.mean(lengths_arr):.2f}")
    print(f"  Length std:    {np.std(lengths_arr):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DQN training and evaluation results"
    )

    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--smooth", type=int, default=10)

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    traj_dir = Path(args.traj_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curve(
        log_dir=log_dir,
        out_dir=out_dir,
        seeds=args.seeds,
        smooth=args.smooth,
    )

    plot_trajectory_states(
        traj_dir=traj_dir,
        out_dir=out_dir,
        seeds=args.seeds,
    )

    plot_trajectory_actions(
        traj_dir=traj_dir,
        out_dir=out_dir,
        seeds=args.seeds,
    )

    plot_evaluation_returns(
        traj_dir=traj_dir,
        out_dir=out_dir,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()