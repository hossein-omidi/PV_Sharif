"""Evaluate trained DQN policy and save trajectories for plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN

from custom_rl import register_envs


DEFAULT_MODEL_DIR = "models/dqn_cartpole"
DEFAULT_TRAJ_DIR = "eval_trajectories"


def find_model_path(model_dir: Path, seed: int) -> Optional[Path]:
    """
    Find trained model for a given seed.

    Expected DQN training structure:
        models/dqn_cartpole/seed_0/best_model/best_model.zip
        models/dqn_cartpole/seed_0/final_model.zip

    Older fallback paths are also checked for safety.
    """
    candidates = [
        model_dir / f"seed_{seed}" / "best_model" / "best_model.zip",
        model_dir / f"seed_{seed}" / "final_model.zip",
        model_dir / f"best_{seed}" / "best_model.zip",
        model_dir / f"final_{seed}.zip",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def validate_eval_env(env: gym.Env) -> None:
    """
    Validate that the environment is compatible with DQN evaluation.

    Expected action space:
        Discrete(2)

    Meaning:
        action 0 -> left
        action 1 -> right
    """
    if not isinstance(env.action_space, spaces.Discrete):
        raise TypeError(
            "DQN evaluation requires a discrete action space. "
            f"Found action_space={env.action_space}."
        )

    if getattr(env.action_space, "start", 0) != 0:
        raise ValueError(
            "Stable-Baselines3 expects Discrete action spaces to start at 0. "
            f"Found action_space={env.action_space}."
        )

    if env.action_space.n != 2:
        raise ValueError(
            "Expected Discrete(2) action space for left/right CartPole. "
            f"Found action_space={env.action_space}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN policy and save trajectories"
    )

    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--out-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--reward", default="sparse", choices=["dense", "sparse"])

    args = parser.parse_args()

    register_envs()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        model_path = find_model_path(model_dir, seed)

        if model_path is None:
            print(f"Skip seed {seed}: no trained model found in {model_dir}")
            continue

        print(f"\nEvaluating seed {seed}")
        print(f"  Model: {model_path}")

        env = gym.make("CustomODECartPole-v0", reward_id=args.reward)
        validate_eval_env(env)

        model = DQN.load(str(model_path), env=env)

        trajectories = []

        for ep in range(args.n_episodes):
            obs, _ = env.reset(seed=seed + 1000 + ep)

            states = []
            actions = []
            rewards = []

            while True:
                action, _ = model.predict(obs, deterministic=True)

                action_arr = np.asarray(action)

                if action_arr.size != 1:
                    raise RuntimeError(
                        f"Invalid action shape from model.predict(): "
                        f"{action_arr.shape}"
                    )

                action_int = int(action_arr.item())

                if not env.action_space.contains(action_int):
                    raise RuntimeError(
                        f"Model produced invalid action {action_int}. "
                        f"Expected action in {env.action_space}."
                    )

                states.append(np.asarray(obs, dtype=np.float64).tolist())
                actions.append(action_int)

                obs, reward, terminated, truncated, _ = env.step(action_int)

                if not np.isfinite(obs).all():
                    raise RuntimeError(
                        f"NaN/Inf observation during evaluation. "
                        f"seed={seed}, episode={ep}"
                    )

                if not np.isfinite(reward):
                    raise RuntimeError(
                        f"NaN/Inf reward during evaluation. "
                        f"seed={seed}, episode={ep}"
                    )

                rewards.append(float(reward))

                if terminated or truncated:
                    break

            episode_return = float(np.sum(rewards))
            episode_length = len(rewards)

            trajectories.append(
                {
                    "seed": seed,
                    "episode": ep,
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                    "return": episode_return,
                    "length": episode_length,
                }
            )

            print(
                f"  Episode {ep + 1:03d}/{args.n_episodes}: "
                f"return={episode_return:.2f}, length={episode_length}"
            )

        out_path = out_dir / f"trajectories_seed{seed}.json"

        with open(out_path, "w") as f:
            json.dump(trajectories, f, indent=2)

        print(f"Saved {args.n_episodes} episodes to {out_path}")

        env.close()

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()