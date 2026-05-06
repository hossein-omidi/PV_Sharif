"""Multi-seed DQN training with Stable-Baselines3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from custom_rl import register_envs


DEFAULT_LOG_DIR = "logs/dqn_cartpole"
DEFAULT_MODEL_DIR = "models/dqn_cartpole"


def validate_dqn_env(env: Any) -> None:
    """
    Validate that the environment is compatible with DQN.

    Expected action space:
        Discrete(2)

    Meaning:
        action 0 -> left
        action 1 -> right
    """
    action_space = env.action_space

    if not isinstance(action_space, spaces.Discrete):
        raise TypeError(
            "DQN requires a discrete action space. "
            f"Found action_space={action_space}."
        )

    if getattr(action_space, "start", 0) != 0:
        raise ValueError(
            "Stable-Baselines3 expects Discrete action spaces to start at 0. "
            f"Found action_space={action_space}."
        )

    if action_space.n != 2:
        raise ValueError(
            "This CartPole DQN setup expects exactly two actions: "
            "0 for left and 1 for right. "
            f"Found action_space={action_space}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN on CustomODECartPole")

    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--reward", default="sparse", choices=["dense", "sparse"])
    parser.add_argument("--total-timesteps", type=int, default=100_000)

    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--save-dir", default=DEFAULT_MODEL_DIR)

    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of vectorized environments. For DQN, 1 is usually sufficient.",
    )

    parser.add_argument(
        "--vec-env",
        choices=["dummy", "subproc"],
        default="dummy",
        help="Vectorized env type.",
    )

    # DQN hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument(
        "--train-freq",
        type=int,
        default=4,
        help="Train the Q-network every N environment steps.",
    )

    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Number of gradient updates after each training trigger.",
    )

    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1_000,
        help="Update the target Q-network every N environment steps.",
    )

    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=10.0,
        help="Maximum gradient norm for DQN gradient clipping.",
    )

    parser.add_argument(
        "--net-arch",
        nargs="+",
        type=int,
        default=[128, 128],
        help="Hidden layer sizes for the DQN Q-network.",
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Evaluate every N environment steps.",
    )

    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=10)

    args = parser.parse_args()

    register_envs()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Validate the registered environment before starting training.
    test_env = gym.make("CustomODECartPole-v0", reward_id=args.reward)
    validate_dqn_env(test_env)
    test_env.close()

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv

    for seed in args.seeds:
        print(f"\nStarting DQN training for seed {seed}")

        seed_log_dir = Path(args.log_dir) / f"seed_{seed}"
        seed_model_dir = Path(args.save_dir) / f"seed_{seed}"

        seed_log_dir.mkdir(parents=True, exist_ok=True)
        seed_model_dir.mkdir(parents=True, exist_ok=True)

        best_model_dir = seed_model_dir / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)

        env = make_vec_env(
            env_id="CustomODECartPole-v0",
            n_envs=args.n_envs,
            seed=seed,
            vec_env_cls=vec_env_cls,
            monitor_dir=str(seed_log_dir),
            env_kwargs={"reward_id": args.reward},
        )

        eval_env = make_vec_env(
            env_id="CustomODECartPole-v0",
            n_envs=1,
            seed=seed + 10_000,
            vec_env_cls=DummyVecEnv,
            env_kwargs={"reward_id": args.reward},
        )

        validate_dqn_env(env)

        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(seed_log_dir),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )

        model = DQN(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            target_update_interval=args.target_update_interval,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            exploration_fraction=args.exploration_fraction,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs={"net_arch": args.net_arch},
            tensorboard_log=str(seed_log_dir / "tensorboard"),
            verbose=1,
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            log_interval=args.log_interval,
        )

        final_model_path = seed_model_dir / "final_model"
        model.save(str(final_model_path))

        env.close()
        eval_env.close()

        print(f"Seed {seed} finished.")
        print(f"  Best model directory: {best_model_dir}")
        print(f"  Final model path:     {final_model_path}")

    print("\nDQN training completed.")
    print(f"Logs:   {args.log_dir}")
    print(f"Models: {args.save_dir}")


if __name__ == "__main__":
    main()