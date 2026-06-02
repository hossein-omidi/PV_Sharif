"""Train a DQN agent on PVSimEnv."""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from custom_rl import register_envs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on PVSimEnv-v0.")

    parser.add_argument(
        "--weather-mode",
        choices=["clearsky", "pvgis_tmy"],
        default="clearsky",
        help="Weather source used by PVSimEnv.",
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5_000,
        help="Number of training timesteps.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/dqn_pvsim",
        help="Directory used for saving the trained model.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/dqn_pvsim",
        help="TensorBoard log directory.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model name. If not provided, a name is chosen from weather mode.",
    )

    return parser.parse_args()


def get_default_model_name(weather_mode: str) -> str:
    if weather_mode == "clearsky":
        return "dqn_pvsim_model"

    return f"dqn_pvsim_{weather_mode}_model"


def main() -> int:
    args = parse_args()

    register_envs()

    print("Creating PVSimEnv-v0...")
    print(f"Weather mode: {args.weather_mode}")

    env = gym.make("PVSimEnv-v0", weather_mode=args.weather_mode)
    raw_env = env.unwrapped

    print("Environment created.")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Env weather mode:  {raw_env.weather_mode}")

    if raw_env.weather_data is not None:
        print(f"Weather data shape: {raw_env.weather_data.shape}")

    env = Monitor(env)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=100,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=250,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
    )

    print("\nTraining DQN on PVSimEnv-v0...")
    print(f"Total timesteps: {args.total_timesteps}")

    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name=f"DQN_{args.weather_mode}",
    )

    model_name = args.model_name or get_default_model_name(args.weather_mode)
    model_path = os.path.join(args.model_dir, model_name)

    model.save(model_path)

    env.close()

    print(f"\nTraining finished.")
    print(f"Model saved to: {model_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())