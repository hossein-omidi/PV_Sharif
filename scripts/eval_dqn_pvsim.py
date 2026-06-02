"""Evaluate a trained DQN agent on PVSimEnv."""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
from stable_baselines3 import DQN

from custom_rl import register_envs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DQN on PVSimEnv-v0.")

    parser.add_argument(
        "--weather-mode",
        choices=["clearsky", "pvgis_tmy"],
        default="clearsky",
        help="Weather source used by PVSimEnv.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Maximum number of evaluation steps.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/dqn_pvsim",
        help="Directory containing the trained model.",
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


def resolve_model_path(model_dir: str, model_name: str) -> str:
    model_path = os.path.join(model_dir, model_name)

    if os.path.exists(model_path):
        return model_path

    if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
        return model_path + ".zip"

    return model_path


def main() -> int:
    args = parse_args()

    register_envs()

    model_name = args.model_name or get_default_model_name(args.weather_mode)
    model_path = resolve_model_path(args.model_dir, model_name)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Run training first, for example:")
        print(
            "python scripts/train_dqn_pvsim.py "
            f"--weather-mode {args.weather_mode}"
        )
        return 1

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

    print(f"\nLoading model: {model_path}")
    model = DQN.load(model_path, env=env)

    obs, info = env.reset(seed=args.seed)

    total_reward = 0.0

    print("\nEvaluating DQN on PVSimEnv-v0...")
    print("Initial observation:", obs)
    print("Initial info:", info)

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        total_reward += reward

        print(f"\nStep {step + 1}")
        print("Action:", int(action))
        print("Observation:", obs)
        print("Reward:", reward)
        print("Weather mode:", info["weather_mode"])
        print("Weather source time:", info["weather_source_time"])
        print("PV power:", info["pv_power"])
        print("Terminated:", terminated)
        print("Truncated:", truncated)

        if terminated or truncated:
            break

    env.close()

    print(f"\nTotal evaluation reward: {total_reward:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())