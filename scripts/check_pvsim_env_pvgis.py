"""Check PVSimEnv with PVGIS TMY weather mode."""

from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from custom_rl import register_envs


def main() -> int:
    register_envs()

    print("Creating PVSimEnv-v0 with weather_mode='pvgis_tmy'...")
    env = gym.make("PVSimEnv-v0", weather_mode="pvgis_tmy")
    raw_env = env.unwrapped

    print("\nPV environment spaces:")
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space:      {env.action_space}")

    print("\nWeather settings:")
    print(f"  weather_mode: {raw_env.weather_mode}")
    print(f"  weather_data_shape: {raw_env.weather_data.shape}")

    print("\nRunning Gymnasium check_env...")
    check_env(raw_env, skip_render_check=True)
    print("check_env passed.")

    print("\nTesting reset and random rollout with PVGIS TMY...")
    obs, info = env.reset(seed=42)

    print("Initial observation:", obs)
    print("Initial info:")
    print(info)

    total_reward = 0.0

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if not np.isfinite(obs).all():
            print(f"ERROR: NaN/Inf observation at step {step}")
            env.close()
            return 1

        if not np.isfinite(reward):
            print(f"ERROR: NaN/Inf reward at step {step}")
            env.close()
            return 1

        print(f"\nStep {step + 1}")
        print("Action:", action)
        print("Observation:", obs)
        print("Reward:", reward)
        print("Weather mode:", info["weather_mode"])
        print("Weather source time:", info["weather_source_time"])
        print("PV power:", info["pv_power"])
        print("Terminated:", terminated)
        print("Truncated:", truncated)

        total_reward += reward

        if terminated or truncated:
            break

    env.close()

    print(f"\nTotal reward: {total_reward:.2f}")
    print("PV env with PVGIS TMY check passed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())