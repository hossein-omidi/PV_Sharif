"""Evaluate the trained DQN agent on the initial PVSimEnv scaffold."""

from __future__ import annotations

import os
import sys

import gymnasium as gym
from stable_baselines3 import DQN

from custom_rl import register_envs


def main() -> int:
    register_envs()

    model_path = "models/dqn_pvsim/dqn_pvsim_model.zip"

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Run scripts/train_dqn_pvsim.py first.")
        return 1

    env = gym.make("PVSimEnv-v0")
    model = DQN.load(model_path, env=env)

    obs, info = env.reset(seed=42)

    total_reward = 0.0

    print("Evaluating DQN on PVSimEnv-v0...")
    print("Initial observation:", obs)

    for step in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        total_reward += reward

        print(f"\nStep {step + 1}")
        print("Action:", int(action))
        print("Observation:", obs)
        print("Reward:", reward)
        print("Info:", info)
        print("Terminated:", terminated)
        print("Truncated:", truncated)

        if terminated or truncated:
            break

    env.close()

    print(f"\nTotal evaluation reward: {total_reward:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())