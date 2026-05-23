"""Train a DQN agent on the initial PVSimEnv scaffold."""

from __future__ import annotations

import os
import sys

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from custom_rl import register_envs


def main() -> int:
    register_envs()

    env = gym.make("PVSimEnv-v0")
    env = Monitor(env)

    os.makedirs("models/dqn_pvsim", exist_ok=True)
    os.makedirs("logs/dqn_pvsim", exist_ok=True)

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
        tensorboard_log="logs/dqn_pvsim",
        seed=42,
    )

    print("Training DQN on PVSimEnv-v0...")
    model.learn(total_timesteps=5_000)

    model_path = "models/dqn_pvsim/dqn_pvsim_model"
    model.save(model_path)

    env.close()

    print(f"Training finished. Model saved to: {model_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())