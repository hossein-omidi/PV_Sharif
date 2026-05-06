"""Verify custom RL environment with Gymnasium check_env and short rollout."""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from custom_rl import register_envs


def main() -> int:
    parser = argparse.ArgumentParser(description="Check custom RL env with Gymnasium.")
    parser.add_argument("--reward", default="sparse", choices=["dense", "sparse"])
    parser.add_argument("--steps", type=int, default=50, help="Rollout steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    register_envs()

    env = gym.make("CustomODECartPole-v0", reward_id=args.reward)
    raw_env = env.unwrapped

    print("Environment spaces:")
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space:      {env.action_space}")

    print("\nChecking action space for DQN compatibility...")
    if not isinstance(env.action_space, spaces.Discrete):
        print("ERROR: DQN requires a discrete action space.")
        print(f"Found action space: {env.action_space}")
        env.close()
        return 1

    if env.action_space.n != 2:
        print("ERROR: Expected Discrete(2) action space for left/right CartPole.")
        print(f"Found action space: {env.action_space}")
        env.close()
        return 1

    print("  Action space OK: Discrete(2)")
    print("  Action 0 -> left")
    print("  Action 1 -> right")

    print("\nRunning Gymnasium check_env (skip_render_check=True)...")
    check_env(raw_env, skip_render_check=True)
    print("check_env passed.")

    print("\nTesting manual discrete actions...")
    obs, info = env.reset(seed=args.seed)

    for action in [0, 1]:
        obs, reward, terminated, truncated, info = env.step(action)

        if not np.isfinite(obs).all():
            print(f"ERROR: NaN/Inf in observation after action {action}")
            env.close()
            return 1

        if not np.isfinite(reward):
            print(f"ERROR: NaN/Inf reward after action {action}")
            env.close()
            return 1

        if not env.observation_space.contains(obs):
            print(f"ERROR: Observation is outside observation_space after action {action}")
            print(f"obs: {obs}")
            env.close()
            return 1

        print(
            f"  action={action}, reward={reward:.4f}, "
            f"terminated={terminated}, truncated={truncated}"
        )

        if terminated or truncated:
            obs, info = env.reset(seed=args.seed)

    print("Manual discrete action test passed.")

    print("\nShort rollout with random discrete actions...")
    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0

    for step in range(args.steps):
        action = env.action_space.sample()

        if not env.action_space.contains(action):
            print(f"ERROR: Sampled invalid action at step {step}: {action}")
            env.close()
            return 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if not np.isfinite(obs).all():
            print(f"ERROR: Step {step}: NaN/Inf in observation")
            env.close()
            return 1

        if not np.isfinite(reward):
            print(f"ERROR: Step {step}: NaN/Inf reward")
            env.close()
            return 1

        if not env.observation_space.contains(obs):
            print(f"ERROR: Step {step}: observation outside observation_space")
            print(f"obs: {obs}")
            env.close()
            return 1

        if terminated or truncated:
            obs, info = env.reset()

    print(f"  Steps: {args.steps}, total_reward: {total_reward:.2f}")
    print("Random rollout test passed.")

    env.close()

    print("\nDeterminism check: same seed -> same first observation...")
    env1 = gym.make("CustomODECartPole-v0", reward_id=args.reward)
    env2 = gym.make("CustomODECartPole-v0", reward_id=args.reward)

    o1, _ = env1.reset(seed=args.seed)
    o2, _ = env2.reset(seed=args.seed)

    env1.close()
    env2.close()

    if np.allclose(o1, o2):
        print("  Deterministic reset OK.")
    else:
        print("ERROR: Reset is non-deterministic with the same seed.")
        print(f"o1: {o1}")
        print(f"o2: {o2}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())