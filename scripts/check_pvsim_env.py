from custom_rl.envs.pv_sim_env import PVSimEnv

env = PVSimEnv()

obs, info = env.reset()
print("Initial observation:", obs)

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {i+1}")
    print("Action:", action)
    print("Observation:", obs)
    print("Reward:", reward)
    print("-" * 30)

    if terminated or truncated:
        break