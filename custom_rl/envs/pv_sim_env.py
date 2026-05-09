import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PVSimEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # state: [solar_altitude, solar_azimuth, panel_tilt, panel_azimuth, irradiance]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([90, 360, 90, 360, 1200], dtype=np.float32),
            dtype=np.float32,
        )

        # actions: 0 stay, 1 tilt up, 2 tilt down, 3 azimuth right, 4 azimuth left
        self.action_space = spaces.Discrete(5)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.array([30, 120, 20, 100, 800], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        solar_alt, solar_az, panel_tilt, panel_az, irradiance = self.state

        if action == 1:
            panel_tilt += 1
        elif action == 2:
            panel_tilt -= 1
        elif action == 3:
            panel_az += 1
        elif action == 4:
            panel_az -= 1

        panel_tilt = np.clip(panel_tilt, 0, 90)
        panel_az = np.clip(panel_az, 0, 360)

        angle_error = abs(solar_alt - panel_tilt) + abs(solar_az - panel_az) / 10
        reward = irradiance * max(0, 1 - angle_error / 100)

        self.state = np.array(
            [solar_alt, solar_az + 1, panel_tilt, panel_az, irradiance],
            dtype=np.float32,
        )

        terminated = solar_az >= 240
        truncated = False

        return self.state, reward, terminated, truncated, {}