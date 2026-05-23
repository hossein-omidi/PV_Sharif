"""Initial PV simulation environment scaffold for Gymnasium."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PVSimEnv(gym.Env):
    """
    Initial Gymnasium environment for PV solar tracking.

    This environment is only a scaffold for now. It does not call the real
    pvsim yet. The temporary reward is based on the alignment between the
    panel orientation and the sun position.

    State:
        [solar_altitude, solar_azimuth, panel_tilt, panel_azimuth, irradiance]

    Actions:
        0: stay
        1: increase panel tilt
        2: decrease panel tilt
        3: increase panel azimuth
        4: decrease panel azimuth
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_steps: int = 500,
        tilt_step_deg: float = 1.0,
        azimuth_step_deg: float = 1.0,
    ) -> None:
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.tilt_step_deg = tilt_step_deg
        self.azimuth_step_deg = azimuth_step_deg

        # state: [solar_altitude, solar_azimuth, panel_tilt, panel_azimuth, irradiance]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([90.0, 360.0, 90.0, 360.0, 1200.0], dtype=np.float32),
            dtype=np.float32,
        )

        # actions: 0 stay, 1 tilt up, 2 tilt down, 3 azimuth right, 4 azimuth left
        self.action_space = spaces.Discrete(5)

        self.state: np.ndarray | None = None
        self.step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.step_count = 0

        # Initial sample condition for the scaffold version.
        # Later this should come from pvsim / weather / solar position data.
        self.state = np.array(
            [30.0, 120.0, 20.0, 100.0, 800.0],
            dtype=np.float32,
        )

        info = {
            "description": "Initial PVSimEnv scaffold reset.",
        }

        return self.state.copy(), info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        solar_altitude, solar_azimuth, panel_tilt, panel_azimuth, irradiance = self.state

        if action == 1:
            panel_tilt += self.tilt_step_deg
        elif action == 2:
            panel_tilt -= self.tilt_step_deg
        elif action == 3:
            panel_azimuth += self.azimuth_step_deg
        elif action == 4:
            panel_azimuth -= self.azimuth_step_deg

        panel_tilt = float(np.clip(panel_tilt, 0.0, 90.0))
        panel_azimuth = float(np.clip(panel_azimuth, 0.0, 360.0))

        tilt_error = abs(float(solar_altitude) - panel_tilt)
        azimuth_error = self._circular_angle_error(float(solar_azimuth), panel_azimuth)

        # Temporary alignment score.
        # This is not the final PV model. Later it should be replaced
        # with actual output power returned by pvsim.
        normalized_error = (tilt_error / 90.0) + (azimuth_error / 180.0)
        alignment_score = max(0.0, 1.0 - normalized_error)

        movement_penalty = 0.01 if action != 0 else 0.0
        reward = float(irradiance * alignment_score - movement_penalty)

        # Temporary artificial sun motion.
        # Later this should be replaced with real solar position / pvsim data.
        next_solar_azimuth = float(np.clip(solar_azimuth + 1.0, 0.0, 360.0))

        self.state = np.array(
            [
                solar_altitude,
                next_solar_azimuth,
                panel_tilt,
                panel_azimuth,
                irradiance,
            ],
            dtype=np.float32,
        )

        self.step_count += 1

        terminated = next_solar_azimuth >= 240.0
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "solar_altitude": float(solar_altitude),
            "solar_azimuth": next_solar_azimuth,
            "panel_tilt": panel_tilt,
            "panel_azimuth": panel_azimuth,
            "irradiance": float(irradiance),
            "tilt_error": tilt_error,
            "azimuth_error": azimuth_error,
            "alignment_score": alignment_score,
            "movement_penalty": movement_penalty,
        }

        return self.state.copy(), reward, terminated, truncated, info

    @staticmethod
    def _circular_angle_error(angle_a: float, angle_b: float) -> float:
        """Return the minimum absolute difference between two angles in degrees."""
        diff = abs(angle_a - angle_b) % 360.0
        return min(diff, 360.0 - diff)
