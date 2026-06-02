"""Initial PV simulation environment scaffold for Gymnasium."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import pvlib
from gymnasium import spaces

from custom_rl.weather import (
    fetch_pvgis_tmy,
    normalize_weather_index,
    validate_weather_columns,
)


class PVSimEnv(gym.Env):
    """
    Initial Gymnasium environment for PV solar tracking.

    This environment uses pvlib to estimate PV output power from panel tilt
    and azimuth. It is still a scaffold and does not use a full real weather
    dataset yet.

    State:
        [solar_altitude, solar_azimuth, panel_tilt, panel_azimuth, poa_irradiance]

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
        max_episode_steps: int = 120,
        tilt_step_deg: float = 1.0,
        azimuth_step_deg: float = 1.0,
        latitude: float = 35.6892,
        longitude: float = 51.3890,
        timezone: str = "Asia/Tehran",
        altitude: float = 1200.0,
        panel_area: float = 1.6,
        panel_efficiency: float = 0.18,
        start_time: str = "2026-06-21 08:00",
        step_minutes: int = 5,
        weather_mode: str = "clearsky",
        pvgis_weather_data: pd.DataFrame | None = None,
    ) -> None:
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.tilt_step_deg = tilt_step_deg
        self.azimuth_step_deg = azimuth_step_deg

        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.altitude = altitude

        self.panel_area = panel_area
        self.panel_efficiency = panel_efficiency

        self.start_time = pd.Timestamp(start_time, tz=self.timezone)
        self.step_minutes = step_minutes

        self.location = pvlib.location.Location(
            latitude=self.latitude,
            longitude=self.longitude,
            tz=self.timezone,
            altitude=self.altitude,
            name="PVSimEnv location",
        )
        
        self.weather_mode = weather_mode.lower()
        self.weather_data: pd.DataFrame | None = None
        self.weather_metadata: dict[str, Any] | None = None
        self._last_weather_source_time: str | None = None

        if self.weather_mode not in {"clearsky", "pvgis_tmy"}:
            raise ValueError(
                "weather_mode must be either 'clearsky' or 'pvgis_tmy'. "
                f"Got: {weather_mode}"
            )

        if self.weather_mode == "pvgis_tmy":
            if pvgis_weather_data is None:
                pvgis_weather_data, self.weather_metadata = fetch_pvgis_tmy(
                    latitude=self.latitude,
                    longitude=self.longitude,
                )

            self.weather_data = normalize_weather_index(
                pvgis_weather_data,
                timezone=self.timezone,
            )
            validate_weather_columns(self.weather_data)
            self.weather_data = self.weather_data.sort_index()

        # state:
        # [solar_altitude, solar_azimuth, panel_tilt, panel_azimuth, poa_irradiance]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([90.0, 360.0, 90.0, 360.0, 1400.0], dtype=np.float32),
            dtype=np.float32,
        )

        # actions: 0 stay, 1 tilt up, 2 tilt down, 3 azimuth right, 4 azimuth left
        self.action_space = spaces.Discrete(5)

        self.state: np.ndarray | None = None
        self.step_count = 0
        self.current_time = self.start_time

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.step_count = 0
        self.current_time = self.start_time

        # Initial panel orientation.
        panel_tilt = 20.0
        panel_azimuth = 100.0

        solar_altitude, solar_azimuth, poa_irradiance, pv_power = (
            self._calculate_pv_power(
                panel_tilt=panel_tilt,
                panel_azimuth=panel_azimuth,
                timestamp=self.current_time,
            )
        )

        self.state = np.array(
            [
                solar_altitude,
                solar_azimuth,
                panel_tilt,
                panel_azimuth,
                poa_irradiance,
            ],
            dtype=np.float32,
        )

        info = {
            "time": str(self.current_time),
            "weather_mode": self.weather_mode,
            "weather_source_time": self._last_weather_source_time,  
            "solar_altitude": solar_altitude,
            "solar_azimuth": solar_azimuth,
            "panel_tilt": panel_tilt,
            "panel_azimuth": panel_azimuth,
            "poa_irradiance": poa_irradiance,
            "pv_power": pv_power,
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

        _, _, panel_tilt, panel_azimuth, _ = self.state

        if action == 1:
            panel_tilt += self.tilt_step_deg
        elif action == 2:
            panel_tilt -= self.tilt_step_deg
        elif action == 3:
            panel_azimuth += self.azimuth_step_deg
        elif action == 4:
            panel_azimuth -= self.azimuth_step_deg

        panel_tilt = float(np.clip(panel_tilt, 0.0, 90.0))
        panel_azimuth = float(panel_azimuth % 360.0)

        # Move simulation time forward.
        self.current_time = self.current_time + pd.Timedelta(minutes=self.step_minutes)

        solar_altitude, solar_azimuth, poa_irradiance, pv_power = (
            self._calculate_pv_power(
                panel_tilt=panel_tilt,
                panel_azimuth=panel_azimuth,
                timestamp=self.current_time,
            )
        )

        tilt_error = abs(solar_altitude - panel_tilt)
        azimuth_error = self._circular_angle_error(solar_azimuth, panel_azimuth)

        movement_penalty = 0.01 if action != 0 else 0.0

        # Main reward: estimated PV power from pvlib.
        reward = float(pv_power - movement_penalty)

        self.state = np.array(
            [
                solar_altitude,
                solar_azimuth,
                panel_tilt,
                panel_azimuth,
                poa_irradiance,
            ],
            dtype=np.float32,
        )

        self.step_count += 1

        # Episode ends if sun is below horizon or max steps reached.
        terminated = solar_altitude <= 0.0
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "time": str(self.current_time),
            "weather_mode": self.weather_mode,
            "weather_source_time": self._last_weather_source_time,
            "solar_altitude": solar_altitude,
            "solar_azimuth": solar_azimuth,
            "panel_tilt": panel_tilt,
            "panel_azimuth": panel_azimuth,
            "poa_irradiance": poa_irradiance,
            "pv_power": pv_power,
            "tilt_error": tilt_error,
            "azimuth_error": azimuth_error,
            "movement_penalty": movement_penalty,
        }

        return self.state.copy(), reward, terminated, truncated, info

    def _calculate_pv_power(
        self,
        panel_tilt: float,
        panel_azimuth: float,
        timestamp: pd.Timestamp,
    ) -> tuple[float, float, float, float]:
        """Estimate PV power using pvlib for a single timestamp."""
        times = pd.DatetimeIndex([timestamp])

        solar_position = self.location.get_solarposition(times)
        irradiance_data = self._get_irradiance_data(
            times=times,
            timestamp=timestamp,
        )

        apparent_elevation = float(solar_position["apparent_elevation"].iloc[0])
        solar_azimuth = float(solar_position["azimuth"].iloc[0])

        # If sun is below the horizon, no useful PV power is produced.
        if apparent_elevation <= 0.0:
            return 0.0, solar_azimuth, 0.0, 0.0

        poa_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=panel_tilt,
            surface_azimuth=panel_azimuth,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=irradiance_data["dni"],
            ghi=irradiance_data["ghi"],
            dhi=irradiance_data["dhi"],
        )

        poa_global = float(poa_irradiance["poa_global"].iloc[0])
        poa_global = max(0.0, poa_global)

        pv_power = poa_global * self.panel_area * self.panel_efficiency

        return apparent_elevation, solar_azimuth, poa_global, float(pv_power)

    def _get_irradiance_data(
        self,
        times: pd.DatetimeIndex,
        timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return irradiance data from either clear-sky pvlib or PVGIS TMY."""
        if self.weather_mode == "clearsky":
            self._last_weather_source_time = str(timestamp)
            return self.location.get_clearsky(times)[["ghi", "dni", "dhi"]]

        if self.weather_mode == "pvgis_tmy":
            weather_row = self._get_pvgis_weather_row(timestamp)
            self._last_weather_source_time = str(weather_row.name)

            # Important:
            # Re-index the selected TMY row to the simulation timestamp.
            # Otherwise pandas may align different years and produce NaN.
            return pd.DataFrame(
                {
                    "ghi": [float(weather_row["ghi"])],
                    "dni": [float(weather_row["dni"])],
                    "dhi": [float(weather_row["dhi"])],
                },
                index=times,
            )

        raise RuntimeError(f"Unsupported weather_mode: {self.weather_mode}")

    def _get_pvgis_weather_row(self, timestamp: pd.Timestamp) -> pd.Series:
        """Map simulation timestamp to the nearest matching PVGIS TMY row."""
        if self.weather_data is None:
            raise RuntimeError("PVGIS weather data is not loaded.")

        target = pd.Timestamp(timestamp)

        if target.tzinfo is None:
            target = target.tz_localize(self.timezone)
        else:
            target = target.tz_convert(self.timezone)

        weather_index = self.weather_data.index
        tmy_year = int(weather_index[0].year)

        # PVGIS TMY usually uses a representative year in the index.
        # We keep month/day/hour/minute from the simulation time, but map the
        # year to the TMY year.
        try:
            target = target.replace(year=tmy_year)
        except ValueError:
            # Handles Feb 29 if the TMY year is not a leap year.
            target = target.replace(year=tmy_year, day=28)

        if weather_index.tz is not None:
            target = target.tz_convert(weather_index.tz)
        else:
            target = target.tz_localize(None)

        nearest_position = weather_index.get_indexer([target], method="nearest")[0]
        return self.weather_data.iloc[nearest_position]
    
    @staticmethod
    def _circular_angle_error(angle_a: float, angle_b: float) -> float:
        """Return the minimum absolute difference between two angles in degrees."""
        diff = abs(angle_a - angle_b) % 360.0
        return min(diff, 360.0 - diff)