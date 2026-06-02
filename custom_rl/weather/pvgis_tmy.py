"""Utilities for using PVGIS TMY weather data with pvlib."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pvlib

REQUIRED_IRRADIANCE_COLUMNS = ("ghi", "dni", "dhi")


def fetch_pvgis_tmy(
    latitude: float,
    longitude: float,
    *,
    outputformat: str = "json",
    map_variables: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetch a Typical Meteorological Year dataset from PVGIS.

    With map_variables=True, pvlib maps PVGIS column names to pvlib names,
    such as ghi, dni, dhi, temp_air, and wind_speed.
    """
    weather, metadata = pvlib.iotools.get_pvgis_tmy(
        latitude=latitude,
        longitude=longitude,
        outputformat=outputformat,
        map_variables=map_variables,
    )
    return weather, metadata


def normalize_weather_index(weather: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Return a copy of weather data with a timezone-aware DatetimeIndex."""
    if not isinstance(weather.index, pd.DatetimeIndex):
        raise TypeError("weather must use a pandas DatetimeIndex")

    normalized = weather.copy()

    if normalized.index.tz is None:
        normalized.index = normalized.index.tz_localize("UTC")

    normalized.index = normalized.index.tz_convert(timezone)
    return normalized


def validate_weather_columns(weather: pd.DataFrame) -> None:
    """Validate that weather has the irradiance columns needed for POA."""
    missing = [col for col in REQUIRED_IRRADIANCE_COLUMNS if col not in weather.columns]
    if missing:
        raise ValueError(
            "PVGIS TMY weather is missing required columns: "
            + ", ".join(missing)
        )


def pick_daylight_row(
    weather: pd.DataFrame,
    *,
    sample_time: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Pick one weather row for a verification calculation.

    If sample_time is not provided, this picks the row with maximum GHI so the
    test is unlikely to accidentally choose nighttime data.
    """
    validate_weather_columns(weather)

    if sample_time is None:
        idx = weather["ghi"].astype(float).idxmax()
        return weather.loc[[idx]]

    timestamp = pd.Timestamp(sample_time)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(weather.index.tz)
    else:
        timestamp = timestamp.tz_convert(weather.index.tz)

    nearest_position = weather.index.get_indexer([timestamp], method="nearest")[0]
    return weather.iloc[[nearest_position]]


def estimate_pv_power_from_weather(
    weather_row: pd.DataFrame | pd.Series,
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    altitude: float,
    panel_tilt: float,
    panel_azimuth: float,
    panel_area: float = 1.6,
    panel_efficiency: float = 0.18,
) -> dict[str, float]:
    """Estimate POA irradiance and simple PV power for one weather row."""
    if isinstance(weather_row, pd.Series):
        weather = weather_row.to_frame().T
    else:
        weather = weather_row.copy()

    if len(weather) != 1:
        raise ValueError("weather_row must contain exactly one row")

    validate_weather_columns(weather)

    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=timezone,
        altitude=altitude,
        name="PVGIS TMY location",
    )

    solar_position = location.get_solarposition(weather.index)

    apparent_elevation = float(solar_position["apparent_elevation"].iloc[0])
    solar_azimuth = float(solar_position["azimuth"].iloc[0])

    if apparent_elevation <= 0.0:
        return {
            "solar_altitude": 0.0,
            "solar_azimuth": solar_azimuth,
            "poa_global": 0.0,
            "pv_power": 0.0,
        }

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=panel_tilt,
        surface_azimuth=panel_azimuth,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=weather["dni"].astype(float),
        ghi=weather["ghi"].astype(float),
        dhi=weather["dhi"].astype(float),
    )

    poa_global = float(poa["poa_global"].iloc[0])

    if np.isnan(poa_global):
        poa_global = 0.0

    poa_global = max(0.0, poa_global)
    pv_power = poa_global * panel_area * panel_efficiency

    return {
        "solar_altitude": apparent_elevation,
        "solar_azimuth": solar_azimuth,
        "poa_global": poa_global,
        "pv_power": float(pv_power),
    }


def make_offline_sample_tmy(timezone: str = "Asia/Tehran") -> pd.DataFrame:
    """Create a small synthetic weather sample for offline verification."""
    times = pd.date_range("2026-06-21 00:00", periods=24, freq="h", tz=timezone)
    hours = np.arange(24)

    daylight_shape = np.maximum(0.0, np.sin((hours - 5) / 14 * np.pi))

    weather = pd.DataFrame(
        {
            "ghi": 900.0 * daylight_shape,
            "dni": 750.0 * daylight_shape,
            "dhi": 150.0 * daylight_shape,
            "temp_air": 25.0 + 10.0 * daylight_shape,
            "wind_speed": 2.0 + 1.5 * daylight_shape,
            "relative_humidity": 35.0 + 10.0 * (1.0 - daylight_shape),
            "pressure": 101325.0,
        },
        index=times,
    )

    return weather