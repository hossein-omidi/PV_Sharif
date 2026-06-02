"""Verification test for PVGIS TMY weather data and PV power calculation."""

from __future__ import annotations

import argparse
import sys

from custom_rl.weather import (
    estimate_pv_power_from_weather,
    fetch_pvgis_tmy,
    make_offline_sample_tmy,
    normalize_weather_index,
    pick_daylight_row,
    validate_weather_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check PVGIS TMY weather data and estimate PV power."
    )

    parser.add_argument("--latitude", type=float, default=35.6892)
    parser.add_argument("--longitude", type=float, default=51.3890)
    parser.add_argument("--timezone", type=str, default="Asia/Tehran")
    parser.add_argument("--altitude", type=float, default=1200.0)
    parser.add_argument("--panel-tilt", type=float, default=30.0)
    parser.add_argument("--panel-azimuth", type=float, default=180.0)

    parser.add_argument(
        "--sample-time",
        type=str,
        default=None,
        help="Optional local timestamp. Example: '2026-06-21 12:00'.",
    )

    parser.add_argument(
        "--offline-sample",
        action="store_true",
        help="Use synthetic local data instead of downloading from PVGIS.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("PVGIS TMY verification test")
    print("--------------------------")

    if args.offline_sample:
        print("Using offline synthetic sample data.")
        weather = make_offline_sample_tmy(timezone=args.timezone)
        metadata = {"source": "offline synthetic sample"}
    else:
        print("Fetching PVGIS TMY weather data...")

        weather, metadata = fetch_pvgis_tmy(
            latitude=args.latitude,
            longitude=args.longitude,
        )

        weather = normalize_weather_index(weather, timezone=args.timezone)

    validate_weather_columns(weather)

    print("\nWeather data loaded.")
    print(f"Shape: {weather.shape}")

    print("Columns:")
    print(list(weather.columns))

    sample_weather = pick_daylight_row(weather, sample_time=args.sample_time)
    sample_time = sample_weather.index[0]

    print("\nSample time:")
    print(sample_time)

    print("\nSample weather row:")
    print(sample_weather)

    result = estimate_pv_power_from_weather(
        sample_weather,
        latitude=args.latitude,
        longitude=args.longitude,
        timezone=args.timezone,
        altitude=args.altitude,
        panel_tilt=args.panel_tilt,
        panel_azimuth=args.panel_azimuth,
    )

    print("\nEstimated values from PVGIS TMY weather:")
    print(f"Solar altitude: {result['solar_altitude']:.2f} deg")
    print(f"Solar azimuth:  {result['solar_azimuth']:.2f} deg")
    print(f"POA global:     {result['poa_global']:.2f} W/m^2")
    print(f"PV power:       {result['pv_power']:.2f} W")

    print("\nMetadata keys:")
    print(list(metadata.keys()) if isinstance(metadata, dict) else type(metadata))

    if result["pv_power"] < 0.0:
        raise RuntimeError("PV power must not be negative.")

    print("\nPVGIS TMY verification passed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())