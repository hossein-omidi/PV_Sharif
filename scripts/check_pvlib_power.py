"""Check a simple pvlib-based PV power calculation."""

from __future__ import annotations

import sys

import pandas as pd
import pvlib


def main() -> int:
    # Example location: Tehran
    latitude = 35.6892
    longitude = 51.3890
    timezone = "Asia/Tehran"

    # Example timestamp
    times = pd.DatetimeIndex(["2026-06-21 12:00"], tz=timezone)

    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=timezone,
        altitude=1200,
        name="Tehran",
    )

    solar_position = location.get_solarposition(times)

    # Simple clear-sky irradiance model
    clear_sky = location.get_clearsky(times)

    # Example panel orientation
    panel_tilt = 30.0
    panel_azimuth = 180.0

    poa_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=panel_tilt,
        surface_azimuth=panel_azimuth,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clear_sky["dni"],
        ghi=clear_sky["ghi"],
        dhi=clear_sky["dhi"],
    )

    # Very simple power approximation
    panel_area = 1.6  # m^2
    efficiency = 0.18

    poa_global = float(poa_irradiance["poa_global"].iloc[0])
    power = poa_global * panel_area * efficiency

    print("Solar position:")
    print(solar_position)

    print("\nClear sky irradiance:")
    print(clear_sky)

    print("\nPOA irradiance:")
    print(poa_irradiance)

    print(f"\nEstimated PV power: {power:.2f} W")

    return 0


if __name__ == "__main__":
    sys.exit(main())