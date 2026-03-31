#!/usr/bin/env python3
"""Fetch daily NASA POWER AG data for predefined coordinate points."""

import argparse
import csv
import time
from typing import Dict, List, Tuple

import requests

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",
    "PRECTOTCORR",
    "PS",
    "RH2M",
    "T2M",
    "WS2M",
]

# (latitude, longitude)
COORDINATES: List[Tuple[float, float]] = [
    (39.35, -101.71),
    (39.40, -101.05),
    (37.97, -100.87),
    (37.75, -100.02),
    (37.04, -100.92),
    (38.48, -100.90),
    (38.88, -99.32),
    (38.36, -98.76),
    (38.84, -97.61),
    (38.06, -97.93),
    (38.37, -97.66),
    (37.69, -97.33),
    (39.57, -97.66),
    (39.18, -96.57),
    (39.06, -95.69),
    (38.40, -96.18),
    (38.62, -95.27),
    (37.68, -95.46),
    (37.34, -95.26),
    (39.85, -95.53),
]

OUTPUT_COLUMNS = [
    "allsky_sfc_sw_dwn",
    "prectotcorr",
    "ps",
    "rh2m",
    "t2m",
    "ws2m",
    "latitude",
    "longitude",
    "year",
    "month",
    "date",
]


def to_float_or_none(value):
    """Convert NASA missing values to None."""
    if value in (-999, -999.0, -99, -99.0, None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_point_data(
    session: requests.Session,
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int,
    retries: int = 4,
    timeout: int = 60,
) -> List[Dict]:
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON",
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(BASE_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()

            parameter_data = payload["properties"]["parameter"]
            any_param = PARAMETERS[0]
            dates = sorted(parameter_data[any_param].keys())

            rows = []
            for date_key in dates:
                year = int(date_key[:4])
                month = int(date_key[4:6])
                date_str = f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:8]}"

                row = {
                    "allsky_sfc_sw_dwn": to_float_or_none(
                        parameter_data["ALLSKY_SFC_SW_DWN"].get(date_key)
                    ),
                    "prectotcorr": to_float_or_none(
                        parameter_data["PRECTOTCORR"].get(date_key)
                    ),
                    "ps": to_float_or_none(parameter_data["PS"].get(date_key)),
                    "rh2m": to_float_or_none(parameter_data["RH2M"].get(date_key)),
                    "t2m": to_float_or_none(parameter_data["T2M"].get(date_key)),
                    "ws2m": to_float_or_none(parameter_data["WS2M"].get(date_key)),
                    "latitude": latitude,
                    "longitude": longitude,
                    "year": year,
                    "month": month,
                    "date": date_str,
                }
                rows.append(row)

            return rows
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                sleep_seconds = attempt * 2
                print(
                    f"[WARN] Gagal ambil data lat={latitude}, lon={longitude} "
                    f"(attempt {attempt}/{retries}): {exc}. Retry {sleep_seconds}s..."
                )
                time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Gagal ambil data untuk lat={latitude}, lon={longitude} setelah {retries} percobaan"
    ) from last_error


def main():
    parser = argparse.ArgumentParser(
        description="Download NASA POWER daily AG data for predefined coordinates."
    )
    parser.add_argument("--start-year", type=int, default=2010, help="Default: 2010")
    parser.add_argument("--end-year", type=int, default=2025, help="Default: 2025")
    parser.add_argument(
        "--output",
        type=str,
        default="NASA_POWER_daily_2010_2025_KAN.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.25,
        help="Pause (seconds) between points to reduce API pressure. Default: 0.25",
    )
    args = parser.parse_args()

    if args.start_year > args.end_year:
        raise ValueError("start-year harus <= end-year")

    print(
        f"Mulai unduh NASA POWER harian untuk {len(COORDINATES)} titik "
        f"({args.start_year}-{args.end_year})..."
    )

    all_rows: List[Dict] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "nasa-power-daily-fetcher/1.0"})
        for idx, (lat, lon) in enumerate(COORDINATES, start=1):
            print(f"[{idx}/{len(COORDINATES)}] Mengambil lat={lat}, lon={lon}")
            point_rows = fetch_point_data(
                session=session,
                latitude=lat,
                longitude=lon,
                start_year=args.start_year,
                end_year=args.end_year,
            )
            all_rows.extend(point_rows)
            time.sleep(args.pause)

    all_rows.sort(key=lambda row: (row["latitude"], row["longitude"], row["date"]))
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Selesai. Total baris: {len(all_rows):,}")
    print(f"File tersimpan: {args.output}")


if __name__ == "__main__":
    main()
