#!/usr/bin/env python3
"""Integrate NASA POWER daily weather data with USDM weekly drought data.

Pipeline:
1) NASA daily -> weekly mean (Tuesday-based week start to match USDM cadence)
2) USDM weekly cleanup
3) Merge by week and state

Outputs:
- Integrated_weekly_KAN.csv
- Integrated_weekly_NEB.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


NASA_FEATURES = [
    "allsky_sfc_sw_dwn",
    "prectotcorr",
    "ps",
    "rh2m",
    "t2m",
    "ws2m",
]

NASA_RENAME = {
    "allsky_sfc_sw_dwn": "ALLSKY_SFC_SW_DWN",
    "prectotcorr": "PRECTOTCORR",
    "ps": "PS",
    "rh2m": "RH2M",
    "t2m": "T2M",
    "ws2m": "WS2M",
}

USDM_COLS = [
    "FIPS",
    "County",
    "State",
    "None",
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "ValidStart",
    "ValidEnd",
]


@dataclass(frozen=True)
class StateConfig:
    code: str
    nasa_path: str
    usdm_path: str
    output_path: str
    county_selection_path: str


STATE_CONFIGS = [
    StateConfig(
        code="KS",
        nasa_path="NASA_POWER_daily_2010_2025_KAN.csv",
        usdm_path="KAN_dm_export_20100101_20251231.csv",
        output_path="Integrated_weekly_KAN.csv",
        county_selection_path="County_selection_KS.csv",
    ),
    StateConfig(
        code="NE",
        nasa_path="NASA_POWER_daily_2010_2025_NEB.csv",
        usdm_path="NEB_dm_export_20100101_20251231.csv",
        output_path="Integrated_weekly_NEB.csv",
        county_selection_path="County_selection_NE.csv",
    ),
]


def to_usdm_week_start(date_series: pd.Series) -> pd.Series:
    """Map dates to Tuesday week-start, consistent with USDM ValidStart cadence."""
    return date_series.dt.to_period("W-MON").dt.start_time


def prepare_nasa_weekly(nasa_csv: Path) -> pd.DataFrame:
    nasa = pd.read_csv(nasa_csv)
    nasa["date"] = pd.to_datetime(nasa["date"])
    nasa["week_start"] = to_usdm_week_start(nasa["date"])

    weekly = (
        nasa.groupby("week_start", as_index=False)[NASA_FEATURES]
        .mean(numeric_only=True)
        .rename(columns=NASA_RENAME)
    )
    return weekly


def prepare_usdm_weekly(usdm_csv: Path, state_code: str) -> pd.DataFrame:
    usdm = pd.read_csv(usdm_csv, usecols=USDM_COLS)
    usdm["ValidStart"] = pd.to_datetime(usdm["ValidStart"])
    usdm["ValidEnd"] = pd.to_datetime(usdm["ValidEnd"])
    usdm["week_start"] = usdm["ValidStart"]
    usdm = usdm[usdm["State"] == state_code].copy()
    return usdm


def select_counties(usdm_weekly: pd.DataFrame, target_count: int) -> pd.DataFrame:
    """Select counties deterministically by ascending FIPS to keep runs reproducible."""
    county_master = (
        usdm_weekly[["FIPS", "County"]]
        .drop_duplicates()
        .sort_values(["FIPS", "County"])
        .head(target_count)
        .reset_index(drop=True)
    )
    return county_master


def integrate_state(base_dir: Path, cfg: StateConfig, selected_fips: set[int]) -> pd.DataFrame:
    nasa_weekly = prepare_nasa_weekly(base_dir / cfg.nasa_path)
    usdm_weekly = prepare_usdm_weekly(base_dir / cfg.usdm_path, cfg.code)
    usdm_weekly = usdm_weekly[usdm_weekly["FIPS"].isin(selected_fips)].copy()

    merged = usdm_weekly.merge(nasa_weekly, on="week_start", how="left")
    merged = merged.sort_values(["FIPS", "week_start"]).reset_index(drop=True)

    merged["Year"] = merged["week_start"].dt.year
    merged["Month"] = merged["week_start"].dt.month
    merged["YearWeek"] = merged["week_start"].dt.strftime("%G%V")

    output_cols = [
        "week_start",
        "ValidEnd",
        "Year",
        "Month",
        "YearWeek",
        "FIPS",
        "County",
        "State",
        "ALLSKY_SFC_SW_DWN",
        "PRECTOTCORR",
        "PS",
        "RH2M",
        "T2M",
        "WS2M",
        "None",
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
    ]

    return merged[output_cols]


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    print("Integrating weekly NASA + USDM datasets...")

    # Build county master lists first, then align both states to the same county count.
    state_usdm = {
        cfg.code: prepare_usdm_weekly(base_dir / cfg.usdm_path, cfg.code) for cfg in STATE_CONFIGS
    }
    county_counts = {
        cfg.code: state_usdm[cfg.code]["FIPS"].nunique() for cfg in STATE_CONFIGS
    }
    target_count = min(county_counts.values())

    print(f"County alignment target: {target_count} counties per state")

    selected_by_state: dict[str, pd.DataFrame] = {}
    for cfg in STATE_CONFIGS:
        selected = select_counties(state_usdm[cfg.code], target_count)
        selected_by_state[cfg.code] = selected

        selection_path = base_dir / cfg.county_selection_path
        selected.to_csv(selection_path, index=False)
        print(f"[{cfg.code}] county list: {cfg.county_selection_path} ({len(selected):,} counties)")

    for cfg in STATE_CONFIGS:
        selected_fips = set(selected_by_state[cfg.code]["FIPS"].astype(int).tolist())
        integrated = integrate_state(base_dir, cfg, selected_fips)
        out_path = base_dir / cfg.output_path
        integrated.to_csv(out_path, index=False)

        missing_nasa_rows = integrated[
            integrated[[
                "ALLSKY_SFC_SW_DWN",
                "PRECTOTCORR",
                "PS",
                "RH2M",
                "T2M",
                "WS2M",
            ]].isna().any(axis=1)
        ].shape[0]

        print(f"[{cfg.code}] rows: {len(integrated):,}")
        print(f"[{cfg.code}] unique weeks: {integrated['week_start'].nunique():,}")
        print(f"[{cfg.code}] unique counties: {integrated['FIPS'].nunique():,}")
        print(f"[{cfg.code}] rows with missing NASA features: {missing_nasa_rows:,}")
        print(f"[{cfg.code}] output: {cfg.output_path}")


if __name__ == "__main__":
    main()