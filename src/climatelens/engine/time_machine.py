"""Climate Time Machine historical replay engine for ClimateLens India."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from climatelens.constants import PROCESSED_DATA_DIR, STATES_MAP

logger = logging.getLogger(__name__)


def get_time_machine_data(year: int) -> dict[str, Any]:
    """Retrieve chronological multi-hazard disaster events for historical animation and replay.

    Assembles:
    - Time-ordered cyclone tracks with coordinates, intensity, and pressure.
    - Historical flood events with start dates, duration, casualties, and affected states.
    - Recorded landslide occurrences with coordinate points and state labels.
    - Rainfall departure status if available for the selected year.
    """
    if not isinstance(year, int) or year < 1800 or year > 2100:
        return {
            "status": "no_data",
            "year": year,
            "message": f"Invalid year {year}. Must be an integer between 1800 and 2100.",
        }

    # 1. Cyclones for year
    cy_path = PROCESSED_DATA_DIR / "cyclone_track_points.parquet"
    cyclones_payload: list[dict[str, Any]] = []
    if cy_path.exists():
        cy_df = pd.read_parquet(cy_path)
        cy_year = cy_df[cy_df["season"] == year].sort_values(["sid", "iso_time"])
        for sid, group in cy_year.groupby("sid", sort=False):
            name = group.iloc[0]["name"]
            points = []
            for _, r in group.iterrows():
                points.append({
                    "iso_time": r["iso_time"],
                    "latitude": float(r["latitude"]),
                    "longitude": float(r["longitude"]),
                    "wmo_wind_kmh": float(r["wmo_wind_kmh"]) if pd.notna(r["wmo_wind_kmh"]) else None,
                    "usa_wind_kmh": float(r["usa_wind_kmh"]) if pd.notna(r["usa_wind_kmh"]) else None,
                    "central_pressure_hpa": float(r["wmo_pres_hpa"]) if pd.notna(r["wmo_pres_hpa"]) else (float(r["usa_pres_hpa"]) if pd.notna(r["usa_pres_hpa"]) else None),
                    "dist2land_km": float(r["dist2land_km"]) if pd.notna(r["dist2land_km"]) else None,
                    "is_interpolated": False,
                })
            wmo_max = group["wmo_wind_kmh"].dropna().max() if group["wmo_wind_kmh"].notna().any() else None
            usa_max = group["usa_wind_kmh"].dropna().max() if group["usa_wind_kmh"].notna().any() else None
            cyclones_payload.append({
                "storm_id": sid,
                "name": name,
                "season": year,
                "start_time": group["iso_time"].min(),
                "end_time": group["iso_time"].max(),
                "max_wind_kmh": round(float(wmo_max), 1) if wmo_max is not None else (round(float(usa_max), 1) if usa_max is not None else None),
                "observation_points_count": len(points),
                "track": points,
            })

    # 2. Floods for year
    fl_path = PROCESSED_DATA_DIR / "flood_events.parquet"
    floods_payload: list[dict[str, Any]] = []
    if fl_path.exists():
        fl_df = pd.read_parquet(fl_path)
        fl_year = fl_df[fl_df["year"] == year].sort_values("start_date", na_position="last")
        for _, r in fl_year.iterrows():
            floods_payload.append({
                "event_id": r["event_id"],
                "start_date": r["start_date"],
                "end_date": r["end_date"],
                "duration_days": float(r["duration_days"]) if pd.notna(r["duration_days"]) else None,
                "primary_state_id": r["primary_state_id"],
                "primary_state_name": r["primary_state_name"],
                "districts": r["districts"],
                "human_fatality": float(r["human_fatality"]) if pd.notna(r["human_fatality"]) else None,
                "human_displaced": float(r["human_displaced"]) if pd.notna(r["human_displaced"]) else None,
                "main_cause": r["main_cause"],
            })

    # 3. Landslides for year
    ls_path = PROCESSED_DATA_DIR / "landslide_events.parquet"
    landslides_payload: list[dict[str, Any]] = []
    if ls_path.exists():
        ls_df = pd.read_parquet(ls_path)
        ls_year = ls_df[ls_df["year"] == year].sort_values("state_id")
        for _, r in ls_year.iterrows():
            landslides_payload.append({
                "event_id": r["event_id"],
                "state_id": r["state_id"],
                "state_name": r["state_name"],
                "location": r["location"],
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "reported_magnitude": float(r["reported_magnitude"]),
            })

    # 4. Rainfall summary for year (if available)
    rf_path = PROCESSED_DATA_DIR / "rainfall_state_year.parquet"
    rainfall_payload: list[dict[str, Any]] = []
    if rf_path.exists():
        rf_df = pd.read_parquet(rf_path)
        rf_year = rf_df[rf_df["year"] == year]
        for _, r in rf_year.iterrows():
            rainfall_payload.append({
                "state_id": r["state_id"],
                "state_name": r["state_name"],
                "monsoon_actual_mm": float(r["monsoon_cumulative_actual_mm"]) if pd.notna(r["monsoon_cumulative_actual_mm"]) else None,
                "monsoon_departure_pct": float(r["monsoon_departure_pct"]) if pd.notna(r["monsoon_departure_pct"]) else None,
                "monsoon_category": r["monsoon_category"],
            })

    # 5. State heatwaves for year
    hw_path = PROCESSED_DATA_DIR / "state_heatwave.parquet"
    heatwaves_payload: list[dict[str, Any]] = []
    if hw_path.exists():
        hw_df = pd.read_parquet(hw_path)
        hw_year = hw_df[hw_df["year"] == year].sort_values("total_heat_wave_days", ascending=False)
        for _, r in hw_year.iterrows():
            if int(r["total_heat_wave_days"]) > 0 or pd.notna(r["summer_max_temp_deg_c"]):
                heatwaves_payload.append({
                    "state_id": r["state_id"],
                    "state_name": r["state_name"],
                    "heat_wave_days": int(r["heat_wave_days"]),
                    "severe_heat_wave_days": int(r["severe_heat_wave_days"]),
                    "total_heat_wave_days": int(r["total_heat_wave_days"]),
                    "summer_max_temp_deg_c": float(r["summer_max_temp_deg_c"]) if pd.notna(r["summer_max_temp_deg_c"]) else None,
                })

    total_events = len(cyclones_payload) + len(floods_payload) + len(landslides_payload) + len(heatwaves_payload)
    if total_events == 0 and not rainfall_payload:
        return {
            "status": "no_data",
            "year": year,
            "message": f"No historical hazard observations recorded in available datasets for year {year}.",
            "data": {
                "cyclones": [],
                "floods": [],
                "landslides": [],
                "rainfall": [],
                "heatwaves": [],
            },
        }

    return {
        "status": "ok",
        "year": year,
        "summary": {
            "cyclones_tracked_count": len(cyclones_payload),
            "flood_events_count": len(floods_payload),
            "landslide_incidents_count": len(landslides_payload),
            "states_with_rainfall_data": len(rainfall_payload),
            "heatwave_affected_states_count": len(heatwaves_payload),
            "total_disaster_events": total_events,
        },
        "data": {
            "cyclones": cyclones_payload,
            "floods": floods_payload,
            "landslides": landslides_payload,
            "rainfall": rainfall_payload,
            "heatwaves": heatwaves_payload,
        },
        "limitations": [
            "Cyclone tracks are North Indian Ocean best tracks (NOAA IBTrACS).",
            "Flood coordinates are not recorded in source; represented at state/district level.",
            "All coordinates provided are original observations without visual interpolation.",
        ],
    }
