"""Cyclone track, intensity, and coastline proximity analysis for ClimateLens India."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nielytics import AnalysisResult
from nielytics.climate.providers.noaa_ibtracs import cyclone_tracks_to_geojson
from nielytics.hazards.cyclone import CycloneAnalyzer
from climatelens.constants import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

SCIENTIFIC_CAVEAT = (
    "NOAA IBTrACS v04r01 North Indian Ocean basin. Proximity to coastline does NOT constitute "
    "verified meteorological landfall. WMO and USA wind metrics use different averaging periods "
    "and should not be directly combined."
)


def _load_cyclone_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "cyclone_track_points.parquet"
    if not path.exists():
        from climatelens.preprocessing.cyclones import process_cyclones
        return process_cyclones()
    return pd.read_parquet(path)


def get_storm_track(storm_id_or_name: str) -> dict[str, Any]:
    """Retrieve chronologically ordered track observations for a specific storm."""
    df = _load_cyclone_data()
    query = storm_id_or_name.strip().upper()

    # Search by SID or Name
    matched = df[(df["sid"] == query) | (df["name"] == query)].sort_values("iso_time").copy()

    if matched.empty:
        return {
            "status": "no_data",
            "query": storm_id_or_name,
            "message": f"Storm '{storm_id_or_name}' not found in North Indian Ocean IBTrACS dataset.",
        }

    sid = str(matched.iloc[0]["sid"])
    name = str(matched.iloc[0]["name"])
    season = int(matched.iloc[0]["season"])

    # Calculate track distance and intensity using NIElytics CycloneAnalyzer
    ca_df = pd.DataFrame({
        "storm_id": matched["sid"],
        "time": pd.to_datetime(matched["iso_time"]),
        "latitude": matched["latitude"],
        "longitude": matched["longitude"],
        "wind": matched["wmo_wind_kmh"].fillna(0.0),
        "pressure": matched["wmo_pres_hpa"],
    }).drop_duplicates(subset=["storm_id", "time"])

    distance_km = None
    max_wind_kmh = None
    min_pressure_hpa = None

    try:
        ca = CycloneAnalyzer(ca_df, wind_unit="km/h", pressure_unit="hPa")
        dist_res = ca.track_distance()
        int_res = ca.cyclone_intensity()
        if dist_res.data.get("storms"):
            distance_km = round(dist_res.data["storms"][0]["distance_km"], 1)
        if int_res.data.get("storms"):
            max_wind_kmh = int_res.data["storms"][0]["max_wind_kmh"]
            min_pressure_hpa = int_res.data["storms"][0]["min_pressure_hpa"]
    except Exception as exc:
        logger.debug("CycloneAnalyzer calculation notice: %s", exc)

    # If WMO wind was missing, check USA wind
    if max_wind_kmh is None or max_wind_kmh == 0.0:
        usa_winds = matched["usa_wind_kmh"].dropna()
        if not usa_winds.empty:
            max_wind_kmh = float(usa_winds.max())

    # Build ordered track points
    track_points = []
    for _, r in matched.iterrows():
        track_points.append({
            "iso_time": r["iso_time"],
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "wmo_wind_kmh": float(r["wmo_wind_kmh"]) if pd.notna(r["wmo_wind_kmh"]) else None,
            "usa_wind_kmh": float(r["usa_wind_kmh"]) if pd.notna(r["usa_wind_kmh"]) else None,
            "central_pressure_hpa": float(r["wmo_pres_hpa"]) if pd.notna(r["wmo_pres_hpa"]) else (float(r["usa_pres_hpa"]) if pd.notna(r["usa_pres_hpa"]) else None),
            "distance_to_coast_km": float(r["dist2land_km"]) if pd.notna(r["dist2land_km"]) else None,
            "landfall_flag": int(r["landfall"]) if pd.notna(r["landfall"]) else None,
            "is_interpolated": bool(r["is_interpolated"]),
        })

    # Find closest observed approach to Indian coastline
    min_dist_to_coast = matched["dist2land_km"].dropna().min() if matched["dist2land_km"].notna().any() else None

    return {
        "status": "ok",
        "storm_id": sid,
        "name": name,
        "season": season,
        "basin": "NI",
        "observation_count": len(matched),
        "start_time": matched["iso_time"].min(),
        "end_time": matched["iso_time"].max(),
        "total_track_distance_km": distance_km,
        "peak_sustained_wind_kmh": max_wind_kmh,
        "min_central_pressure_hpa": min_pressure_hpa,
        "min_distance_to_coast_km": float(min_dist_to_coast) if min_dist_to_coast is not None else None,
        "track_points": track_points,
        "caveat": SCIENTIFIC_CAVEAT,
    }


def analyze_cyclone_hazard(
    year: int | None = None,
    min_wind_kmh: float | None = None,
    coastal_within_km: float | None = None,
) -> dict[str, Any]:
    """List and summarize historical storms in the North Indian Ocean basin."""
    df = _load_cyclone_data()

    filtered = df.copy()
    if year is not None:
        filtered = filtered[filtered["season"] == year]
    if min_wind_kmh is not None:
        filtered = filtered[(filtered["wmo_wind_kmh"] >= min_wind_kmh) | (filtered["usa_wind_kmh"] >= min_wind_kmh)]
    if coastal_within_km is not None:
        filtered = filtered[filtered["dist2land_km"] <= coastal_within_km]

    if filtered.empty:
        return {
            "status": "no_data",
            "year": year,
            "message": "No cyclone tracks match the specified filter criteria.",
        }

    # Group by storm
    storms_summary = []
    for sid, group in filtered.groupby("sid", sort=False):
        name = group.iloc[0]["name"]
        season = int(group.iloc[0]["season"])
        wmo_max = group["wmo_wind_kmh"].dropna().max() if group["wmo_wind_kmh"].notna().any() else None
        usa_max = group["usa_wind_kmh"].dropna().max() if group["usa_wind_kmh"].notna().any() else None
        min_pres = group["wmo_pres_hpa"].dropna().min() if group["wmo_pres_hpa"].notna().any() else None
        min_dist = group["dist2land_km"].dropna().min() if group["dist2land_km"].notna().any() else None

        storms_summary.append({
            "storm_id": sid,
            "name": name,
            "season": season,
            "observation_count": len(group),
            "start_time": group["iso_time"].min(),
            "end_time": group["iso_time"].max(),
            "max_wmo_wind_kmh": round(float(wmo_max), 1) if wmo_max is not None else None,
            "max_usa_wind_kmh": round(float(usa_max), 1) if usa_max is not None else None,
            "min_pressure_hpa": round(float(min_pres), 1) if min_pres is not None else None,
            "min_distance_to_coast_km": round(float(min_dist), 1) if min_dist is not None else None,
        })

    # Sort storms by season descending, then max wind
    storms_summary.sort(key=lambda s: (s["season"], s["max_wmo_wind_kmh"] or 0), reverse=True)
    # NIElytics CycloneAnalyzer frequency
    freq_by_year = []
    try:
        freq_ca_df = pd.DataFrame({
            "storm_id": filtered["sid"],
            "time": filtered["iso_time"],
            "latitude": filtered["latitude"],
            "longitude": filtered["longitude"],
            "wind": filtered["wmo_wind_kmh"].fillna(0.0),
            "pressure": filtered["wmo_pres_hpa"].fillna(1013.0),
        }).drop_duplicates(subset=["storm_id", "time"])
        if not freq_ca_df.empty:
            ca_freq = CycloneAnalyzer(freq_ca_df, wind_unit="km/h", pressure_unit="hPa")
            freq_res = ca_freq.cyclone_frequency()
            freq_by_year = freq_res.data.get("years", [])
    except Exception as exc:
        logger.debug("CycloneAnalyzer cyclone_frequency notice: %s", exc)

    return {
        "status": "ok",
        "year": year,
        "total_storms_count": len(storms_summary),
        "total_track_points_count": len(filtered),
        "frequency_by_year": freq_by_year,
        "storms": storms_summary,
        "caveat": SCIENTIFIC_CAVEAT,
    }


def get_cyclone_tracks_geojson(year: int | None = None, storm_id: str | None = None) -> dict[str, Any]:
    """Generate authoritative GeoJSON LineString tracks using nielytics.climate.providers.noaa_ibtracs."""
    df = _load_cyclone_data()
    filtered = df.copy()
    if year is not None:
        filtered = filtered[filtered["season"] == year]
    if storm_id is not None:
        q = storm_id.strip().upper()
        filtered = filtered[(filtered["sid"] == q) | (filtered["name"] == q)]

    records = []
    for sid, group in filtered.groupby("sid", sort=False):
        last_pt = None
        for _, row in group.iterrows():
            pt = (float(row["latitude"]), float(row["longitude"]))
            if pt == last_pt:
                continue
            last_pt = pt
            records.append({
                "sid": str(row["sid"]),
                "time": str(row["iso_time"]),
                "name": str(row["name"]),
                "latitude": pt[0],
                "longitude": pt[1],
            })
    res = AnalysisResult(
        feature="ibtracs_observations",
        data={"records": records},
        metadata={
            "source": "NOAA IBTrACS v04r01",
            "source_url": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs",
            "retrieved_at": "2024-01-01",
            "cached": True,
            "coverage": {
                "start": str(filtered["iso_time"].min()) if not filtered.empty else "N/A",
                "end": str(filtered["iso_time"].max()) if not filtered.empty else "N/A",
            },
            "limitations": ["Tracks with fewer than two distinct geolocated points are omitted."],
        },
        status="success",
    )
    return cyclone_tracks_to_geojson(res)
