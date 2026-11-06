"""Landslide event density and spatial hazard analysis for ClimateLens India."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nielytics.geospatial import points_to_geojson
from nielytics.hazards.landslide import LandslideAnalyzer

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    normalize_state_id,
)

logger = logging.getLogger(__name__)

SCIENTIFIC_CAVEAT = (
    "NASA Global Landslide Catalog (GLC) for India (2007-2016). Events compiled by NASA GSFC "
    "from media, disaster reports, and scientific publications. Used for spatial density and hazard frequency."
)


def _load_landslide_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "landslide_events.parquet"
    if not path.exists():
        from climatelens.preprocessing.landslide import process_landslide
        return process_landslide()
    return pd.read_parquet(path)


def analyze_landslide_hazard(
    state_id: str | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    """Analyze landslide event distribution, annual frequencies, and spatial clusters."""
    df = _load_landslide_data()

    sid = normalize_state_id(state_id) if state_id else None
    if state_id and not sid:
        raise ValueError(f"Invalid state_id '{state_id}'")

    filtered = df.copy()
    if sid:
        filtered = filtered[filtered["state_id"] == sid]
    if year is not None:
        filtered = filtered[filtered["year"] == year]

    if filtered.empty:
        return {
            "status": "no_data",
            "state_id": sid,
            "state_name": STATES_MAP[sid]["state_name"] if sid else "All India",
            "year": year,
            "message": "No landslide events match the specified filter criteria.",
        }

    # NIElytics LandslideAnalyzer frequency analysis
    nielytics_freq = None
    try:
        events_input = filtered[["event_date", "event_id"]].copy().rename(columns={"event_date": "date"})
        dummy_env = pd.DataFrame({"date": [filtered["event_date"].iloc[0]], "rainfall": [10.0], "slope": [20.0]})
        analyzer = LandslideAnalyzer(dummy_env, rainfall_unit="mm", slope_unit="degrees")
        freq_result = analyzer.landslide_frequency(events_input, event_date_column="date")
        nielytics_freq = freq_result.data.get("years", [])
    except Exception as exc:
        logger.debug("LandslideAnalyzer frequency error: %s", exc)

    # Annual frequency breakdown
    annual_counts = (
        filtered.groupby("year")
        .agg(
            events_count=("event_id", "count"),
            mean_magnitude=("reported_magnitude", lambda s: round(float(s.mean()), 2)),
            fatalities=("fatality_count", "sum"),
            injuries=("injury_count", "sum"),
        )
        .reset_index()
    )
    # State ranking (if looking national)
    state_summary = []
    if not sid:
        state_df = (
            filtered.groupby(["state_id", "state_name"])
            .agg(
                events_count=("event_id", "count"),
                mean_magnitude=("reported_magnitude", lambda s: round(float(s.mean()), 2)),
            )
            .reset_index()
            .sort_values("events_count", ascending=False)
        )
        state_summary = state_df.to_dict(orient="records")

    # Spatial clusters / hotspots
    # Western Ghats: lat [8, 21], lon [73, 77]
    # Himalayas: lat [26, 36], lon [74, 96]
    wg_count = int((
        (filtered["latitude"] >= 8.0) & (filtered["latitude"] <= 21.0) &
        (filtered["longitude"] >= 73.0) & (filtered["longitude"] <= 77.0)
    ).sum())

    him_count = int((
        (filtered["latitude"] >= 26.0) & (filtered["latitude"] <= 36.0) &
        (filtered["longitude"] >= 74.0) & (filtered["longitude"] <= 96.0)
    ).sum())

    # Format points for GeoJSON overlay
    sample_points = []
    for _, r in filtered.head(500).iterrows():
        sample_points.append({
            "event_id": r["event_id"],
            "year": int(r["year"]),
            "state_id": r["state_id"],
            "state_name": r["state_name"],
            "location": r["location"],
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "reported_magnitude": float(r["reported_magnitude"]),
        })

    return {
        "status": "ok",
        "state_id": sid,
        "state_name": STATES_MAP[sid]["state_name"] if sid else "All India",
        "year": year,
        "period": {"start": int(filtered["year"].min()), "end": int(filtered["year"].max())},
        "total_events_count": len(filtered),
        "mean_reported_magnitude": round(float(filtered["reported_magnitude"].mean()), 2),
        "hotspot_distribution": {
            "himalayan_belt_events": him_count,
            "western_ghats_events": wg_count,
            "other_regions_events": len(filtered) - him_count - wg_count,
        },
        "annual_frequency": annual_counts.to_dict(orient="records"),
        "state_breakdown": state_summary,
        "sample_points": sample_points,
        "caveat": SCIENTIFIC_CAVEAT,
    }


def get_landslide_geojson(state_id: str | None = None, year: int | None = None) -> dict[str, Any]:
    """Generate authoritative GeoJSON point feature collection using nielytics.climate.providers.nasa_landslides."""
    from nielytics import AnalysisResult
    from nielytics.climate.providers.nasa_landslides import landslide_events_to_geojson

    df = _load_landslide_data()
    sid = normalize_state_id(state_id) if state_id else None
    if sid:
        df = df[df["state_id"] == sid]
    if year is not None:
        df = df[df["year"] == year]

    geojson_records = []
    for _, row in df.iterrows():
        geojson_records.append({
            "event_id": str(row["event_id"]),
            "longitude": float(row["longitude"]),
            "latitude": float(row["latitude"]),
            "date": str(row["event_date"]),
            "event_date_raw": str(row["event_date"]),
            "country_code": "IN",
            "landslide_trigger": str(row.get("landslide_trigger", "unknown")),
            "location_accuracy": str(row.get("location_accuracy", "unknown")),
        })

    analysis_res = AnalysisResult(
        feature="landslide_events",
        data={"records": geojson_records},
        metadata={
            "source": "NASA Global Landslide Catalog (GLC)",
            "source_url": "https://data.nasa.gov",
            "retrieved_at": "2024-01-01",
            "coverage": {
                "start": str(df["event_date"].min()) if not df.empty else "N/A",
                "end": str(df["event_date"].max()) if not df.empty else "N/A",
            },
            "limitations": ["NASA GLC media-reported catalog."],
        },
        status="success",
    )
    return landslide_events_to_geojson(analysis_res)
