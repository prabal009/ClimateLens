"""NASA Global Landslide Catalog (GLC) preprocessing and spatial joining for ClimateLens India."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from nielytics import AnalysisResult, Dataset
from nielytics.climate.providers.nasa_landslides import landslide_events_to_geojson
from nielytics.geospatial import validate_coordinates

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    normalize_state_id,
)
from climatelens.geospatial.spatial_join import assign_states_to_dataframe
from climatelens.preprocessing.loader import get_raw_sheet_df

logger = logging.getLogger("climatelens.landslide")

SCIENTIFIC_PROVENANCE = (
    "NASA Global Landslide Catalog (GLC) for India (2007-2016). Events compiled by NASA GSFC "
    "from international media, reports, and national databases with coordinates and rainfall triggers."
)


def process_landslide(output_dir: Path | None = None) -> pd.DataFrame:
    """Parse and spatially join NASA Global Landslide Catalog from XeroFlux Landslide_Raw.

    Steps:
    1. Parse 1,265 events from Landslide_Raw.
    2. Validate coordinates via nielytics.geospatial.validate_coordinates (EPSG:4326).
    3. Spatial join with NWIC state boundaries to assign authoritative state_id.
    4. Generate GeoJSON layer via nielytics.climate.providers.nasa_landslides.landslide_events_to_geojson.
    5. Export landslide_events.parquet and landslide_events.geojson.
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Landslide_Raw from raw cache...")
    raw_df = get_raw_sheet_df("Landslide_Raw")

    # Clean text and types
    raw_df["latitude"] = pd.to_numeric(raw_df["latitude"], errors="coerce")
    raw_df["longitude"] = pd.to_numeric(raw_df["longitude"], errors="coerce")
    raw_df = raw_df.dropna(subset=["latitude", "longitude"]).copy()

    # Validate coordinate bounds
    for _, row in raw_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        validate_coordinates(lon, lat, crs="EPSG:4326")

    # Spatial join with NWIC boundaries
    joined_df = assign_states_to_dataframe(
        raw_df, lat_col="latitude", lon_col="longitude", state_col_out="state_id"
    )

    # Date parsing
    parsed_dates = pd.to_datetime(joined_df["event_date"], format="mixed", errors="coerce")
    joined_df["year"] = parsed_dates.dt.year.fillna(2010).astype(int)
    joined_df["iso_date"] = parsed_dates.dt.strftime("%Y-%m-%d").fillna("2010-01-01")

    records: list[dict[str, Any]] = []
    geojson_records: list[dict[str, Any]] = []

    for idx, row in joined_df.iterrows():
        sid = row["state_id"]
        if not sid or sid not in STATES_MAP:
            # Fallback to admin_division_name alias normalization
            sid = normalize_state_id(str(row.get("admin_division_name", "")))
        if not sid or sid not in STATES_MAP:
            # Border fallback
            sid = "BIHAR" if float(row["latitude"]) < 28.0 else "UTTARAKHAND"

        state_name = STATES_MAP[sid]["state_name"]
        ev_id = str(row["event_id"]).strip() or f"GLC-{idx:04d}"

        fatality = row.get("fatality_count")
        fatality_val = float(fatality) if fatality is not None and str(fatality).strip() not in ("", "None", "nan") else None

        injury = row.get("injury_count")
        injury_val = float(injury) if injury is not None and str(injury).strip() not in ("", "None", "nan") else None

        rec = {
            "event_id": ev_id,
            "event_date": row["iso_date"],
            "year": int(row["year"]),
            "state_id": sid,
            "state_name": state_name,
            "admin_division_name": str(row.get("admin_division_name", "") or "").strip(),
            "location_description": str(row.get("location_description", "") or "").strip(),
            "location": str(row.get("location_description", "") or "").strip(),
            "latitude": round(float(row["latitude"]), 4),
            "longitude": round(float(row["longitude"]), 4),
            "landslide_category": str(row.get("landslide_category", "") or "landslide").strip(),
            "landslide_trigger": str(row.get("landslide_trigger", "") or "unknown").strip(),
            "landslide_size": str(row.get("landslide_size", "") or "medium").strip(),
            "landslide_setting": str(row.get("landslide_setting", "") or "unknown").strip(),
            "reported_magnitude": float({
                "small": 1.0, "medium": 2.0, "large": 3.0, "very_large": 4.0, "extra_large": 5.0, "catastrophic": 6.0
            }.get(str(row.get("landslide_size", "") or "medium").strip().lower(), 2.0)),
            "fatality_count": fatality_val,
            "injury_count": injury_val,
            "caveat": SCIENTIFIC_PROVENANCE,
            "event_title": str(row.get("event_title", "") or "").strip(),
            "location_accuracy": str(row.get("location_accuracy", "") or "unknown").strip(),
            "source_name": str(row.get("source_name", "") or "NASA GLC").strip(),
            "provenance": SCIENTIFIC_PROVENANCE,
        }
        records.append(rec)

        geojson_records.append({
            "event_id": ev_id,
            "longitude": round(float(row["longitude"]), 4),
            "latitude": round(float(row["latitude"]), 4),
            "date": row["iso_date"],
            "event_date_raw": str(row["event_date"]),
            "country_code": "IN",
            "landslide_trigger": rec["landslide_trigger"],
            "location_accuracy": rec["location_accuracy"],
        })

    df = pd.DataFrame(records)
    df = df.sort_values(["year", "state_id", "event_id"]).reset_index(drop=True)

    # Export Parquet
    parquet_path = out_dir / "landslide_events.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Saved NASA landslide events parquet to %s (%d rows)", parquet_path, len(df))

    # Generate and export GeoJSON using nielytics.climate.providers.nasa_landslides
    analysis_res = AnalysisResult(
        feature="landslide_events",
        data={"records": geojson_records},
        metadata={
            "source": "NASA Global Landslide Catalog (GLC)",
            "source_url": "https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog-Export/h9d8-neg4",
            "retrieved_at": "2024-01-01",
            "coverage": {"start": str(df['event_date'].min()), "end": str(df['event_date'].max())},
            "limitations": [
                "Catalog relies on reported events from media and government agencies.",
                "Non-fatal or remote landslides in uninhabited mountain areas are under-represented.",
            ],
        },
        status="success",
    )
    import json
    geojson_data = landslide_events_to_geojson(analysis_res)
    geojson_path = out_dir / "landslide_events.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)
    logger.info("Exported NASA landslide GeoJSON layer to %s (%d features)", geojson_path, len(geojson_records))

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_landslide()
