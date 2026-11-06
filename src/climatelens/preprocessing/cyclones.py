"""NOAA IBTrACS cyclone track preprocessing for ClimateLens India."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nielytics import Dataset
from nielytics.geospatial import validate_coordinates

from climatelens.constants import PROCESSED_DATA_DIR, resolve_raw_file
from climatelens.preprocessing.loader import get_raw_sheet_df

logger = logging.getLogger(__name__)

SCIENTIFIC_CAVEAT = (
    "NOAA IBTrACS v04r01 North Indian Ocean basin. Proximity to coastline does NOT constitute "
    "verified meteorological landfall. WMO and USA wind metrics use different averaging periods "
    "and should not be directly combined."
)


def _to_clean_float(val: Any) -> float | None:
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", " "):
        return None
    try:
        f = float(s)
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _to_clean_str(val: Any) -> str | None:
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "null", "unnamed") else None


def process_cyclones(output_dir: Path | None = None) -> pd.DataFrame:
    """Parse NOAA IBTrACS NI CSV (1842-2024).

    Steps:
    1. Read CSV skipping units row.
    2. Filter basin == 'NI' (North Indian Ocean).
    3. Sort tracks by SID and ISO_TIME ascending.
    4. Validate coordinates via nielytics.geospatial.validate_coordinates.
    5. Convert wind from knots to km/h (1 knot = 1.852 km/h).
    6. Retain DIST2LAND and LANDFALL flags with caveats.
    7. Export cyclone_track_points.parquet.
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_df = get_raw_sheet_df("Cyclone_Raw")
        logger.info("Loaded %d rows from Cyclone_Raw", len(raw_df))
        # Skip units row if present (e.g. LAT == 'degrees_north' or SEASON == 'Year')
        if len(raw_df) > 0 and str(raw_df.iloc[0].get("SEASON", "")).strip().lower() == "year":
            raw_df = raw_df.iloc[1:].copy()
        ds = Dataset(raw_df)
        raw_df = ds.frame.copy()
    except Exception as exc:
        csv_path = resolve_raw_file("cyclone")
        logger.info("Falling back to cyclone CSV from %s (%s)", csv_path, exc)
        raw_df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)

    # Filter basin == 'NI'
    ni_df = raw_df[raw_df["BASIN"] == "NI"].copy()
    records: list[dict[str, Any]] = []

    for _, row in ni_df.iterrows():
        sid = str(row["SID"]).strip()
        season = int(row["SEASON"])
        name = _to_clean_str(row.get("NAME")) or "UNNAMED"
        iso_time = str(row["ISO_TIME"]).strip()
        subbasin = str(row.get("SUBBASIN", "")).strip() or None
        nature = str(row.get("NATURE", "")).strip() or None
        track_type = str(row.get("TRACK_TYPE", "main")).strip() or "main"

        lat = float(row["LAT"])
        lon = float(row["LON"])

        # Validate coordinates using nielytics
        validate_coordinates(lon, lat, crs="EPSG:4326")

        wmo_wind_kts = _to_clean_float(row.get("WMO_WIND"))
        wmo_wind_kmh = round(wmo_wind_kts * 1.852, 1) if wmo_wind_kts is not None else None

        usa_wind_kts = _to_clean_float(row.get("USA_WIND"))
        usa_wind_kmh = round(usa_wind_kts * 1.852, 1) if usa_wind_kts is not None else None

        wmo_pres = _to_clean_float(row.get("WMO_PRES"))
        usa_pres = _to_clean_float(row.get("USA_PRES"))
        dist2land = _to_clean_float(row.get("DIST2LAND"))
        landfall = _to_clean_float(row.get("LANDFALL"))

        records.append({
            "sid": sid,
            "season": season,
            "name": name,
            "iso_time": iso_time,
            "basin": "NI",
            "subbasin": subbasin,
            "nature": nature,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "wmo_wind_kts": wmo_wind_kts,
            "wmo_wind_kmh": wmo_wind_kmh,
            "usa_wind_kts": usa_wind_kts,
            "usa_wind_kmh": usa_wind_kmh,
            "wmo_pres_hpa": wmo_pres,
            "usa_pres_hpa": usa_pres,
            "dist2land_km": dist2land,
            "landfall": int(landfall) if landfall is not None else None,
            "track_type": track_type,
            "is_interpolated": False,
            "caveat": SCIENTIFIC_CAVEAT,
        })

    df = pd.DataFrame(records)
    # Sort strictly by storm identifier and chronological observation time
    df = df.sort_values(["sid", "iso_time"]).reset_index(drop=True)

    parquet_path = out_dir / "cyclone_track_points.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved cyclone track points parquet to %s (%d rows)", parquet_path, len(df))

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_cyclones()
