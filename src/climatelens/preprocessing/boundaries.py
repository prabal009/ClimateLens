"""State boundary preprocessing and GeoJSON standardization for ClimateLens India."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
from nielytics.geospatial import validate_coordinates
from shapely.geometry import mapping, shape

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    normalize_state_id,
    resolve_raw_file,
)

logger = logging.getLogger(__name__)


def process_boundaries(
    output_dir: Path | None = None,
    simplify_tolerance: float = 0.01,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load, standardize, and export India state boundaries.

    Steps:
    1. Extract state_NWIC.GeoJSON from state_nwic_geojson.zip.
    2. Reproject from EPSG:7755 to EPSG:4326 (WGS84).
    3. Validate coordinate bounds via nielytics.geospatial.validate_coordinates.
    4. Standardize 36 State/UT records with canonical metadata.
    5. Export full-resolution and simplified GeoJSON files.
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = resolve_raw_file("boundaries")
    logger.info("Loading boundaries from %s", zip_path)

    with zipfile.ZipFile(zip_path) as z:
        with z.open("state_NWIC.GeoJSON") as f:
            raw_gdf = gpd.read_file(f)

    # Reproject from EPSG:7755 to EPSG:4326
    if raw_gdf.crs != "EPSG:4326":
        gdf = raw_gdf.to_crs(epsg=4326)
    else:
        gdf = raw_gdf.copy()

    # Validate coordinate bounds with nielytics
    minx, miny, maxx, maxy = gdf.total_bounds
    validate_coordinates(minx, miny, crs="EPSG:4326")
    validate_coordinates(maxx, maxy, crs="EPSG:4326")

    # Standardize records
    standardized_rows: list[dict[str, Any]] = []
    geometries = []

    for _, row in gdf.iterrows():
        raw_name = str(row.get("state_name", "")).strip()
        raw_code = str(row.get("state", "")).strip()
        raw_stcode = str(row.get("stcode", "")).strip()

        state_id = normalize_state_id(raw_name) or normalize_state_id(raw_code)
        if not state_id or state_id not in STATES_MAP:
            raise ValueError(f"Unrecognized state record: name='{raw_name}', code='{raw_code}'")

        meta = STATES_MAP[state_id]
        standardized_rows.append({
            "state_id": state_id,
            "state_name": meta["state_name"],
            "state_code": meta["state_code"],
            "stcode": meta["stcode"],
            "type": meta["type"],
            "region": meta["region"],
            "source_agency": str(row.get("src_agency", "Survey of India (SOI)")),
        })
        geometries.append(row["geometry"])

    clean_gdf = gpd.GeoDataFrame(standardized_rows, geometry=geometries, crs="EPSG:4326")
    clean_gdf = clean_gdf.sort_values("state_name").reset_index(drop=True)

    if len(clean_gdf) != 36:
        raise ValueError(f"Expected 36 State/UT features, found {len(clean_gdf)}")

    # Full resolution export
    full_path = out_dir / "state_boundaries.geojson"
    clean_gdf.to_file(full_path, driver="GeoJSON")
    logger.info("Saved full-resolution boundaries to %s (%d bytes)", full_path, full_path.stat().st_size)

    # Simplified export for fast web display (topology preserved)
    simp_gdf = clean_gdf.copy()
    simp_gdf["geometry"] = clean_gdf.geometry.simplify(tolerance=simplify_tolerance, preserve_topology=True)
    simp_path = out_dir / "state_boundaries_simplified.geojson"
    simp_gdf.to_file(simp_path, driver="GeoJSON")
    logger.info("Saved simplified boundaries to %s (%d bytes)", simp_path, simp_path.stat().st_size)

    return clean_gdf, simp_gdf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_boundaries()
