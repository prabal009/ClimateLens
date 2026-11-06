"""Spatial indexing and point-in-polygon queries for ClimateLens India."""

from __future__ import annotations

import logging
from pathlib import Path
import warnings
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from climatelens.constants import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

_CACHED_GDF: gpd.GeoDataFrame | None = None


def get_boundary_gdf(simplified: bool = False) -> gpd.GeoDataFrame:
    """Get the boundary GeoDataFrame, loading from processed files or generating if needed."""
    global _CACHED_GDF
    filename = "state_boundaries_simplified.geojson" if simplified else "state_boundaries.geojson"
    path = PROCESSED_DATA_DIR / filename

    if not path.exists():
        from climatelens.preprocessing.boundaries import process_boundaries
        clean_gdf, simp_gdf = process_boundaries()
        return simp_gdf if simplified else clean_gdf

    if not simplified and _CACHED_GDF is not None:
        return _CACHED_GDF

    gdf = gpd.read_file(path)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    if not simplified:
        _CACHED_GDF = gdf
    return gdf


def find_state_for_point(
    latitude: float | None,
    longitude: float | None,
    boundary_gdf: gpd.GeoDataFrame | None = None,
) -> str | None:
    """Find the canonical state_id for a given (lat, lon) coordinate point."""
    if latitude is None or longitude is None:
        return None
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (ValueError, TypeError):
        return None

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    # Fast bounding box check for India
    if not (6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0):
        return None

    gdf = boundary_gdf if boundary_gdf is not None else get_boundary_gdf(simplified=True)
    pt = Point(lon, lat)

    # Spatial index query
    possible_matches_index = list(gdf.sindex.intersection(pt.bounds))
    if not possible_matches_index:
        return None

    possible_matches = gdf.iloc[possible_matches_index]
    exact_matches = possible_matches[possible_matches.contains(pt)]
    if not exact_matches.empty:
        return str(exact_matches.iloc[0]["state_id"])

    # If slightly outside coastline (e.g. within 0.05 degrees ~ 5 km), find nearest
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        distances = possible_matches.distance(pt)
        min_dist = distances.min()
        if min_dist <= 0.05:
            nearest_idx = distances.idxmin()
            return str(gdf.loc[nearest_idx, "state_id"])
    return None


def assign_states_to_dataframe(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    state_col_out: str = "spatial_state_id",
) -> pd.DataFrame:
    """Assign canonical state_id to each row with valid latitude/longitude."""
    res = df.copy()
    if res.empty:
        res[state_col_out] = None
        return res

    boundary_gdf = get_boundary_gdf(simplified=True)
    valid_coords = (
        res[lat_col].notna()
        & res[lon_col].notna()
        & (res[lat_col] >= 6.0)
        & (res[lat_col] <= 38.0)
        & (res[lon_col] >= 68.0)
        & (res[lon_col] <= 98.0)
    )

    boundary_sub = boundary_gdf[["state_id", "geometry"]].rename(
        columns={"state_id": "_boundary_state_id"}
    )
    valid_subset = res.loc[valid_coords]
    points_gdf = gpd.GeoDataFrame(
        valid_subset[[c for c in valid_subset.columns if c != "geometry"]],
        geometry=gpd.points_from_xy(valid_subset[lon_col], valid_subset[lat_col]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points_gdf, boundary_sub, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    res[state_col_out] = joined["_boundary_state_id"]

    # Fallback to nearest boundary for coastal points within ~5km buffer
    null_mask = res[state_col_out].isna() & valid_coords
    if null_mask.any():
        res.loc[null_mask, state_col_out] = [
            find_state_for_point(lat, lon, boundary_gdf)
            for lat, lon in zip(res.loc[null_mask, lat_col], res.loc[null_mask, lon_col])
        ]
    return res
