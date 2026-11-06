"""Unit tests for state boundary standardization and geospatial indexing."""

import geopandas as gpd
import pytest

from climatelens.constants import PROCESSED_DATA_DIR
from climatelens.geospatial.spatial_join import (
    find_state_for_point,
    get_boundary_gdf,
)


def test_processed_boundaries_existence_and_crs():
    full_path = PROCESSED_DATA_DIR / "state_boundaries.geojson"
    simp_path = PROCESSED_DATA_DIR / "state_boundaries_simplified.geojson"

    assert full_path.exists(), "state_boundaries.geojson does not exist"
    assert simp_path.exists(), "state_boundaries_simplified.geojson does not exist"

    gdf = gpd.read_file(full_path)
    assert len(gdf) == 36
    assert gdf.crs == "EPSG:4326"

    # Coordinates bounds must match India extent
    minx, miny, maxx, maxy = gdf.total_bounds
    assert 68.0 <= minx <= 70.0
    assert 6.0 <= miny <= 9.0
    assert 96.0 <= maxx <= 98.0
    assert 36.0 <= maxy <= 38.0

    # Simplified GeoJSON must also have 36 features and much smaller file size
    gdf_simp = gpd.read_file(simp_path)
    assert len(gdf_simp) == 36
    assert gdf_simp.crs == "EPSG:4326"
    assert simp_path.stat().st_size < full_path.stat().st_size / 10


def test_spatial_join_point_in_polygon():
    # Known cities and landmarks
    test_cases = [
        ("Mumbai", 18.922, 72.834, "MAHARASHTRA"),
        ("Chennai", 13.0827, 80.2707, "TAMIL_NADU"),
        ("New Delhi", 28.6139, 77.2090, "DELHI"),
        ("Darjeeling", 27.036, 88.2627, "WEST_BENGAL"),
        ("Bengaluru", 12.9716, 77.5946, "KARNATAKA"),
        ("Shimla", 31.1048, 77.1734, "HIMACHAL_PRADESH"),
    ]

    for name, lat, lon, expected_sid in test_cases:
        sid = find_state_for_point(lat, lon)
        assert sid == expected_sid, f"Failed for {name}: expected {expected_sid}, got {sid}"


def test_spatial_join_out_of_bounds():
    # Coordinate outside India (e.g. London, Tokyo, null)
    assert find_state_for_point(51.5074, -0.1278) is None
    assert find_state_for_point(35.6762, 139.6503) is None
    assert find_state_for_point(None, None) is None
    assert find_state_for_point(999.0, 999.0) is None
