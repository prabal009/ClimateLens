"""Unit tests for landslide inventory preprocessing, spatial join, and uncalibrated magnitude caveats."""

import pandas as pd
import pytest

from climatelens.constants import PROCESSED_DATA_DIR
from climatelens.hazards.landslide_hazard import analyze_landslide_hazard


@pytest.fixture(scope="module")
def landslide_df():
    path = PROCESSED_DATA_DIR / "landslide_events.parquet"
    assert path.exists()
    return pd.read_parquet(path)


def test_landslide_record_count_and_columns(landslide_df):
    assert len(landslide_df) == 1265
    assert "event_id" in landslide_df.columns
    assert "state_id" in landslide_df.columns
    assert "latitude" in landslide_df.columns
    assert "longitude" in landslide_df.columns
    assert "landslide_trigger" in landslide_df.columns
    assert "reported_magnitude" in landslide_df.columns

def test_spatial_join_completeness(landslide_df):
    # All 1,265 records must have an authoritative state_id assigned via polygon intersection
    assert landslide_df["state_id"].notna().all()
    assert landslide_df["state_name"].notna().all()

def test_coordinate_validity(landslide_df):
    # Latitude in [8.0, 36.0], Longitude in [72.0, 97.0]
    assert (landslide_df["latitude"] >= 8.0).all()
    assert (landslide_df["latitude"] <= 36.0).all()
    assert (landslide_df["longitude"] >= 72.0).all()
    assert (landslide_df["longitude"] <= 97.0).all()

def test_uncalibrated_magnitude_caveat(landslide_df):
    assert "caveat" in landslide_df.columns
    assert "nasa" in landslide_df.iloc[0]["caveat"].lower()

def test_landslide_hazard_analyzer():
    res = analyze_landslide_hazard(state_id="UTTARAKHAND")
    assert res["status"] == "ok"
    assert res["state_id"] == "UTTARAKHAND"
    assert res["total_events_count"] > 0
    assert "annual_frequency" in res
    assert "hotspot_distribution" in res


def test_landslide_geojson_and_nielytics():
    from climatelens.hazards.landslide_hazard import get_landslide_geojson
    gj = get_landslide_geojson(state_id="KERALA")
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 53
    assert gj["features"][0]["geometry"]["type"] == "Point"
