"""Unit tests for NOAA IBTrACS cyclone track preprocessing and CycloneAnalyzer integration."""

import pandas as pd
import pytest

from climatelens.constants import PROCESSED_DATA_DIR
from climatelens.hazards.cyclone_hazard import (
    analyze_cyclone_hazard,
    get_storm_track,
)


@pytest.fixture(scope="module")
def cyclone_df():
    path = PROCESSED_DATA_DIR / "cyclone_track_points.parquet"
    assert path.exists()
    return pd.read_parquet(path)


def test_cyclone_record_count_and_basin(cyclone_df):
    assert len(cyclone_df) == 57880
    assert cyclone_df["basin"].unique().tolist() == ["NI"]
    assert cyclone_df["sid"].nunique() >= 1800


def test_monotonic_chronological_ordering(cyclone_df):
    # Within every storm track, iso_time must be monotonically non-decreasing
    for sid, group in cyclone_df.groupby("sid"):
        assert group["iso_time"].is_monotonic_increasing, f"Storm {sid} is not monotonically ordered"


def test_cyclone_fani_2019(cyclone_df):
    fani_track = get_storm_track("FANI")
    assert fani_track["status"] == "ok"
    assert fani_track["name"] == "FANI"
    assert fani_track["season"] == 2019
    assert fani_track["observation_count"] == 71
    assert fani_track["total_track_distance_km"] > 3000.0
    assert fani_track["peak_sustained_wind_kmh"] >= 200.0  # Category 4
    assert fani_track["min_central_pressure_hpa"] < 940.0


def test_cyclone_hazard_catalog():
    res = analyze_cyclone_hazard(year=2019)
    assert res["status"] == "ok"
    assert res["year"] == 2019
    assert res["total_storms_count"] > 5
    storm_names = [s["name"] for s in res["storms"]]
    assert "FANI" in storm_names
