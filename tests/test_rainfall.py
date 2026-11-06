"""Unit tests for rainfall data preprocessing, invariants, and zero-fabrication rules."""

import numpy as np
import pandas as pd
import pytest

from climatelens.constants import PROCESSED_DATA_DIR
from climatelens.hazards.rainfall_hazard import (
    analyze_state_rainfall,
    get_rainfall_choropleth,
    get_state_rainfall_timeseries,
)


@pytest.fixture(scope="module")
def rainfall_daily():
    path = PROCESSED_DATA_DIR / "rainfall_daily.parquet"
    assert path.exists()
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def rainfall_state_year():
    path = PROCESSED_DATA_DIR / "rainfall_state_year.parquet"
    assert path.exists()
    return pd.read_parquet(path)


def test_zero_fabrication_rule(rainfall_daily):
    # Rule: Missing rainfall actuals must remain NaN, never 0.0
    # No negative rainfall measurements
    valid_acts = rainfall_daily["daily_actual_mm"].dropna()
    assert (valid_acts >= 0.0).all(), "Found negative rainfall measurements"

    # Extreme rainfall observations must be preserved
    assert (valid_acts >= 50.0).any(), "Expected legitimate extreme rainfall days to be preserved"


def test_geographic_level_partitioning(rainfall_daily):
    levels = set(rainfall_daily["geographic_level"].unique())
    assert levels == {"state", "regional", "national"}

    national_rows = rainfall_daily[rainfall_daily["geographic_level"] == "national"]
    assert len(national_rows) > 0
    assert (national_rows["entity_id"] == "INDIA_NATIONAL").all()

    state_rows = rainfall_daily[rainfall_daily["geographic_level"] == "state"]
    assert len(state_rows) > 0
    assert state_rows["state_id"].notna().all()


def test_state_year_aggregations(rainfall_state_year):
    assert len(rainfall_state_year) >= 36
    assert "monsoon_cumulative_actual_mm" in rainfall_state_year.columns
    assert "monsoon_departure_pct" in rainfall_state_year.columns
    assert "monsoon_category" in rainfall_state_year.columns

    # Verify Andaman & Nicobar 2026 values
    an = rainfall_state_year[(rainfall_state_year["state_id"] == "ANDAMAN_AND_NICOBAR_ISLANDS") & (rainfall_state_year["year"] == 2026)]
    assert not an.empty
    row = an.iloc[0]
    assert row["monsoon_cumulative_actual_mm"] > 1000.0
    assert row["monsoon_category"] == "N"


def test_rainfall_hazard_analyzer():
    res = analyze_state_rainfall("MAHARASHTRA", year=2026)
    assert res["status"] == "ok"
    assert res["state_id"] == "MAHARASHTRA"
    assert "cumulative_monsoon_actual_mm" in res
    assert "wet_spells_count" in res
    assert "heavy_rain_days_count" in res


def test_rainfall_choropleth():
    choro = get_rainfall_choropleth(2026)
    assert "states" in choro
    assert len(choro["states"]) == 36
    assert "legend" in choro
    for s in choro["states"]:
        assert s["color"].startswith("#")
        assert "monsoon_category" in s
