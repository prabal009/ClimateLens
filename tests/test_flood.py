"""Unit tests for flood events preprocessing, casualty preservation, and state reconciliation."""

import pandas as pd
import pytest

from climatelens.constants import PROCESSED_DATA_DIR
from climatelens.hazards.flood_hazard import analyze_flood_hazard


@pytest.fixture(scope="module")
def flood_df():
    path = PROCESSED_DATA_DIR / "flood_events.parquet"
    assert path.exists()
    return pd.read_parquet(path)


def test_flood_record_count_and_columns(flood_df):
    assert len(flood_df) == 6876
    assert "event_id" in flood_df.columns
    assert "primary_state_id" in flood_df.columns
    assert "human_fatality" in flood_df.columns


def test_zero_casualty_fabrication(flood_df):
    # Rule: Missing fatalities must remain null/None, NEVER imputed as 0
    missing_count = flood_df["human_fatality"].isna().sum()
    assert missing_count > 3000, f"Expected >3000 missing fatalities, found {missing_count}"

    # Non-missing fatalities must be non-negative
    valid_fatalities = flood_df["human_fatality"].dropna()
    assert (valid_fatalities >= 0).all()


def test_coordinates_uninvented_null_preservation(flood_df):
    # Latitude and longitude are absent in source IFI-Impacts v3; must remain null
    assert flood_df["latitude"].isna().all()
    assert flood_df["longitude"].isna().all()


def test_state_reconciliation_coverage(flood_df):
    # Reconciled primary states should cover >99.9% of records
    reconciled_count = flood_df["primary_state_id"].notna().sum()
    assert reconciled_count >= 6870


def test_flood_hazard_analyzer():
    res = analyze_flood_hazard(state_id="KERALA", start_year=2000, end_year=2023)
    assert res["status"] == "ok"
    assert res["state_id"] == "KERALA"
    assert res["total_events_recorded"] > 0
    assert "annual_frequency" in res
    assert "major_events" in res
