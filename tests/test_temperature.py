"""Unit tests for regional maximum temperature processing, baselines, and warming trends."""

import pandas as pd
import pytest

from climatelens.constants import PROCESSED_DATA_DIR, TEMPERATURE_REGIONS
from climatelens.hazards.heat_hazard import analyze_regional_temperature


@pytest.fixture(scope="module")
def temperature_df():
    path = PROCESSED_DATA_DIR / "regional_temperature.parquet"
    assert path.exists()
    return pd.read_parquet(path)


def test_temperature_records_count_and_regions(temperature_df):
    # 8 regions * 107 years (1901-2007) * 12 months = 10,272
    assert len(temperature_df) == 10272
    regions = set(temperature_df["region_code"].unique())
    assert regions == set(TEMPERATURE_REGIONS.keys())
    assert temperature_df["tmax_celsius"].notna().all()


def test_baseline_and_anomalies(temperature_df):
    assert "baseline_normal_celsius" in temperature_df.columns
    assert "monthly_anomaly_celsius" in temperature_df.columns

    # Baseline 1971-2000 must be populated for every record
    assert temperature_df["baseline_normal_celsius"].notna().all()

    # Invariant: monthly_anomaly == round(tmax - baseline_normal, 2)
    expected_anomaly = (temperature_df["tmax_celsius"] - temperature_df["baseline_normal_celsius"]).round(2)
    diff = (temperature_df["monthly_anomaly_celsius"] - expected_anomaly).abs().max()
    assert diff <= 0.01

    # Caveat must be present
    assert "caveat" in temperature_df.columns
    assert "not individual daily heatwave declarations" in temperature_df.iloc[0]["caveat"].lower()


def test_regional_temperature_hazard_analyzer():
    res = analyze_regional_temperature("ALLIN", 1901, 2007)
    assert res["status"] == "ok"
    assert res["region_code"] == "ALLIN"
    assert "warmest_year" in res
    assert "coolest_year" in res
    assert "decadal_trends" in res
    assert len(res["decadal_trends"]) >= 10
