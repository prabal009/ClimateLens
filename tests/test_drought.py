"""Unit tests for national drought timeseries, invariants, and severity hierarchy."""

import pandas as pd
import pytest

from climatelens.constants import PROCESSED_DATA_DIR
from climatelens.hazards.drought_hazard import analyze_national_drought


@pytest.fixture(scope="module")
def drought_df():
    path = PROCESSED_DATA_DIR / "national_drought.parquet"
    assert path.exists()
    return pd.read_parquet(path)


def test_drought_row_count_and_dates(drought_df):
    assert len(drought_df) == 277
    assert drought_df["geographic_level"].unique().tolist() == ["national"]
    assert drought_df["entity_id"].unique().tolist() == ["INDIA_NATIONAL"]


def test_drought_sum_invariant(drought_df):
    # Invariant: no_drought_pct + d0_d4_pct == 100.0 within 0.05%
    sum_pct = drought_df["no_drought_pct"] + drought_df["d0_d4_pct"]
    diff = (sum_pct - 100.0).abs()
    assert (diff <= 0.05).all(), f"Drought sum invariant failed, max diff={diff.max()}"


def test_drought_hierarchy_monotonicity(drought_df):
    # Hierarchy: 0 <= d4 <= d3_d4 <= d2_d4 <= d1_d4 <= d0_d4 <= 100
    tol = 1e-4
    assert (drought_df["d0_d4_pct"] >= 0.0 - tol).all()
    assert (drought_df["d0_d4_pct"] <= 100.0 + tol).all()
    assert (drought_df["d0_d4_pct"] >= drought_df["d1_d4_pct"] - tol).all()
    assert (drought_df["d1_d4_pct"] >= drought_df["d2_d4_pct"] - tol).all()
    assert (drought_df["d2_d4_pct"] >= drought_df["d3_d4_pct"] - tol).all()
    assert (drought_df["d3_d4_pct"] >= drought_df["d4_pct"] - tol).all()
    assert (drought_df["d4_pct"] >= 0.0 - tol).all()


def test_drought_hazard_analyzer():
    res = analyze_national_drought(2021, 2026)
    assert res["status"] == "ok"
    assert res["geographic_level"] == "national"
    assert "peak_abnormally_dry" in res
    assert "peak_moderate_drought" in res
    assert "sharpest_weekly_expansions" in res
    assert len(res["timeseries"]) == 277
