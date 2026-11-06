"""Comprehensive verification suite for NIElytics v2.0 upstream repository integration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import nielytics
from nielytics import (
    ClimateAnalytics,
    ClimateProjectionEngine,
    DatasetHarmonizer,
    HarmonizationSchema,
    HeatwaveAnalyzer,
    Insight,
    InsightEngine,
    MultiHazardAnalyzer,
    StandardClimateRecord,
    TemperatureForecaster,
)
from climatelens.adapter import (
    get_climate_projections,
    get_hazard_chart,
    get_multi_hazard_assessment,
    get_report_data,
    get_temperature_forecast,
)
from climatelens.constants import PROCESSED_DATA_DIR, STATES_MAP, normalize_state_id
from climatelens.engine.forecasting import (
    forecast_summer_heatwaves,
    get_state_cmip6_projections,
)
from climatelens.preprocessing.loader import (
    HARMONIZATION_SCHEMAS,
    harmonize_raw_dataset,
)


def test_nielytics_version() -> None:
    """Verify active environment reports nielytics==0.2.0.dev1."""
    assert nielytics.__version__ == "0.2.0.dev1", f"Expected '0.2.0.dev1', got '{nielytics.__version__}'"


def test_climate_analytics_facade() -> None:
    """Test ClimateAnalytics facade dataset registration, state heatwave analysis, and to_geojson()."""
    ca = ClimateAnalytics(offline=True)

    # 1. Dataset registration
    hw_sample = pd.DataFrame({
        "date": pd.date_range("2020-05-01", periods=10, freq="D"),
        "state": ["Maharashtra"] * 10,
        "tmax": [41.0, 42.5, 45.2, 46.8, 47.1, 44.0, 43.5, 42.0, 41.5, 40.5],
    })
    schema_hw = HarmonizationSchema(hazard="heatwave", metric="tmax", value_column="tmax", state_column="state")
    reg_res = ca.register_dataset("heatwave", hw_sample, schema=schema_hw)
    assert reg_res.feature == "dataset_harmonization"
    assert "heatwave" in ca._datasets
    assert len(ca._datasets["heatwave"]) == 10

    # 2. State heatwave analysis
    res_analyze = ca.analyze("heatwave", region="Maharashtra", normal_tmax=40.0, terrain="plains")
    assert res_analyze.feature == "imd_heatwave_classification"
    counts = res_analyze.data.get("counts", {})
    assert counts.get("severe_heatwave", 0) >= 1 or counts.get("heatwave", 0) >= 1

    # 3. to_geojson()
    cy_sample = pd.DataFrame({
        "date": ["2019-05-01", "2019-05-02"],
        "wind": [150.0, 180.0],
        "latitude": [18.5, 19.2],
        "longitude": [85.0, 85.8],
    })
    schema_cy = HarmonizationSchema(
        hazard="cyclone",
        metric="wind",
        value_column="wind",
        latitude_column="latitude",
        longitude_column="longitude",
    )
    ca.register_dataset("cyclone", cy_sample, schema=schema_cy)
    geojson = ca.to_geojson("cyclone")
    assert geojson.get("type") == "FeatureCollection"
    assert len(geojson.get("features", [])) == 2
    f0 = geojson["features"][0]
    assert f0.get("geometry", {}).get("type") == "Point"
    assert f0.get("geometry", {}).get("coordinates") == [85.0, 18.5]


def test_dataset_harmonizer() -> None:
    """Test harmonization into StandardClimateRecord format and unit conversion."""
    # Test across all registered schemas in climatelens
    for sheet_name in ["Heatwave_Raw", "Rainfall_Raw", "Flood_Raw", "Landslide_Raw", "Cyclone_Raw"]:
        assert sheet_name in HARMONIZATION_SCHEMAS
        df_harm = harmonize_raw_dataset(sheet_name)
        assert isinstance(df_harm, pd.DataFrame)
        assert not df_harm.empty
        required_cols = {"date", "hazard", "metric", "value", "unit", "source"}
        assert required_cols.issubset(set(df_harm.columns))

    # Test unit conversion (C -> F -> C)
    test_df = pd.DataFrame({
        "date": ["2023-05-01", "2023-05-02"],
        "temp_f": [104.0, 113.0],  # 40 C and 45 C
    })
    schema_conv = HarmonizationSchema(
        hazard="heatwave",
        metric="tmax",
        source_unit="F",
        target_unit="C",
        date_column="date",
        value_column="temp_f",
    )
    harm_conv = DatasetHarmonizer(schema_conv).harmonize(test_df)
    records = harm_conv.data.get("records", [])
    assert len(records) == 2
    assert pytest.approx(records[0]["value"], abs=0.1) == 40.0
    assert pytest.approx(records[1]["value"], abs=0.1) == 45.0
    assert records[0]["unit"] == "C"


def test_multi_hazard_analyzer() -> None:
    """Test compare_rainfall_and_flood, compare_rainfall_and_landslides, and composite_hazard_indicator."""
    # 1. compare_rainfall_and_flood
    rf_data = pd.DataFrame({
        "date": pd.date_range("2023-07-01", periods=10, freq="D"),
        "rainfall": [10.0, 20.0, 45.0, 80.0, 120.0, 15.0, 0.0, 0.0, 5.0, 2.0],
    })
    fl_data = pd.DataFrame({
        "date": [pd.Timestamp("2023-07-05")],
    })
    res_rf_fl = MultiHazardAnalyzer.compare_rainfall_and_flood(
        rainfall_data=rf_data,
        flood_events=fl_data,
        region="MAHARASHTRA",
        antecedent_days=3,
    )
    assert res_rf_fl.feature == "rainfall_flood_correlation"
    corrs = res_rf_fl.data.get("correlations", [])
    assert len(corrs) == 1
    # 3-day sum preceding 2023-07-05: 2023-07-02 to 2023-07-05 sum: 20 + 45 + 80 + 120 = 265
    assert corrs[0]["antecedent_rainfall_sum_mm"] >= 100.0

    # 2. compare_rainfall_and_landslides
    ls_data = pd.DataFrame({
        "date": [pd.Timestamp("2023-07-06")],
    })
    res_rf_ls = MultiHazardAnalyzer.compare_rainfall_and_landslides(
        rainfall_data=rf_data,
        landslide_events=ls_data,
        region="MAHARASHTRA",
        antecedent_days=3,
    )
    assert res_rf_ls.feature == "rainfall_landslide_correlation"
    assert len(res_rf_ls.data.get("correlations", [])) == 1

    # 3. composite_hazard_indicator
    metrics = {
        "heatwave_intensity": 15.0,
        "flood_frequency": 5.0,
        "landslide_density": 2.0,
        "daily_rainfall_mm": 12.5,
    }
    res_comp = MultiHazardAnalyzer.composite_hazard_indicator(metrics, normalization="min_max")
    assert res_comp.feature == "composite_hazard_indicator"
    score = res_comp.data.get("composite_score", 0.0)
    assert 0.0 <= score <= 100.0
    assert "Composite Score" in res_comp.data.get("mathematical_formula", "")


def test_insight_engine() -> None:
    """Test InsightEngine.generate_insights() rule-based output schema."""
    engine = InsightEngine()

    # Pass an AnalysisResult from imd_heatwave_classification
    temp_df = pd.DataFrame({
        "date": pd.date_range("2023-05-01", periods=15, freq="D"),
        "tmax": [45.0, 46.5, 47.2, 48.0, 47.5, 44.0, 43.0, 46.0, 47.1, 45.5, 42.0, 41.0, 40.0, 42.0, 43.0],
    })
    analyzer = HeatwaveAnalyzer(temp_df, date_column="date", temperature_column="tmax", unit="C")
    res_imd = analyzer.imd_heatwave_classification(normal_tmax=40.0, terrain="plains")

    res_insights = engine.generate_insights([res_imd])
    assert res_insights.feature == "deterministic_insights"
    insights = res_insights.data.get("insights", [])
    assert len(insights) >= 1
    ins0 = insights[0]
    assert "headline" in ins0
    assert "description" in ins0
    assert "metrics" in ins0
    assert "category" in ins0
    assert ins0["category"] == "frequency_spike"


def test_climate_projections() -> None:
    """Test ClimateProjectionEngine on SSP1-2.6, SSP2-4.5, SSP5-8.5 for 2030/2040/2050."""
    res = get_state_cmip6_projections("MAHARASHTRA")
    assert res["status"] == "ok"
    assert res["state_id"] == "MAHARASHTRA"
    assert res["baseline_temperature_c"] > 30.0
    assert res["target_years"] == [2030, 2040, 2050]

    scenarios = res["scenarios"]
    for ssp in ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]:
        assert ssp in scenarios
        proj = scenarios[ssp].get("projections", [])
        assert len(proj) == 3
        years_map = {p["target_year"]: p["projected_absolute_temperature_c"] for p in proj}
        assert 2030 in years_map and 2050 in years_map
        # Ensure SSP5-8.5 2050 > SSP1-2.6 2050
        ssp1_2050 = scenarios["SSP1-2.6"]["projections"][2]["projected_absolute_temperature_c"]
        ssp5_2050 = scenarios["SSP5-8.5"]["projections"][2]["projected_absolute_temperature_c"]
        assert ssp5_2050 > ssp1_2050

    # Trend extrapolation slope must be positive (warming trend)
    slope = res["trend_extrapolation"]["historical_slope_c_per_year"]
    assert slope > 0.0


def test_temperature_forecaster() -> None:
    """Test TemperatureForecaster 7-day predictions, 80%/95% intervals, and heatwave risk probability."""
    res = forecast_summer_heatwaves("MAHARASHTRA", horizon_days=7, threshold_c=40.0)
    assert res["status"] == "ok"
    assert res["state_id"] == "MAHARASHTRA"
    assert res["horizon_days"] == 7

    preds = res["predictions"]
    assert len(preds) == 7
    for p in preds:
        assert "predicted_tmax_c" in p
        assert "uncertainty_intervals" in p
        intervals = p["uncertainty_intervals"]
        assert "80%" in intervals and "95%" in intervals
        assert intervals["95%"]["lower"] <= intervals["80%"]["lower"]
        assert intervals["80%"]["upper"] <= intervals["95%"]["upper"]

    risks = res["heatwave_risk"]
    assert len(risks) == 7
    for r in risks:
        assert "exceedance_probability" in r
        assert 0.0 <= r["exceedance_probability"] <= 1.0
        assert r["risk_level"] in ("Low", "Moderate", "High")


def test_adapter_v2_endpoints() -> None:
    """Test get_climate_projections, get_temperature_forecast, get_multi_hazard_assessment."""
    # 1. get_climate_projections
    p_res = get_climate_projections("MAHARASHTRA")
    assert p_res["status"] == "ok"
    assert "data" in p_res
    assert "scenarios" in p_res["data"]
    assert "trend_extrapolation" in p_res["data"]

    # 2. get_temperature_forecast
    f_res = get_temperature_forecast("MAHARASHTRA", horizon=7)
    assert f_res["status"] == "ok"
    assert "data" in f_res
    assert len(f_res["data"]["predictions"]) == 7
    assert len(f_res["data"]["heatwave_risk"]) == 7

    # 3. get_multi_hazard_assessment
    m_res = get_multi_hazard_assessment("MAHARASHTRA", year=2019)
    assert m_res["status"] == "ok"
    data = m_res["data"]
    assert "composite_hazard_scores" in data
    assert "MAHARASHTRA" in data["composite_hazard_scores"]
    comp_score = data["composite_hazard_scores"]["MAHARASHTRA"]["composite_score"]
    assert 0.0 <= comp_score <= 1.0
    assert "regional_multi_hazard_overlap" in data
    assert "rainfall_flood_correlation" in data
    assert "rainfall_landslide_correlation" in data
    assert "compound_heat_drought" in data


def test_full_suite_regression() -> None:
    """Verify state normalization works seamlessly with nielytics geospatial module."""
    assert normalize_state_id("Orissa") == "ODISHA"
    assert normalize_state_id("West Bengal") == "WEST_BENGAL"
    assert normalize_state_id("Jammu and Kashmir") == "JAMMU_AND_KASHMIR"
    assert normalize_state_id("DELHI (UT)") == "DELHI"


def test_invalid_state_adapter_resiliency() -> None:
    """Assert querying v2 endpoints with nonexistent states returns clean no_data responses."""
    p_res = get_climate_projections("NONEXISTENT_STATE")
    assert p_res["status"] == "no_data"
    assert "Unrecognized or invalid state_id" in p_res.get("meta", {}).get("limitations", [""])[0]

    f_res = get_temperature_forecast("NONEXISTENT_STATE")
    assert f_res["status"] == "no_data"
    assert "Unrecognized or invalid state_id" in f_res.get("meta", {}).get("limitations", [""])[0]

    m_res = get_multi_hazard_assessment("NONEXISTENT_STATE")
    assert m_res["status"] == "no_data"
    assert "Unrecognized or invalid state_id" in m_res.get("meta", {}).get("limitations", [""])[0]


def test_island_ut_no_data_handling() -> None:
    """Assert querying states without historical records (e.g. Lakshadweep) returns no_data."""
    p_res = get_climate_projections("LAKSHADWEEP")
    assert p_res["status"] == "no_data"
    assert "No historical summer temperature observations recorded" in p_res.get("meta", {}).get("limitations", [""])[0]

    f_res = get_temperature_forecast("LAKSHADWEEP")
    assert f_res["status"] == "no_data"
    assert "No historical summer temperature observations recorded" in f_res.get("meta", {}).get("limitations", [""])[0]


def test_v2_hazard_charts() -> None:
    """Assert get_hazard_chart generates valid Plotly specs for forecasting, projections, and multi-hazard."""
    # 1. Forecasting timeseries and risk
    fc_ts = get_hazard_chart("forecasting", "timeseries", "MAHARASHTRA")
    assert fc_ts["status"] == "ok"
    assert "data" in fc_ts["data"] and len(fc_ts["data"]["data"]) > 0

    fc_risk = get_hazard_chart("forecasting", "risk", "MAHARASHTRA")
    assert fc_risk["status"] == "ok"
    assert "data" in fc_risk["data"] and len(fc_risk["data"]["data"]) > 0

    # 2. Projections line and bar
    pj_line = get_hazard_chart("projections", "line", "MAHARASHTRA")
    assert pj_line["status"] == "ok"
    assert "data" in pj_line["data"] and len(pj_line["data"]["data"]) > 0

    pj_bar = get_hazard_chart("cmip6", "bar", "MAHARASHTRA")
    assert pj_bar["status"] == "ok"
    assert "data" in pj_bar["data"] and len(pj_bar["data"]["data"]) > 0

    # 3. Multi-hazard ranking and distribution
    mh_rank = get_hazard_chart("multi_hazard", "ranking")
    assert mh_rank["status"] == "ok"
    assert "data" in mh_rank["data"] and len(mh_rank["data"]["data"]) > 0

    mh_dist = get_hazard_chart("multi_hazard", "distribution")
    assert mh_dist["status"] == "ok"
    assert "data" in mh_dist["data"] and len(mh_dist["data"]["data"]) > 0


def test_augmented_state_report() -> None:
    """Assert get_report_data includes v2 forecasting, projections, and multi-hazard payloads."""
    rep = get_report_data("MAHARASHTRA")
    assert rep["status"] == "ok"
    data = rep["data"]
    assert "climate_projections" in data
    assert "temperature_forecast" in data
    assert "multi_hazard_assessment" in data
    assert "cmip6_south_asia" in rep["meta"]["source_ids"]
    assert "nielytics_v2_forecasting" in rep["meta"]["source_ids"]
