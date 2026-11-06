"""Automated tests for XeroFlux Dataset ingestion, deep NIElytics library integration, and new analytics endpoints."""

from __future__ import annotations

import pytest
import pandas as pd
from nielytics import (
    AnalysisResult,
    CycloneAnalyzer,
    Dataset,
    DroughtAnalyzer,
    FloodAnalyzer,
    HeatwaveAnalyzer,
    LandslideAnalyzer,
)
from nielytics.climate.providers.nasa_landslides import landslide_events_to_geojson
from nielytics.climate.providers.noaa_ibtracs import cyclone_tracks_to_geojson
from nielytics.geospatial import validate_coordinates

from climatelens.adapter import (
    compare_states,
    get_hazard_chart,
    get_heatwave_map,
    get_report_data,
    get_state_heatwave_trend,
    get_time_machine_data,
)
from climatelens.constants import (
    PROCESSED_DATA_DIR,
    RAW_CACHE_DIR,
    STATES_MAP,
)
from climatelens.hazards.charts import (
    generate_bar_chart,
    generate_distribution_chart,
    generate_geographic_chart,
    generate_hazard_chart,
    generate_timeseries_chart,
)
from climatelens.hazards.cyclone_hazard import (
    analyze_cyclone_hazard,
    get_cyclone_tracks_geojson,
    get_storm_track,
)
from climatelens.hazards.drought_hazard import analyze_national_drought
from climatelens.hazards.flood_hazard import analyze_flood_hazard
from climatelens.hazards.heat_hazard import (
    analyze_state_heatwaves,
    get_state_heatwave_summary_for_year,
)
from climatelens.hazards.landslide_hazard import (
    analyze_landslide_hazard,
    get_landslide_geojson,
)
from climatelens.hazards.rainfall_hazard import analyze_state_rainfall
from climatelens.preprocessing.loader import (
    SHEET_FILE_MAPPING,
    find_xeroflux_workbook,
    get_raw_sheet_df,
    load_raw_dataset,
)


def test_xeroflux_workbook_discovery():
    wb_path = find_xeroflux_workbook()
    assert wb_path.is_file()
    assert "XeroFlux Dataset" in wb_path.name or "XeroFlux Datasets" in wb_path.name


def test_raw_parquet_cache_presence():
    for sheet, fname in SHEET_FILE_MAPPING.items():
        p = RAW_CACHE_DIR / fname
        assert p.is_file(), f"Missing cache file: {p}"
        assert p.stat().st_size > 0, f"Empty cache file: {p}"


def test_nielytics_dataset_profiling():
    # Test wrapping sheets into Dataset and profiling
    ds = load_raw_dataset("Heatwave_Raw", standardize=True, profile=True)
    assert isinstance(ds, Dataset)
    prof = ds.profile()
    assert prof["rows"] == 1740
    assert prof["columns"] == 7
    assert prof["memory_bytes"] > 0
    missing = ds.missing_report()
    assert isinstance(missing, pd.DataFrame)
    assert "summer_max_temp_deg_c" in missing.index

    # Check text cleaning
    cleaned = ds.clean_text()
    assert len(cleaned.frame) == 1740


def test_state_heatwave_processing():
    hw_p = PROCESSED_DATA_DIR / "state_heatwave.parquet"
    assert hw_p.is_file()
    df = pd.read_parquet(hw_p)
    assert len(df) == 1740
    assert df["year"].min() == 1967
    assert df["year"].max() == 2024
    assert df["state_id"].notnull().all()
    assert df["state_name"].notnull().all()
    # Check null summer max temps preserved without fabrication
    assert df["summer_max_temp_deg_c"].isnull().sum() > 0


def test_heatwave_analyzer_nielytics():
    res = analyze_state_heatwaves("MAHARASHTRA", 1967, 2024)
    assert res["status"] == "ok"
    assert res["total_heat_wave_days"] > 0
    assert "peak_heatwave_year" in res
    assert "nielytics_analysis" in res
    nielytics_res = res["nielytics_analysis"]
    assert nielytics_res is not None
    assert nielytics_res["threshold_c"] == 40.0
    assert nielytics_res["years_summer_peak_ge_40c"] > 0
    assert len(nielytics_res["peak_season_spell_episodes"]) > 0


def test_state_heatwave_summary_for_year():
    summary_2024 = get_state_heatwave_summary_for_year(2024)
    assert len(summary_2024) == 30
    assert summary_2024[0]["total_heat_wave_days"] >= summary_2024[-1]["total_heat_wave_days"]
    assert summary_2024[0]["state_id"] == "RAJASTHAN"
    assert summary_2024[0]["total_heat_wave_days"] == 60


def test_flood_analyzer_rainfall_wet_spells_and_extremes():
    res = analyze_state_rainfall("MAHARASHTRA", 2026)
    assert res["status"] == "ok"
    assert res["wet_spells_count"] > 0
    assert len(res["wet_spells"]) > 0
    assert "dry_spells_count" in res
    assert res["dry_spells_count"] > 0


def test_drought_analyzer_dry_spells():
    df = pd.DataFrame({
        "date": pd.date_range("2026-06-01", periods=10, freq="D"),
        "rainfall": [0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 5.0, 10.0],
    })
    dr = DroughtAnalyzer(df, date_column="date", rainfall_column="rainfall", unit="mm")
    res = dr.dry_spell_duration(threshold_mm=0.0, minimum_days=3)
    assert res.status == "success"
    assert len(res.data["episodes"]) == 2


def test_drought_raw_invariants():
    dr = pd.read_parquet(PROCESSED_DATA_DIR / "national_drought.parquet")
    assert len(dr) == 277
    # Sum invariant
    diff = (dr["no_drought_pct"] + dr["d0_d4_pct"]) - 100.0
    assert (diff.abs() <= 0.05).all()
    # Monotonicity
    tol = 1e-4
    assert (dr["d4_pct"] <= dr["d3_d4_pct"] + tol).all()
    assert (dr["d3_d4_pct"] <= dr["d2_d4_pct"] + tol).all()
    assert (dr["d2_d4_pct"] <= dr["d1_d4_pct"] + tol).all()
    assert (dr["d1_d4_pct"] <= dr["d0_d4_pct"] + tol).all()


def test_flood_event_frequency_nielytics():
    res = analyze_flood_hazard()
    assert res["status"] == "ok"
    assert res["total_events_recorded"] == 6876
    assert "nielytics_frequency" in res
    assert len(res["nielytics_frequency"]) > 0


def test_nasa_landslide_catalog_geojson():
    res = analyze_landslide_hazard()
    assert res["status"] == "ok"
    assert res["total_events_count"] == 1265
    assert res["hotspot_distribution"]["himalayan_belt_events"] > 700
    assert res["hotspot_distribution"]["western_ghats_events"] > 100

    gj = get_landslide_geojson(state_id="KERALA")
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 53
    # Check coordinates valid
    for feat in gj["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        validate_coordinates(lon, lat, crs="EPSG:4326")


def test_cyclone_analyzer_and_tracks_geojson():
    haz = analyze_cyclone_hazard(year=2020)
    assert haz["status"] == "ok"
    assert haz["total_storms_count"] == 10
    assert len(haz["frequency_by_year"]) == 1

    gj = get_cyclone_tracks_geojson(year=2020)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 10
    for feat in gj["features"]:
        assert feat["geometry"]["type"] in ("LineString", "MultiLineString")
        assert len(feat["geometry"]["coordinates"]) >= 2

    # Specific storm track
    amphan = get_storm_track("AMPHAN")
    assert amphan["status"] == "ok"
    assert amphan["total_track_distance_km"] > 1500.0
    assert amphan["peak_sustained_wind_kmh"] > 200.0


def test_interactive_chart_generation():
    # Timeseries chart
    hw_ts = generate_hazard_chart("heatwave", "timeseries", state_id="MAHARASHTRA")
    assert "data" in hw_ts and "layout" in hw_ts
    assert len(hw_ts["data"]) > 0

    # Bar chart
    rf_bar = generate_hazard_chart("rainfall", "bar", year=2026)
    assert "data" in rf_bar and "layout" in rf_bar
    assert len(rf_bar["data"]) > 0

    # Geographic chart
    ls_geo = generate_hazard_chart("landslide", "geographic")
    assert "data" in ls_geo and "layout" in ls_geo
    assert len(ls_geo["data"]) > 0

    # Distribution chart
    dr_dist = generate_hazard_chart("drought", "distribution")
    assert "data" in dr_dist and "layout" in dr_dist
    assert len(dr_dist["data"]) > 0


def test_adapter_heatwave_endpoints():
    trend = get_state_heatwave_trend("MAHARASHTRA", 1967, 2024)
    assert trend["status"] == "ok"
    assert trend["data"]["state_id"] == "MAHARASHTRA"
    assert trend["meta"]["metric"] == "heat_wave_and_severe_days"
    assert trend["meta"]["observation_count"] == 58

    hw_map = get_heatwave_map(2024)
    assert hw_map["status"] == "ok"
    assert len(hw_map["data"]["states"]) == 30

    chart = get_hazard_chart("heatwave", "timeseries", state_id="RAJASTHAN")
    assert chart["status"] == "ok"
    assert "data" in chart["data"]


def test_adapter_heatwave_comparison():
    res = compare_states("MAHARASHTRA", "RAJASTHAN", "heat_wave_days", {"start": 2024, "end": 2024})
    assert res["status"] == "ok"
    data = res["data"]
    assert data["state_a"]["state_id"] == "MAHARASHTRA"
    assert data["state_b"]["state_id"] == "RAJASTHAN"
    assert data["comparison"]["difference_b_minus_a"] > 0
    assert data["comparison"]["higher_state"] == "Rajasthan"
    res_temp = compare_states("MAHARASHTRA", "RAJASTHAN", "summer_max_temp", {"start": 2024, "end": 2024})
    assert res_temp["status"] == "ok"
    assert res_temp["data"]["state_a"]["value"] is not None
    assert res_temp["data"]["state_b"]["value"] is not None

def test_time_machine_and_report_heatwave_integration():
    tm = get_time_machine_data(2024)
    assert tm["status"] == "ok"
    assert "heatwaves" in tm["data"]
    assert len(tm["data"]["heatwaves"]) > 0
    assert tm["summary"]["heatwave_affected_states_count"] > 0

    rep = get_report_data("MAHARASHTRA")
    assert rep["status"] == "ok"
    assert "heatwave_summary" in rep["data"]
    assert rep["data"]["heatwave_summary"]["status"] == "ok"
    assert "imd_mospi_heatwave" in rep["meta"]["source_ids"]
