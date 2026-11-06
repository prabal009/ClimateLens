"""ClimateLens India unified backend analytics adapter.

Fulfills Section 11 and Section 12 specifications, providing stable,
JSON-safe analytical methods adhering to the shared application data contract.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from nielytics import ClimateAnalytics
from nielytics.harmonization import HarmonizationSchema
from climatelens.constants import (
    DATASET_METADATA,
    PROCESSED_DATA_DIR,
    STATES_MAP,
    TEMPERATURE_REGIONS,
    normalize_state_id,
)
from climatelens.engine.comparison import compare_states as _compare_states_engine
from climatelens.engine.insight_engine import generate_insights as _generate_insights_engine
from climatelens.engine.multi_hazard import (
    compute_state_composite_hazard_scores,
    evaluate_heat_drought_compound,
    evaluate_rainfall_flood_correlation,
    evaluate_rainfall_landslide_correlation,
    get_regional_hazard_overlap,
    investigate_multi_hazard as _multi_hazard_engine,
)
from climatelens.engine.forecasting import (
    forecast_summer_heatwaves as _forecast_summer_heatwaves,
    get_state_cmip6_projections as _get_state_cmip6_projections,
)
from climatelens.engine.time_machine import get_time_machine_data as _time_machine_engine
from climatelens.geospatial.spatial_join import get_boundary_gdf
from climatelens.hazards.cyclone_hazard import (
    analyze_cyclone_hazard,
    get_storm_track,
)
from climatelens.hazards.drought_hazard import analyze_national_drought
from climatelens.hazards.flood_hazard import analyze_flood_hazard
from climatelens.hazards.charts import generate_hazard_chart
from climatelens.hazards.heat_hazard import (
    analyze_regional_temperature,
    analyze_state_heatwaves,
    get_state_heatwave_summary_for_year,
)
from climatelens.hazards.landslide_hazard import analyze_landslide_hazard
from climatelens.hazards.rainfall_hazard import (
    analyze_state_rainfall,
    get_rainfall_choropleth,
    get_state_rainfall_timeseries,
)



def init_climate_analytics() -> ClimateAnalytics:
    """Initialize unified ClimateAnalytics facade and register preprocessed hazard datasets."""
    ca = ClimateAnalytics(offline=True)
    try:
        # 1. Heatwave
        hw_path = PROCESSED_DATA_DIR / "state_heatwave.parquet"
        if hw_path.exists():
            hw_df = pd.read_parquet(hw_path)
            hw_input = pd.DataFrame({
                "date": pd.to_datetime(hw_df["year"].astype(str) + "-05-15"),
                "state": hw_df["state_name"],
                "tmax": hw_df["summer_max_temp_deg_c"].astype(float),
            }).dropna()
            schema_hw = HarmonizationSchema(hazard="heatwave", metric="tmax", value_column="tmax", state_column="state")
            ca.register_dataset("heatwave", hw_input, schema=schema_hw)

        # 2. Rainfall
        rf_path = PROCESSED_DATA_DIR / "rainfall_daily.parquet"
        if rf_path.exists():
            rf_df = pd.read_parquet(rf_path)
            rf_input = rf_df[["date", "entity_name", "daily_actual_mm"]].rename(
                columns={"daily_actual_mm": "rainfall", "entity_name": "state"}
            ).dropna()
            schema_rf = HarmonizationSchema(hazard="flood", metric="rainfall", value_column="rainfall", state_column="state")
            ca.register_dataset("rainfall", rf_input, schema=schema_rf)

        # 3. Flood
        fl_path = PROCESSED_DATA_DIR / "flood_events.parquet"
        if fl_path.exists():
            fl_df = pd.read_parquet(fl_path)
            fl_input = fl_df[["start_date", "primary_state_name", "human_fatality"]].rename(
                columns={"start_date": "date", "primary_state_name": "state", "human_fatality": "fatalities"}
            ).dropna()
            schema_fl = HarmonizationSchema(hazard="flood", metric="fatalities", value_column="fatalities", state_column="state")
            ca.register_dataset("flood", fl_input, schema=schema_fl)

        # 4. Landslide
        ls_path = PROCESSED_DATA_DIR / "landslide_events.parquet"
        if ls_path.exists():
            ls_df = pd.read_parquet(ls_path)
            ls_input = pd.DataFrame({
                "date": ls_df["event_date"],
                "state": ls_df["state_name"],
                "events": 1.0,
            }).dropna()
            schema_ls = HarmonizationSchema(hazard="landslide", metric="events", value_column="events", state_column="state")
            ca.register_dataset("landslide", ls_input, schema=schema_ls)

        # 5. Cyclone
        cy_path = PROCESSED_DATA_DIR / "cyclone_track_points.parquet"
        if cy_path.exists():
            cy_df = pd.read_parquet(cy_path)
            cy_close = cy_df[cy_df["dist2land_km"] <= 100.0]
            cy_input = pd.DataFrame({
                "date": cy_close["iso_time"].str[:10],
                "wind": cy_close["wmo_wind_kmh"].fillna(cy_close["usa_wind_kmh"]),
                "latitude": cy_close["latitude"],
                "longitude": cy_close["longitude"],
            }).dropna().head(500)
            schema_cy = HarmonizationSchema(
                hazard="cyclone",
                metric="wind",
                value_column="wind",
                latitude_column="latitude",
                longitude_column="longitude",
            )
            ca.register_dataset("cyclone", cy_input, schema=schema_cy)
    except Exception as exc:
        logger.warning("ClimateAnalytics initialization warning: %s", exc)

    return ca


# Initialize unified facade on startup
analytics_facade = init_climate_analytics()


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types, Timestamps, and NaNs into standard JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)[:19].replace("T", " ")
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    return str(obj)


def get_available_metadata() -> dict[str, Any]:
    """Retrieve metadata, available dates, and spatial boundaries for all 7 datasets."""
    manifest_p = PROCESSED_DATA_DIR / "manifest.json"
    if manifest_p.exists():
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    else:
        from climatelens.preprocessing.manifest_builder import generate_manifest
        manifest_data = generate_manifest()

    return _sanitize_for_json({
        "status": "ok",
        "data": manifest_data,
        "meta": {
            "metric": "metadata_manifest",
            "unit": "catalog",
            "geographic_level": "national",
            "time_resolution": "multi-resolution",
            "period": {"start": "1842", "end": "2026"},
            "source_ids": list(DATASET_METADATA.keys()),
            "method": "direct catalog inspection",
            "observation_count": len(DATASET_METADATA),
            "limitations": ["Metadata describes verified historical archives and processing provenance."],
        },
    })


def get_state_boundaries(simplified: bool = True) -> dict[str, Any]:
    """Retrieve validated GeoJSON FeatureCollection of 36 Indian States and Union Territories."""
    filename = "state_boundaries_simplified.geojson" if simplified else "state_boundaries.geojson"
    path = PROCESSED_DATA_DIR / filename

    if not path.exists():
        from climatelens.preprocessing.boundaries import process_boundaries
        clean_gdf, simp_gdf = process_boundaries()
        target_gdf = simp_gdf if simplified else clean_gdf
        geojson_dict = json.loads(target_gdf.to_json())
    else:
        with open(path, "r", encoding="utf-8") as f:
            geojson_dict = json.load(f)

    return _sanitize_for_json({
        "status": "ok",
        "data": geojson_dict,
        "meta": {
            "metric": "state_boundaries_geojson",
            "unit": "geographic_features",
            "geographic_level": "state",
            "time_resolution": "year",
            "period": {"start": "2025", "end": "2025"},
            "source_ids": ["nwic_state_boundary"],
            "method": "topological boundary extraction (0.01 deg tolerance)" if simplified else "full resolution Survey of India boundary",
            "observation_count": len(geojson_dict.get("features", [])),
            "limitations": ["EPSG:4326 coordinate reference system. Simplified geometry for fast dashboard rendering."],
        },
    })


def get_rainfall_map(year: int = 2026) -> dict[str, Any]:
    """Retrieve state-wise rainfall choropleth layer with departure categories and color codes."""
    res = get_rainfall_choropleth(year=year)
    states_data = res.get("states", [])
    has_data = any(s.get("monsoon_category") != "No Data" for s in states_data)

    return _sanitize_for_json({
        "status": "ok" if has_data else "no_data",
        "data": states_data,
        "meta": {
            "metric": "rainfall_departure_category",
            "unit": "category / %",
            "geographic_level": "state",
            "time_resolution": "season",
            "period": {"start": f"{year}-06-01", "end": f"{year}-09-30"},
            "source_ids": ["imd_rainfall_daily"],
            "method": "Cumulative seasonal departure from normal: ((Actual - Normal) / Normal) * 100",
            "observation_count": sum(1 for s in states_data if s.get("monsoon_category") != "No Data"),
            "legend": res.get("legend", {}),
            "limitations": ["Represents seasonal cumulative rainfall as of September 30 from IMD NWIC portal."],
        },
    })


def get_state_rainfall_trend(
    state_id: str,
    start_year: int = 2022,
    end_year: int = 2026,
) -> dict[str, Any]:
    """Retrieve time-ordered rainfall timeseries and analytical indicators for a specific state."""
    sid = normalize_state_id(state_id)
    if not sid or sid not in STATES_MAP:
        return {
            "status": "no_data",
            "data": [],
            "meta": {
                "metric": "state_rainfall_trend",
                "unit": "mm",
                "geographic_level": "state",
                "time_resolution": "day",
                "period": {"start": str(start_year), "end": str(end_year)},
                "source_ids": ["imd_rainfall_daily"],
                "method": "timeseries extraction",
                "observation_count": 0,
                "limitations": [f"Unrecognized or invalid state_id '{state_id}'."],
            },
        }

    timeseries = get_state_rainfall_timeseries(sid, start_year, end_year)
    state_meta = STATES_MAP[sid]

    return _sanitize_for_json({
        "status": "ok" if timeseries else "no_data",
        "data": {
            "state_id": sid,
            "state_name": state_meta["state_name"],
            "records": timeseries,
        },
        "meta": {
            "metric": "daily_and_cumulative_rainfall",
            "unit": "mm",
            "geographic_level": "state",
            "time_resolution": "day",
            "period": {"start": str(start_year), "end": str(end_year)},
            "source_ids": ["imd_rainfall_daily"],
            "method": "daily observations and cumulative series from IMD portal",
            "observation_count": len(timeseries),
            "limitations": ["Data reflects daily snapshot dates available in IMD NWIC portal dataset."],
        },
    })


def get_national_drought_trend(
    start_year: int = 2021,
    end_year: int = 2026,
) -> dict[str, Any]:
    """Retrieve national drought area percentage timeseries and severity transitions."""
    res = analyze_national_drought(start_year=start_year, end_year=end_year)
    if res.get("status") == "no_data":
        return {
            "status": "no_data",
            "data": [],
            "meta": {
                "metric": "drought_area_percentage",
                "unit": "%",
                "geographic_level": "national",
                "time_resolution": "week",
                "period": {"start": str(start_year), "end": str(end_year)},
                "source_ids": ["iitgn_national_drought"],
                "method": "weekly timeseries aggregation",
                "observation_count": 0,
                "limitations": ["National aggregate timeseries from IIT Gandhinagar."],
            },
        }

    return _sanitize_for_json({
        "status": "ok",
        "data": res,
        "meta": {
            "metric": "drought_area_percentage",
            "unit": "%",
            "geographic_level": "national",
            "time_resolution": "week",
            "period": {"start": str(start_year), "end": str(end_year)},
            "source_ids": ["iitgn_national_drought"],
            "method": "weekly national drought monitor area aggregation",
            "observation_count": res.get("observation_weeks_count", 0),
            "limitations": [
                "Strictly national-level drought percentages; does NOT provide state-level data.",
                "Sum invariant (no_drought + d0_d4 == 100%) verified within 0.05% tolerance.",
            ],
        },
    })


def get_regional_temperature_trend(
    region: str = "ALLIN",
    start_year: int = 1901,
    end_year: int = 2007,
) -> dict[str, Any]:
    """Retrieve historical regional maximum temperature and 1971-2000 anomaly trend."""
    try:
        res = analyze_regional_temperature(region_code=region, start_year=start_year, end_year=end_year)
    except ValueError as exc:
        return {
            "status": "no_data",
            "data": [],
            "meta": {
                "metric": "maximum_temperature_anomaly",
                "unit": "°C",
                "geographic_level": "regional",
                "time_resolution": "month",
                "period": {"start": str(start_year), "end": str(end_year)},
                "source_ids": ["iitm_regional_temperature"],
                "method": "monthly anomaly vs 1971-2000 baseline normal",
                "observation_count": 0,
                "limitations": [str(exc)],
            },
        }

    return _sanitize_for_json({
        "status": res.get("status", "ok"),
        "data": res,
        "meta": {
            "metric": "maximum_temperature_and_anomaly",
            "unit": "°C",
            "geographic_level": "regional",
            "time_resolution": "month",
            "period": {"start": str(start_year), "end": str(end_year)},
            "source_ids": ["iitm_regional_temperature"],
            "method": "monthly anomaly relative to 1971-2000 climatological normal",
            "observation_count": res.get("observation_count", 0),
            "limitations": [
                "Monthly mean maximum temperatures are regional climate indicators, NOT daily heatwave declarations.",
                "Published for 8 homogeneous climatic regions, not individual states.",
            ],
        },
    })


def get_state_heatwave_trend(
    state_id: str,
    start_year: int = 1967,
    end_year: int = 2024,
) -> dict[str, Any]:
    """Retrieve annual state heatwave and severe heatwave occurrences from IMD/MoSPI Envistats (1967-2024)."""
    sid = normalize_state_id(state_id)
    if not sid or sid not in STATES_MAP:
        return {
            "status": "no_data",
            "data": {},
            "meta": {
                "metric": "heat_wave_days",
                "unit": "days",
                "geographic_level": "state",
                "time_resolution": "annual",
                "period": {"start": str(start_year), "end": str(end_year)},
                "source_ids": ["imd_mospi_heatwave"],
                "method": "annual recorded heatwave days from IMD / MoSPI Envistats",
                "observation_count": 0,
                "limitations": [f"Invalid state_id '{state_id}'."],
            },
        }

    res = analyze_state_heatwaves(sid, start_year=start_year, end_year=end_year)
    return _sanitize_for_json({
        "status": res.get("status", "ok"),
        "data": res,
        "meta": {
            "metric": "heat_wave_and_severe_days",
            "unit": "days",
            "geographic_level": "state",
            "time_resolution": "annual",
            "period": {"start": str(start_year), "end": str(end_year)},
            "source_ids": ["imd_mospi_heatwave"],
            "method": "annual summer heatwave records and peak maximum temperatures",
            "observation_count": res.get("observation_years", 0),
            "limitations": [
                "Official IMD/MoSPI records for summer season (April-June).",
                "Mountainous/northeastern states have null summer peak temperatures where stations are absent.",
            ],
        },
    })


def get_heatwave_map(year: int = 2024) -> dict[str, Any]:
    """Retrieve state-wise choropleth layer of heatwave days and summer peak temperatures for a single year."""
    records = get_state_heatwave_summary_for_year(year=year)
    return _sanitize_for_json({
        "status": "ok" if records else "no_data",
        "data": {
            "year": year,
            "states": records,
        },
        "meta": {
            "metric": "state_heatwave_choropleth",
            "unit": "days",
            "geographic_level": "state",
            "time_resolution": "annual",
            "period": {"year": str(year)},
            "source_ids": ["imd_mospi_heatwave"],
            "method": "summer season state heatwave compilation",
            "observation_count": len(records),
            "limitations": [
                "States without recorded heatwaves report 0 heatwave days.",
            ],
        },
    })


def get_hazard_chart(
    hazard: str,
    chart_type: str,
    state_id: str | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    """Retrieve ready-to-render Plotly chart specifications generated directly by nielytics.core.visualization."""
    try:
        spec = generate_hazard_chart(hazard=hazard, chart_type=chart_type, state_id=state_id, year=year)
        status = "ok"
        limitations = ["Interactive Plotly specification generated via nielytics.core.visualization."]
    except Exception as exc:
        spec = {"data": [], "layout": {"title": f"Chart error: {exc}"}}
        status = "error"
        limitations = [str(exc)]

    return _sanitize_for_json({
        "status": status,
        "data": spec,
        "meta": {
            "hazard": hazard,
            "chart_type": chart_type,
            "state_id": normalize_state_id(state_id) if state_id else None,
            "year": year,
            "source_ids": ["nielytics_core_visualization"],
            "method": "Plotly chart generation using author's nielytics library",
            "limitations": limitations,
        },
    })


def get_flood_events(filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Retrieve cataloged flood events matching optional state or temporal filters."""
    f = filters or {}
    state_id = f.get("state_id")
    start_year = f.get("start_year")
    end_year = f.get("end_year")
    min_fatalities = f.get("min_fatalities")

    res = analyze_flood_hazard(
        state_id=state_id,
        start_year=start_year,
        end_year=end_year,
        min_fatalities=min_fatalities,
    )

    return _sanitize_for_json({
        "status": res.get("status", "ok"),
        "data": res,
        "meta": {
            "metric": "flood_event_inventory",
            "unit": "events / fatalities",
            "geographic_level": "state" if state_id else "national",
            "time_resolution": "event",
            "period": res.get("period", {"start": "1967", "end": "2023"}),
            "source_ids": ["iitd_flood_inventory"],
            "method": "IFI-Impacts historical catalog event aggregation",
            "observation_count": res.get("total_events_recorded", 0),
            "limitations": [
                "Blank casualty/impact fields are retained as null; zero casualties are never imputed.",
                "Coordinates are absent in source CSV; events mapped to primary state and districts.",
            ],
        },
    })


def get_landslide_events(filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Retrieve landslide event locations, spatial clusters, and frequencies."""
    f = filters or {}
    state_id = f.get("state_id")
    year = f.get("year")

    res = analyze_landslide_hazard(state_id=state_id, year=year)

    return _sanitize_for_json({
        "status": res.get("status", "ok"),
        "data": res,
        "meta": {
            "metric": "landslide_incidents",
            "unit": "incident_points",
            "geographic_level": "point",
            "time_resolution": "year",
            "period": res.get("period", {"start": "2004", "end": "2024"}),
            "source_ids": ["riseuppant_landslide_inventory"],
            "method": "spatial join against NWIC state polygons",
            "observation_count": res.get("total_events_count", 0),
            "limitations": [
                "Community-hosted GitHub source; original scientific provenance is unverified.",
                "Reported magnitude scale is uncalibrated and used for frequency/density only.",
            ],
        },
    })


def get_cyclone_list(year: int | None = None) -> dict[str, Any]:
    """Retrieve list of historical cyclone storms in the North Indian Ocean basin."""
    res = analyze_cyclone_hazard(year=year)

    return _sanitize_for_json({
        "status": res.get("status", "ok"),
        "data": res.get("storms", []),
        "meta": {
            "metric": "cyclone_storm_catalog",
            "unit": "storms",
            "geographic_level": "regional",
            "time_resolution": "season",
            "period": {"start": str(year) if year else "1842", "end": str(year) if year else "2024"},
            "source_ids": ["noaa_ibtracs_ni"],
            "method": "NOAA IBTrACS best-track grouping by storm SID",
            "observation_count": res.get("total_storms_count", 0),
            "limitations": [
                "Covers North Indian Ocean (NI) basin (Bay of Bengal and Arabian Sea).",
                "Proximity to coast does not constitute confirmed meteorological landfall.",
            ],
        },
    })


def get_cyclone_track(storm_id: str) -> dict[str, Any]:
    """Retrieve chronologically ordered coordinates, intensity, and pressure for a cyclone track."""
    res = get_storm_track(storm_id)

    return _sanitize_for_json({
        "status": res.get("status", "ok"),
        "data": res,
        "meta": {
            "metric": "cyclone_track_points",
            "unit": "track_coordinates / km/h / hPa",
            "geographic_level": "point",
            "time_resolution": "event",
            "period": {"start": res.get("start_time", "unknown"), "end": res.get("end_time", "unknown")},
            "source_ids": ["noaa_ibtracs_ni"],
            "method": "successive track coordinate sequence and NIElytics CycloneAnalyzer",
            "observation_count": res.get("observation_count", 0),
            "limitations": [
                "Real coordinates only; visual interpolation is flagged is_interpolated: False.",
                "WMO and USA wind metrics represent distinct agency averaging periods.",
            ],
        },
    })


def get_time_machine_data(year: int) -> dict[str, Any]:
    """Assemble chronological time-ordered multi-hazard event feed for historical replay."""
    res = _time_machine_engine(year)

    return _sanitize_for_json({
        "status": res.get("status", "ok"),
        "data": res.get("data", {}),
        "summary": res.get("summary", {}),
        "meta": {
            "metric": "time_machine_historical_replay",
            "unit": "multi-hazard_event_feed",
            "geographic_level": "national",
            "time_resolution": "event",
            "period": {"start": f"{year}-01-01", "end": f"{year}-12-31"},
            "source_ids": ["noaa_ibtracs_ni", "iitd_flood_inventory", "riseuppant_landslide_inventory", "imd_rainfall_daily"],
            "method": "chronological cross-dataset time-ordered aggregation",
            "observation_count": res.get("summary", {}).get("total_disaster_events", 0),
            "limitations": [
                "Preserves original timestamps and coordinates without synthetic intermediate positions.",
                "Cyclone track coordinates represent ocean and overland track points.",
            ],
        },
    })


def compare_states(
    state_a: str,
    state_b: str,
    metric: str,
    period: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two Indian states on an identical metric within the same temporal window."""
    try:
        res = _compare_states_engine(state_a, state_b, metric, period)
    except ValueError as exc:
        return _sanitize_for_json({
            "status": "no_data",
            "message": str(exc),
            "data": {},
            "meta": {
                "metric": f"state_comparison_{metric}",
                "unit": "dimensionless",
                "geographic_level": "state",
                "time_resolution": "multi-year",
                "period": period or {},
                "source_ids": ["imd_rainfall_daily", "iitd_flood_inventory", "riseuppant_landslide_inventory"],
                "method": "standardized comparative ratio and percentage difference",
                "observation_count": 0,
                "limitations": [
                    "Division by zero is strictly protected.",
                    "Comparison of national drought or regional temperature at state level is prohibited.",
                ],
            },
        })

    return _sanitize_for_json({
        "status": res.get("status", "ok"),
        "data": res,
        "meta": {
            "metric": f"state_comparison_{metric}",
            "unit": res.get("unit", "dimensionless"),
            "geographic_level": "state",
            "time_resolution": "multi-year",
            "period": res.get("period", {}),
            "source_ids": ["imd_rainfall_daily", "iitd_flood_inventory", "riseuppant_landslide_inventory"],
            "method": "standardized comparative ratio and percentage difference",
            "observation_count": (res.get("state_a", {}).get("observation_count", 0) + res.get("state_b", {}).get("observation_count", 0)),
            "limitations": [
                "Division by zero is strictly protected.",
                "Comparison of national drought or regional temperature at state level is prohibited.",
            ],
        },
    })


def generate_insights(
    state_id: str | None = None,
    period: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate explainable, evidence-grounded insights strictly derived from verified measurements."""
    insights_list = _generate_insights_engine(state_id=state_id, period=period)

    return _sanitize_for_json({
        "status": "ok" if insights_list else "no_data",
        "data": insights_list,
        "meta": {
            "metric": "explainable_climate_insights",
            "unit": "insights",
            "geographic_level": "state" if state_id else "national",
            "time_resolution": "multi-year",
            "period": period or {"start": "1901", "end": "2026"},
            "source_ids": list(DATASET_METADATA.keys()),
            "method": "deterministic statistical ranking and anomaly detection",
            "observation_count": len(insights_list),
            "limitations": [
                "Insights are mathematically computed from verified datasets without LLM hallucination.",
                "No external AI API keys required for insight generation.",
            ],
        },
    })


def investigate_multi_hazard(filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Investigate descriptive associations between overlapping historical hazard records."""
    res = _multi_hazard_engine(filters)

    return _sanitize_for_json({
        "status": res.get("status", "ok"),
        "data": res.get("associations", []),
        "meta": {
            "metric": "multi_hazard_co_occurrence",
            "unit": "associations",
            "geographic_level": "national",
            "time_resolution": "event",
            "period": {"start": "1967", "end": "2024"},
            "source_ids": ["noaa_ibtracs_ni", "iitd_flood_inventory", "riseuppant_landslide_inventory"],
            "method": "spatiotemporal intersection of disaster records",
            "observation_count": res.get("co_occurrences_found_count", 0),
            "limitations": [
                "Descriptive co-occurrence only; direct causation is NOT inferred.",
            ],
        },
    })


def get_report_data(
    state_id: str,
    period: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate consolidated multi-hazard climate risk profile for a single state."""
    sid = normalize_state_id(state_id)
    if not sid or sid not in STATES_MAP:
        return {
            "status": "no_data",
            "data": {},
            "meta": {
                "metric": "state_comprehensive_report",
                "unit": "report",
                "geographic_level": "state",
                "time_resolution": "multi-year",
                "period": period or {},
                "source_ids": list(DATASET_METADATA.keys()),
                "method": "state-level multi-hazard profile aggregation",
                "observation_count": 0,
                "limitations": [f"Invalid state_id '{state_id}'."],
            },
        }

    state_meta = STATES_MAP[sid]
    rainfall_info = analyze_state_rainfall(sid)
    flood_info = analyze_flood_hazard(state_id=sid)
    landslide_info = analyze_landslide_hazard(state_id=sid)
    heatwave_info = analyze_state_heatwaves(sid)
    insights_info = _generate_insights_engine(state_id=sid)

    composite_scores = compute_state_composite_hazard_scores(state_id=sid)
    multi_hazard_assessment = {
        "composite_hazard_score": composite_scores.get(sid, {}),
        "rainfall_flood_correlation": evaluate_rainfall_flood_correlation(state_id=sid),
        "rainfall_landslide_correlation": evaluate_rainfall_landslide_correlation(state_id=sid),
        "compound_heat_drought": evaluate_heat_drought_compound(state_id=sid),
    }

    report_payload = {
        "state_id": sid,
        "state_name": state_meta["state_name"],
        "state_code": state_meta["state_code"],
        "census_stcode": state_meta["stcode"],
        "region": state_meta["region"],
        "type": state_meta["type"],
        "rainfall_summary": rainfall_info,
        "flood_summary": flood_info,
        "landslide_summary": landslide_info,
        "heatwave_summary": heatwave_info,
        "insights": insights_info,
        "climate_projections": _get_state_cmip6_projections(sid),
        "temperature_forecast": _forecast_summer_heatwaves(sid, horizon_days=7),
        "multi_hazard_assessment": multi_hazard_assessment,
    }
    return _sanitize_for_json({
        "status": "ok",
        "data": report_payload,
        "meta": {
            "metric": "state_comprehensive_report",
            "unit": "multi-metric",
            "geographic_level": "state",
            "time_resolution": "multi-year",
            "period": period or {"start": "1967", "end": "2026"},
            "source_ids": [
                "imd_rainfall_daily",
                "iitd_flood_inventory",
                "nasa_landslide_catalog",
                "imd_mospi_heatwave",
                "cmip6_south_asia",
                "nielytics_v2_forecasting",
            ],
            "method": "consolidated state profile assembly",
            "observation_count": (
                rainfall_info.get("observed_days_count", 0) +
                flood_info.get("total_events_recorded", 0) +
                landslide_info.get("total_events_count", 0) +
                heatwave_info.get("observation_years", 0)
            ),
            "limitations": [
                "Combines independent observational datasets.",
                "National drought is excluded from state report to prevent false spatial attribution.",
            ],
        },
    })


def get_climate_projections(state_id: str) -> dict[str, Any]:
    """Retrieve state-specific CMIP6 SSP scenarios (2030, 2040, 2050) and Theil-Sen warming rates."""
    sid = normalize_state_id(state_id)
    if not sid or sid not in STATES_MAP:
        return _sanitize_for_json({
            "status": "no_data",
            "data": {},
            "meta": {
                "metric": "cmip6_climate_projections",
                "unit": "degrees_celsius",
                "geographic_level": "state",
                "limitations": [f"Unrecognized or invalid state_id '{state_id}'."],
            },
        })
    try:
        res = _get_state_cmip6_projections(sid)
        if res.get("status") == "no_data":
            return _sanitize_for_json({
                "status": "no_data",
                "data": res,
                "meta": {
                    "metric": "cmip6_climate_projections",
                    "unit": "degrees_celsius",
                    "geographic_level": "state",
                    "limitations": [res.get("message", "No historical summer temperature observations recorded.")],
                },
            })
        return _sanitize_for_json({
            "status": "ok",
            "data": res,
            "meta": {
                "metric": "cmip6_climate_projections",
                "unit": "degrees_celsius",
                "geographic_level": "state",
                "scenarios": ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"],
                "target_years": [2030, 2040, 2050],
            },
        })
    except Exception as exc:
        logger.exception("Error calculating climate projections for %s: %s", state_id, exc)
        return _sanitize_for_json({
            "status": "error",
            "message": str(exc),
            "state_id": state_id,
        })


def get_temperature_forecast(state_id: str, horizon: int = 7) -> dict[str, Any]:
    """Retrieve short-term predictive temperature forecasts with 80%/95% prediction intervals and heatwave risk."""
    sid = normalize_state_id(state_id)
    if not sid or sid not in STATES_MAP:
        return _sanitize_for_json({
            "status": "no_data",
            "data": {},
            "meta": {
                "metric": "temperature_forecast",
                "unit": "degrees_celsius",
                "geographic_level": "state",
                "horizon_days": horizon,
                "limitations": [f"Unrecognized or invalid state_id '{state_id}'."],
            },
        })
    try:
        res = _forecast_summer_heatwaves(sid, horizon_days=horizon)
        if res.get("status") == "no_data":
            return _sanitize_for_json({
                "status": "no_data",
                "data": res,
                "meta": {
                    "metric": "temperature_forecast",
                    "unit": "degrees_celsius",
                    "geographic_level": "state",
                    "horizon_days": horizon,
                    "limitations": [res.get("message", "No historical summer temperature observations recorded.")],
                },
            })
        return _sanitize_for_json({
            "status": "ok",
            "data": res,
            "meta": {
                "metric": "temperature_forecast",
                "unit": "degrees_celsius",
                "geographic_level": "state",
                "horizon_days": horizon,
                "confidence_intervals": ["80%", "95%"],
            },
        })
    except Exception as exc:
        logger.exception("Error generating temperature forecast for %s: %s", state_id, exc)
        return _sanitize_for_json({
            "status": "error",
            "message": str(exc),
            "state_id": state_id,
        })


def get_multi_hazard_assessment(
    state_id: str | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    """Retrieve cross-hazard correlation and composite multi-hazard index from MultiHazardAnalyzer."""
    if state_id is not None:
        sid = normalize_state_id(state_id)
        if not sid or sid not in STATES_MAP:
            return _sanitize_for_json({
                "status": "no_data",
                "data": {},
                "meta": {
                    "metric": "multi_hazard_assessment",
                    "unit": "composite_index",
                    "geographic_level": "state",
                    "limitations": [f"Unrecognized or invalid state_id '{state_id}'."],
                },
            })
    else:
        sid = None
    try:
        composite_scores = compute_state_composite_hazard_scores(state_id=sid)
        regional_overlap = get_regional_hazard_overlap()
        rf_fl = evaluate_rainfall_flood_correlation(state_id=sid, year=year)
        rf_ls = evaluate_rainfall_landslide_correlation(state_id=sid, year=year)
        compound_hd = evaluate_heat_drought_compound(state_id=sid)

        payload = {
            "state_id": sid,
            "year": year,
            "composite_hazard_scores": composite_scores,
            "regional_multi_hazard_overlap": regional_overlap,
            "rainfall_flood_correlation": rf_fl,
            "rainfall_landslide_correlation": rf_ls,
            "compound_heat_drought": compound_hd,
            "scientific_caveat": (
                "Multi-hazard compound indicators and composite scores are exploratory screening proxies, "
                "not official civil defense alerts or actuarial certifications."
            ),
        }
        return _sanitize_for_json({
            "status": "ok",
            "data": payload,
            "meta": {
                "metric": "multi_hazard_assessment",
                "unit": "composite_index",
                "geographic_level": "state" if sid else "national",
            },
        })
    except Exception as exc:
        logger.exception("Error evaluating multi-hazard assessment: %s", exc)
        return _sanitize_for_json({
            "status": "error",
            "message": str(exc),
        })
