"""Multi-hazard compound analysis and cross-hazard connection explorer for ClimateLens India."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nielytics.hazards import MultiHazardAnalyzer

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    normalize_state_id,
)

logger = logging.getLogger(__name__)

SCIENTIFIC_CAVEAT = (
    "Cross-hazard co-occurrence represents descriptive historical association only. "
    "Do NOT infer direct causation from temporal coincidence across distinct observational networks."
)


def compute_state_composite_hazard_scores(
    state_id: str | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute normalized (0.0 to 1.0) and explainable composite risk indicators for states."""
    hw_path = PROCESSED_DATA_DIR / "state_heatwave.parquet"
    fl_path = PROCESSED_DATA_DIR / "flood_events.parquet"
    ls_path = PROCESSED_DATA_DIR / "landslide_events.parquet"
    rf_path = PROCESSED_DATA_DIR / "rainfall_daily.parquet"

    hw_by_state: dict[str, float] = {}
    fl_by_state: dict[str, float] = {}
    ls_by_state: dict[str, float] = {}
    rf_by_state: dict[str, float] = {}

    if hw_path.exists():
        hw_df = pd.read_parquet(hw_path)
        hw_by_state = hw_df.groupby("state_id")["total_heat_wave_days"].mean().to_dict()

    if fl_path.exists():
        fl_df = pd.read_parquet(fl_path)
        fl_by_state = fl_df.groupby("primary_state_id").size().to_dict()

    if ls_path.exists():
        ls_df = pd.read_parquet(ls_path)
        ls_by_state = ls_df.groupby("state_id").size().to_dict()

    if rf_path.exists():
        rf_df = pd.read_parquet(rf_path)
        rf_by_state = rf_df.groupby("state_id")["daily_actual_mm"].mean().to_dict()

    sid = normalize_state_id(state_id) if state_id else None
    target_sids = [sid] if sid and sid in STATES_MAP else list(STATES_MAP.keys())
    scores: dict[str, dict[str, Any]] = {}

    for sid in target_sids:
        meta = STATES_MAP.get(sid, {})
        metrics = {
            "heatwave_intensity": float(hw_by_state.get(sid, 0.0)),
            "flood_frequency": float(fl_by_state.get(sid, 0.0)),
            "landslide_density": float(ls_by_state.get(sid, 0.0)),
            "daily_rainfall_mm": float(rf_by_state.get(sid, 0.0)),
        }
        res = MultiHazardAnalyzer.composite_hazard_indicator(metrics, weights=weights)
        score_100 = float(res.data.get("composite_score", 0.0))
        scores[sid] = {
            "state_id": sid,
            "state_name": meta.get("state_name", sid),
            "composite_score_pct": score_100,
            "composite_score": round(score_100 / 100.0, 4),
            "normalized_metrics": res.data.get("normalized_metrics", {}),
            "raw_metrics": res.data.get("raw_metrics", {}),
            "weights": res.data.get("weights", {}),
            "mathematical_formula": res.data.get("mathematical_formula", ""),
        }

    return scores


def get_regional_hazard_overlap(region_type: str = "state") -> dict[str, Any]:
    """Produce cross-hazard exposure matrix across all 36 Indian States/UTs using MultiHazardAnalyzer."""
    hw_path = PROCESSED_DATA_DIR / "state_heatwave.parquet"
    fl_path = PROCESSED_DATA_DIR / "flood_events.parquet"
    ls_path = PROCESSED_DATA_DIR / "landslide_events.parquet"

    hw_states: list[str] = []
    fl_states: list[str] = []
    ls_states: list[str] = []

    if hw_path.exists():
        hw_df = pd.read_parquet(hw_path)
        hw_states = hw_df["state_name"].dropna().unique().tolist()
    if fl_path.exists():
        fl_df = pd.read_parquet(fl_path)
        fl_states = fl_df["primary_state_name"].dropna().unique().tolist()
    if ls_path.exists():
        ls_df = pd.read_parquet(ls_path)
        ls_states = ls_df["state_name"].dropna().unique().tolist()

    all_state_names = [v["state_name"] for v in STATES_MAP.values()]
    coastal_states = [
        "Odisha", "Andhra Pradesh", "West Bengal", "Tamil Nadu",
        "Gujarat", "Maharashtra", "Kerala", "Goa", "Karnataka"
    ]

    hazard_dfs = {
        "heatwave": pd.DataFrame({"state": hw_states}),
        "flood": pd.DataFrame({"state": fl_states}),
        "landslide": pd.DataFrame({"state": ls_states}),
        "rainfall": pd.DataFrame({"state": all_state_names}),
        "cyclone": pd.DataFrame({"state": coastal_states}),
    }

    res = MultiHazardAnalyzer.regional_multi_hazard_overlap(hazard_dfs, region_type=region_type)
    return res.data


def evaluate_rainfall_flood_correlation(
    state_id: str | None = None,
    year: int | None = None,
    antecedent_days: int = 3,
    limit_events: int = 50,
) -> dict[str, Any]:
    """Quantify 3-day antecedent precipitation volume preceding flood declaration dates."""
    rf_path = PROCESSED_DATA_DIR / "rainfall_daily.parquet"
    fl_path = PROCESSED_DATA_DIR / "flood_events.parquet"

    if not rf_path.exists() or not fl_path.exists():
        return {"total_flood_events_evaluated": 0, "correlations": []}

    rf_df = pd.read_parquet(rf_path)
    fl_df = pd.read_parquet(fl_path)

    sid = normalize_state_id(state_id) if state_id else None
    if sid:
        rf_df = rf_df[rf_df["state_id"] == sid]
        fl_df = fl_df[fl_df["primary_state_id"] == sid]
    if year:
        fl_df = fl_df[fl_df["year"] == year]

    if rf_df.empty or fl_df.empty:
        return {"total_flood_events_evaluated": 0, "correlations": []}

    r_sub = rf_df[["date", "daily_actual_mm"]].rename(columns={"daily_actual_mm": "rainfall"}).dropna()
    f_sub = fl_df[["start_date"]].dropna().head(limit_events)

    res = MultiHazardAnalyzer.compare_rainfall_and_flood(
        rainfall_data=r_sub,
        flood_events=f_sub,
        region=sid,
        rain_date_col="date",
        flood_date_col="start_date",
        antecedent_days=antecedent_days,
    )
    return res.data


def evaluate_rainfall_landslide_correlation(
    state_id: str | None = None,
    year: int | None = None,
    antecedent_days: int = 3,
    limit_events: int = 50,
) -> dict[str, Any]:
    """Quantify antecedent moisture saturation prior to slope failures."""
    rf_path = PROCESSED_DATA_DIR / "rainfall_daily.parquet"
    ls_path = PROCESSED_DATA_DIR / "landslide_events.parquet"

    if not rf_path.exists() or not ls_path.exists():
        return {"total_landslide_events_evaluated": 0, "correlations": []}

    rf_df = pd.read_parquet(rf_path)
    ls_df = pd.read_parquet(ls_path)

    sid = normalize_state_id(state_id) if state_id else None
    if sid:
        rf_df = rf_df[rf_df["state_id"] == sid]
        ls_df = ls_df[ls_df["state_id"] == sid]
    if year:
        ls_df = ls_df[ls_df["year"] == year]

    if rf_df.empty or ls_df.empty:
        return {"total_landslide_events_evaluated": 0, "correlations": []}

    r_sub = rf_df[["date", "daily_actual_mm"]].rename(columns={"daily_actual_mm": "rainfall"}).dropna()
    l_sub = ls_df[["event_date"]].dropna().head(limit_events)

    res = MultiHazardAnalyzer.compare_rainfall_and_landslides(
        rainfall_data=r_sub,
        landslide_events=l_sub,
        region=sid,
        rain_date_col="date",
        landslide_date_col="event_date",
        antecedent_days=antecedent_days,
    )
    return res.data


def evaluate_heat_drought_compound(
    state_id: str | None = None,
    temp_threshold_c: float = 40.0,
) -> dict[str, Any]:
    """Compute joint compound high-heat and dry conditions."""
    hw_path = PROCESSED_DATA_DIR / "state_heatwave.parquet"
    rf_path = PROCESSED_DATA_DIR / "rainfall_daily.parquet"

    if not hw_path.exists() or not rf_path.exists():
        return {"common_observation_days": 0, "compound_event_days": 0, "events": []}

    hw_df = pd.read_parquet(hw_path)
    rf_df = pd.read_parquet(rf_path)

    sid = normalize_state_id(state_id) if state_id else None
    if sid:
        hw_df = hw_df[hw_df["state_id"] == sid]
        rf_df = rf_df[rf_df["state_id"] == sid]

    # Synthesize daily heat proxy from summer max temperatures aligned to summer dates in rainfall
    if hw_df.empty or rf_df.empty:
        return {"common_observation_days": 0, "compound_event_days": 0, "events": []}

    mean_summer_tmax = float(hw_df["summer_max_temp_deg_c"].dropna().mean()) if hw_df["summer_max_temp_deg_c"].notna().any() else 38.0
    rf_dates = rf_df["date"].drop_duplicates().sort_values()

    heat_df = pd.DataFrame({
        "date": rf_dates,
        "tmax": [mean_summer_tmax] * len(rf_dates),
    })
    drought_df = rf_df[["date", "daily_actual_mm"]].rename(columns={"daily_actual_mm": "rainfall"}).dropna()

    res = MultiHazardAnalyzer.compare_heat_and_drought(
        heat_data=heat_df,
        drought_data=drought_df,
        region=sid,
        temp_threshold_c=temp_threshold_c,
    )
    return res.data


def investigate_multi_hazard(
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explore descriptive associations between overlapping historical hazard records.

    Examines temporal and spatial co-occurrence such as:
    - Cyclone coastal proximity accompanied by flood records in coastal states.
    - Landslide occurrences in states with high precipitation anomalies.
    Augmented with nielytics MultiHazardAnalyzer cross-hazard modeling and composite risk scores.
    """
    f = filters or {}
    year = f.get("year")
    state_id = f.get("state_id")

    sid = normalize_state_id(state_id) if state_id else None

    # Load flood events and cyclone tracks
    fl_path = PROCESSED_DATA_DIR / "flood_events.parquet"
    cy_path = PROCESSED_DATA_DIR / "cyclone_track_points.parquet"
    ls_path = PROCESSED_DATA_DIR / "landslide_events.parquet"

    associations: list[dict[str, Any]] = []

    if fl_path.exists() and cy_path.exists():
        fl_df = pd.read_parquet(fl_path)
        cy_df = pd.read_parquet(cy_path)

        # Focus on coastal storms that approached within 100km of Indian coast
        close_storms = cy_df[cy_df["dist2land_km"] <= 100.0]
        if year is not None:
            close_storms = close_storms[close_storms["season"] == year]
            fl_df = fl_df[fl_df["year"] == year]

        # Check overlapping storm periods with flood events
        for sid_storm, storm_points in close_storms.groupby("sid"):
            storm_name = storm_points.iloc[0]["name"]
            storm_season = int(storm_points.iloc[0]["season"])
            start_time = storm_points["iso_time"].min()[:10]
            end_time = storm_points["iso_time"].max()[:10]

            # Find floods during this window (+/- 3 days)
            matching_floods = fl_df[
                (fl_df["start_date"] >= start_time) &
                (fl_df["start_date"] <= end_time)
            ]
            if sid:
                matching_floods = matching_floods[matching_floods["primary_state_id"] == sid]

            if not matching_floods.empty:
                affected_states = matching_floods["primary_state_name"].dropna().unique().tolist()
                associations.append({
                    "hazard_pair": "Cyclone Coastal Proximity + Recorded Flood Event",
                    "year": storm_season,
                    "event_a": {
                        "hazard": "cyclone",
                        "name": storm_name,
                        "storm_id": sid_storm,
                        "period": f"{start_time} to {end_time}",
                        "min_distance_to_coast_km": float(storm_points["dist2land_km"].min()),
                        "peak_wind_kmh": float(storm_points["wmo_wind_kmh"].dropna().max()) if storm_points["wmo_wind_kmh"].notna().any() else None,
                    },
                    "event_b": {
                        "hazard": "flood",
                        "matching_flood_events_count": len(matching_floods),
                        "affected_states": affected_states,
                        "total_fatalities_reported": float(matching_floods["human_fatality"].sum()) if matching_floods["human_fatality"].notna().any() else None,
                    },
                    "association_type": "Temporal and regional co-occurrence",
                    "causation_established": False,
                    "note": f"Cyclone {storm_name} passed within {float(storm_points['dist2land_km'].min()):.0f} km of coast while {len(matching_floods)} flood events occurred in {', '.join(affected_states)}.",
                })

    # Landslide and topography co-occurrence
    if ls_path.exists():
        ls_df = pd.read_parquet(ls_path)
        if year is not None:
            ls_df = ls_df[ls_df["year"] == year]
        if sid:
            ls_df = ls_df[ls_df["state_id"] == sid]

        top_ls_states = ls_df["state_name"].value_counts().head(3)
        for sname, count in top_ls_states.items():
            associations.append({
                "hazard_pair": "Hilly / Ghat Terrain + Landslide Density",
                "year": year or "2004-2024",
                "event_a": {"hazard": "geomorphological_susceptibility", "region": sname},
                "event_b": {"hazard": "landslide", "incident_count": int(count)},
                "association_type": "Spatial concentration in steep relief zones",
                "causation_established": False,
                "note": f"State {sname} recorded {count} landslide events, reflecting high topography and monsoon susceptibility.",
            })

    # Compute nielytics advanced multi-hazard metrics
    composite_scores = compute_state_composite_hazard_scores(state_id=sid)
    regional_overlap = get_regional_hazard_overlap()
    rf_fl_res = evaluate_rainfall_flood_correlation(state_id=sid, year=year)
    rf_ls_res = evaluate_rainfall_landslide_correlation(state_id=sid, year=year)
    compound_hd = evaluate_heat_drought_compound(state_id=sid)

    nielytics_payload = {
        "regional_overlap": regional_overlap,
        "rainfall_flood_correlation": rf_fl_res,
        "rainfall_landslide_correlation": rf_ls_res,
        "compound_heat_drought": compound_hd,
    }

    return {
        "status": "ok" if (associations or composite_scores) else "no_data",
        "query_filters": f,
        "co_occurrences_found_count": len(associations),
        "associations": associations[:15],
        "composite_hazard_scores": composite_scores,
        "nielytics_multi_hazard": nielytics_payload,
        "scientific_caveat": SCIENTIFIC_CAVEAT,
    }
