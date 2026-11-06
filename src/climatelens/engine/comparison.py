"""State-vs-State analytical comparison engine for ClimateLens India."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    normalize_state_id,
)

logger = logging.getLogger(__name__)

SUPPORTED_METRICS = {
    "annual_rainfall": {
        "label": "Monsoon Cumulative Rainfall",
        "unit": "mm",
        "dataset": "rainfall",
        "field": "monsoon_cumulative_actual_mm",
    },
    "rainfall_anomaly": {
        "label": "Rainfall Departure from Normal",
        "unit": "%",
        "dataset": "rainfall",
        "field": "monsoon_departure_pct",
    },
    "flood_event_count": {
        "label": "Historical Flood Events Count",
        "unit": "events",
        "dataset": "flood",
    },
    "landslide_event_count": {
        "label": "Historical Landslide Incidents Count",
        "unit": "incidents",
        "dataset": "landslide",
    },
    "heat_wave_days": {
        "label": "Total Heatwave Days",
        "field": "total_heat_wave_days",
        "unit": "days",
        "dataset": "heatwave",
    },
    "summer_max_temp": {
        "label": "Summer Peak Maximum Temperature",
        "field": "summer_max_temp_deg_c",
        "unit": "°C",
        "dataset": "heatwave",
    },
}

PROHIBITED_METRICS = {
    "national_drought": "Drought monitor data is exclusively national-level; state-wise disaggregation is scientifically prohibited.",
    "drought_area_pct": "Drought monitor data is exclusively national-level; state-wise disaggregation is scientifically prohibited.",
    "regional_temperature": "IITM temperature data is published for 8 homogeneous climatic regions, not individual states.",
}


def compare_states(
    state_a: str,
    state_b: str,
    metric: str,
    period: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two Indian states on a standardized climate or disaster metric.

    Enforces:
    - Same temporal window
    - Same unit and measurement definition
    - Protected zero-division handling for percentage change
    - Scientific guardrail: rejects state comparison for national drought or regional temperature
    """
    period_dict = period or {}
    start_year = period_dict.get("start")
    end_year = period_dict.get("end")

    # Guardrail 1: Prohibited metrics
    metric_clean = metric.lower().strip()
    if metric_clean in PROHIBITED_METRICS:
        return {
            "status": "no_data",
            "state_a": state_a,
            "state_b": state_b,
            "metric": metric,
            "message": f"State-vs-state comparison is scientifically prohibited for '{metric}'.",
            "scientific_rationale": PROHIBITED_METRICS[metric_clean],
        }

    # Guardrail 2: Supported metric verification
    if metric_clean not in SUPPORTED_METRICS:
        return {
            "status": "no_data",
            "metric": metric,
            "message": f"Unsupported comparison metric '{metric}'. Supported metrics: {list(SUPPORTED_METRICS.keys())}",
        }

    # Guardrail 3: State normalization
    sid_a = normalize_state_id(state_a)
    sid_b = normalize_state_id(state_b)
    if not sid_a or sid_a not in STATES_MAP:
        raise ValueError(f"Invalid state_a: '{state_a}'")
    if not sid_b or sid_b not in STATES_MAP:
        raise ValueError(f"Invalid state_b: '{state_b}'")

    meta_a = STATES_MAP[sid_a]
    meta_b = STATES_MAP[sid_b]
    metric_meta = SUPPORTED_METRICS[metric_clean]

    val_a: float | None = None
    val_b: float | None = None
    obs_count_a: int = 0
    obs_count_b: int = 0
    actual_start: int | None = start_year
    actual_end: int | None = end_year

    if metric_meta["dataset"] == "rainfall":
        rf_path = PROCESSED_DATA_DIR / "rainfall_state_year.parquet"
        if not rf_path.exists():
            from climatelens.preprocessing.rainfall import process_rainfall
            process_rainfall()
        rf_df = pd.read_parquet(rf_path)

        yr = int(end_year) if end_year else (int(start_year) if start_year else 2026)
        actual_start = actual_end = yr

        row_a = rf_df[(rf_df["state_id"] == sid_a) & (rf_df["year"] == yr)]
        row_b = rf_df[(rf_df["state_id"] == sid_b) & (rf_df["year"] == yr)]

        field = metric_meta["field"]
        if not row_a.empty and pd.notna(row_a.iloc[0][field]):
            val_a = round(float(row_a.iloc[0][field]), 2)
            obs_count_a = int(row_a.iloc[0]["observed_days_count"])
        if not row_b.empty and pd.notna(row_b.iloc[0][field]):
            val_b = round(float(row_b.iloc[0][field]), 2)
            obs_count_b = int(row_b.iloc[0]["observed_days_count"])

    elif metric_meta["dataset"] == "flood":
        fl_path = PROCESSED_DATA_DIR / "flood_events.parquet"
        if not fl_path.exists():
            from climatelens.preprocessing.flood import process_flood
            process_flood()
        fl_df = pd.read_parquet(fl_path)

        mask_a = fl_df["primary_state_id"] == sid_a
        mask_b = fl_df["primary_state_id"] == sid_b
        if start_year is not None:
            mask_a &= fl_df["year"] >= start_year
            mask_b &= fl_df["year"] >= start_year
        if end_year is not None:
            mask_a &= fl_df["year"] <= end_year
            mask_b &= fl_df["year"] <= end_year

        val_a = float(mask_a.sum())
        val_b = float(mask_b.sum())
        obs_count_a = int(val_a)
        obs_count_b = int(val_b)
        actual_start = int(fl_df["year"].min()) if start_year is None else start_year
        actual_end = int(fl_df["year"].max()) if end_year is None else end_year

    elif metric_meta["dataset"] == "landslide":
        ls_path = PROCESSED_DATA_DIR / "landslide_events.parquet"
        if not ls_path.exists():
            from climatelens.preprocessing.landslide import process_landslide
            process_landslide()
        ls_df = pd.read_parquet(ls_path)

        mask_a = ls_df["state_id"] == sid_a
        mask_b = ls_df["state_id"] == sid_b
        if start_year is not None:
            mask_a &= ls_df["year"] >= start_year
            mask_b &= ls_df["year"] >= start_year
        if end_year is not None:
            mask_a &= ls_df["year"] <= end_year
            mask_b &= ls_df["year"] <= end_year

        val_a = float(mask_a.sum())
        val_b = float(mask_b.sum())
        obs_count_a = int(val_a)
        obs_count_b = int(val_b)
        actual_start = int(ls_df["year"].min()) if start_year is None else start_year
        actual_end = int(ls_df["year"].max()) if end_year is None else end_year

    elif metric_meta["dataset"] == "heatwave":
        hw_path = PROCESSED_DATA_DIR / "state_heatwave.parquet"
        if not hw_path.exists():
            from climatelens.preprocessing.heatwave import process_heatwave
            process_heatwave()
        hw_df = pd.read_parquet(hw_path)

        df_a = hw_df[hw_df["state_id"] == sid_a]
        df_b = hw_df[hw_df["state_id"] == sid_b]
        if start_year is not None:
            df_a = df_a[df_a["year"] >= start_year]
            df_b = df_b[df_b["year"] >= start_year]
        if end_year is not None:
            df_a = df_a[df_a["year"] <= end_year]
            df_b = df_b[df_b["year"] <= end_year]

        field = metric_meta["field"]
        if not df_a.empty:
            if metric_clean == "summer_max_temp":
                val_a = round(float(df_a[field].dropna().max()), 2) if df_a[field].notna().any() else None
            else:
                val_a = round(float(df_a[field].dropna().sum()), 2) if df_a[field].notna().any() else 0.0
            obs_count_a = len(df_a)
        if not df_b.empty:
            if metric_clean == "summer_max_temp":
                val_b = round(float(df_b[field].dropna().max()), 2) if df_b[field].notna().any() else None
            else:
                val_b = round(float(df_b[field].dropna().sum()), 2) if df_b[field].notna().any() else 0.0
            obs_count_b = len(df_b)

        actual_start = int(hw_df["year"].min()) if start_year is None else start_year
        actual_end = int(hw_df["year"].max()) if end_year is None else end_year

    # Check for missing data
    if val_a is None or val_b is None:
        missing_states = []
        if val_a is None:
            missing_states.append(meta_a["state_name"])
        if val_b is None:
            missing_states.append(meta_b["state_name"])
        return {
            "status": "no_data",
            "state_a": {"state_id": sid_a, "state_name": meta_a["state_name"], "value": val_a},
            "state_b": {"state_id": sid_b, "state_name": meta_b["state_name"], "value": val_b},
            "metric": metric_clean,
            "message": f"Data unavailable for {', '.join(missing_states)} for {metric_meta['label']} in requested period.",
        }

    # Protected difference and percentage change calculation
    diff = round(val_b - val_a, 2)
    pct_change = None
    note = None

    if val_a != 0.0:
        pct_change = round(((val_b - val_a) / abs(val_a)) * 100.0, 2)
    else:
        note = "Percentage difference undefined due to zero baseline value in State A."

    return {
        "status": "ok",
        "metric": metric_clean,
        "metric_label": metric_meta["label"],
        "unit": metric_meta["unit"],
        "period": {"start": str(actual_start), "end": str(actual_end)},
        "state_a": {
            "state_id": sid_a,
            "state_name": meta_a["state_name"],
            "value": val_a,
            "observation_count": obs_count_a,
        },
        "state_b": {
            "state_id": sid_b,
            "state_name": meta_b["state_name"],
            "value": val_b,
            "observation_count": obs_count_b,
        },
        "comparison": {
            "difference_b_minus_a": diff,
            "percent_difference": pct_change,
            "higher_state": meta_b["state_name"] if val_b > val_a else (meta_a["state_name"] if val_a > val_b else "Equal"),
            "note": note,
        },
    }
