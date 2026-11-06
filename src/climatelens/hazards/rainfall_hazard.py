"""Rainfall and extreme precipitation hazard analysis for ClimateLens India."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nielytics.hazards.drought import DroughtAnalyzer
from nielytics.hazards.flood import FloodAnalyzer
from climatelens.constants import (
    PROCESSED_DATA_DIR,
    RAINFALL_CATEGORIES,
    STATES_MAP,
    normalize_state_id,
)

logger = logging.getLogger(__name__)


def _load_rainfall_daily() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "rainfall_daily.parquet"
    if not path.exists():
        from climatelens.preprocessing.rainfall import process_rainfall
        daily_df, _ = process_rainfall()
        return daily_df
    return pd.read_parquet(path)


def _load_rainfall_state_year() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "rainfall_state_year.parquet"
    if not path.exists():
        from climatelens.preprocessing.rainfall import process_rainfall
        _, year_df = process_rainfall()
        return year_df
    return pd.read_parquet(path)


def get_rainfall_choropleth(year: int = 2026) -> dict[str, Any]:
    """Generate state choropleth layer with departure categories and color hex codes."""
    year_df = _load_rainfall_state_year()
    filtered = year_df[year_df["year"] == year].copy()

    records = []
    for sid, meta in sorted(STATES_MAP.items(), key=lambda x: x[1]["state_name"]):
        match = filtered[filtered["state_id"] == sid]
        if not match.empty:
            row = match.iloc[0]
            cat_code = row.get("monsoon_category")
            dep_pct = row.get("monsoon_departure_pct")
            act_mm = row.get("monsoon_cumulative_actual_mm")
            norm_mm = row.get("monsoon_cumulative_normal_mm")
        else:
            cat_code = None
            dep_pct = None
            act_mm = None
            norm_mm = None

        # Resolve category color
        color = "#94a3b8"  # slate for no data
        category_label = "No Data"
        if pd.notna(dep_pct):
            d = float(dep_pct)
            if d >= 60.0:
                category_label = "Large Excess"
                color = RAINFALL_CATEGORIES["Large Excess"]["color"]
            elif d >= 20.0:
                category_label = "Excess"
                color = RAINFALL_CATEGORIES["Excess"]["color"]
            elif d >= -19.0:
                category_label = "Normal"
                color = RAINFALL_CATEGORIES["Normal"]["color"]
            elif d >= -59.0:
                category_label = "Deficient"
                color = RAINFALL_CATEGORIES["Deficient"]["color"]
            elif d > -100.0:
                category_label = "Large Deficient"
                color = RAINFALL_CATEGORIES["Large Deficient"]["color"]
            else:
                category_label = "No Rain"
                color = RAINFALL_CATEGORIES["No Rain"]["color"]

        records.append({
            "state_id": sid,
            "state_name": meta["state_name"],
            "state_code": meta["state_code"],
            "monsoon_actual_mm": round(float(act_mm), 2) if pd.notna(act_mm) else None,
            "monsoon_normal_mm": round(float(norm_mm), 2) if pd.notna(norm_mm) else None,
            "monsoon_departure_pct": round(float(dep_pct), 2) if pd.notna(dep_pct) else None,
            "monsoon_category": category_label,
            "color": color,
        })

    return {
        "year": year,
        "states": records,
        "legend": {k: {"color": v["color"], "departure_range": f"{v['min_departure']}% to {v['max_departure']}%" if v["max_departure"] is not None else f"+{v['min_departure']}% or higher"} for k, v in RAINFALL_CATEGORIES.items()},
    }


def analyze_state_rainfall(state_id: str, year: int = 2026) -> dict[str, Any]:
    """Perform detailed rainfall hazard analysis on an Indian state using NIElytics FloodAnalyzer."""
    sid = normalize_state_id(state_id)
    if not sid or sid not in STATES_MAP:
        raise ValueError(f"Invalid state_id '{state_id}'")

    daily_df = _load_rainfall_daily()
    state_daily = daily_df[(daily_df["entity_id"] == sid) & (daily_df["date"].dt.year == year)].copy()

    if state_daily.empty:
        return {
            "status": "no_data",
            "state_id": sid,
            "state_name": STATES_MAP[sid]["state_name"],
            "year": year,
            "message": f"No rainfall observations found for {STATES_MAP[sid]['state_name']} in year {year}.",
        }

    # Format for NIElytics FloodAnalyzer
    # Required columns: date, rainfall
    fa_df = state_daily[["date", "daily_actual_mm"]].copy()
    fa_df = fa_df.rename(columns={"daily_actual_mm": "rainfall"})
    # Drop rows with NaN rainfall for complete-case analyzer execution
    clean_fa = fa_df.dropna(subset=["rainfall"]).copy()
    clean_fa["date"] = clean_fa["date"].dt.strftime("%Y-%m-%d")

    wet_spells: list[dict[str, Any]] = []
    extreme_days: list[dict[str, Any]] = []
    dry_spells: list[dict[str, Any]] = []

    if len(clean_fa) >= 2:
        try:
            analyzer = FloodAnalyzer(clean_fa, date_column="date", rainfall_column="rainfall", unit="mm")
            # Wet spell duration (>= 2.5 mm for at least 2 consecutive days)
            wet_res = analyzer.wet_spell_duration(threshold_mm=2.5, minimum_days=2)
            wet_spells = wet_res.data.get("episodes", [])
            # Extreme rainfall days (IMD heavy rainfall >= 64.5 mm)
            extreme_res = analyzer.extreme_rainfall_days(threshold_mm=64.5)
            extreme_days = extreme_res.data.get("dates", []) or extreme_res.data.get("days", [])
        except Exception as exc:
            logger.warning("FloodAnalyzer encountered exception: %s", exc)

        try:
            dr_analyzer = DroughtAnalyzer(clean_fa, date_column="date", rainfall_column="rainfall", unit="mm")
            dry_res = dr_analyzer.dry_spell_duration(threshold_mm=0.0, minimum_days=3)
            dry_spells = dry_res.data.get("episodes", [])
        except Exception as exc:
            logger.warning("DroughtAnalyzer encountered exception: %s", exc)
    acts = state_daily["daily_actual_mm"].dropna()
    total_in_period = float(acts.sum()) if not acts.empty else None
    mean_daily = float(acts.mean()) if not acts.empty else None
    max_daily = float(acts.max()) if not acts.empty else None

    # Latest cumulative observation
    latest_cum = state_daily.sort_values("date").tail(1).iloc[0]

    return {
        "status": "ok",
        "state_id": sid,
        "state_name": STATES_MAP[sid]["state_name"],
        "year": year,
        "observed_days_count": len(state_daily),
        "total_observed_rainfall_mm": round(total_in_period, 2) if total_in_period is not None else None,
        "mean_daily_rainfall_mm": round(mean_daily, 2) if mean_daily is not None else None,
        "max_daily_rainfall_mm": round(max_daily, 2) if max_daily is not None else None,
        "cumulative_monsoon_actual_mm": round(float(latest_cum["cumulative_actual_mm"]), 2) if pd.notna(latest_cum["cumulative_actual_mm"]) else None,
        "cumulative_monsoon_normal_mm": round(float(latest_cum["cumulative_normal_mm"]), 2) if pd.notna(latest_cum["cumulative_normal_mm"]) else None,
        "cumulative_monsoon_departure_pct": round(float(latest_cum["cumulative_departure_pct"]), 2) if pd.notna(latest_cum["cumulative_departure_pct"]) else None,
        "cumulative_monsoon_category": latest_cum["cumulative_category"],
        "wet_spells_count": len(wet_spells),
        "wet_spells": wet_spells[:10],
        "heavy_rain_days_count": len(extreme_days),
        "heavy_rain_days": extreme_days[:10],
        "dry_spells_count": len(dry_spells),
        "dry_spells": dry_spells[:10],
    }


def get_state_rainfall_timeseries(
    state_id: str,
    start_year: int = 2022,
    end_year: int = 2026,
) -> list[dict[str, Any]]:
    """Get time-ordered daily rainfall series for a state."""
    sid = normalize_state_id(state_id)
    if not sid or sid not in STATES_MAP:
        raise ValueError(f"Invalid state_id '{state_id}'")

    daily_df = _load_rainfall_daily()
    mask = (
        (daily_df["entity_id"] == sid) &
        (daily_df["date"].dt.year >= start_year) &
        (daily_df["date"].dt.year <= end_year)
    )
    filtered = daily_df[mask].sort_values("date")

    rows = []
    for _, r in filtered.iterrows():
        rows.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "daily_actual_mm": round(float(r["daily_actual_mm"]), 2) if pd.notna(r["daily_actual_mm"]) else None,
            "daily_normal_mm": round(float(r["daily_normal_mm"]), 2) if pd.notna(r["daily_normal_mm"]) else None,
            "daily_departure_pct": round(float(r["daily_departure_pct"]), 2) if pd.notna(r["daily_departure_pct"]) else None,
            "daily_category": r["daily_category"],
            "cumulative_actual_mm": round(float(r["cumulative_actual_mm"]), 2) if pd.notna(r["cumulative_actual_mm"]) else None,
            "cumulative_normal_mm": round(float(r["cumulative_normal_mm"]), 2) if pd.notna(r["cumulative_normal_mm"]) else None,
            "cumulative_departure_pct": round(float(r["cumulative_departure_pct"]), 2) if pd.notna(r["cumulative_departure_pct"]) else None,
        })
    return rows
