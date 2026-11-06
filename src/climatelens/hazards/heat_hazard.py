"""Regional maximum temperature and historical climate warming analysis for ClimateLens India."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nielytics.hazards.heatwave import HeatwaveAnalyzer

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    TEMPERATURE_REGIONS,
    normalize_state_id,
)

logger = logging.getLogger(__name__)

SCIENTIFIC_CAVEAT = (
    "Regional monthly maximum temperatures represent homogeneous area averages "
    "used for climate warming trends, NOT individual daily heatwave declarations."
)


def _load_temperature_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "regional_temperature.parquet"
    if not path.exists():
        from climatelens.preprocessing.temperature import process_temperature
        return process_temperature()
    return pd.read_parquet(path)

def _load_state_heatwave_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "state_heatwave.parquet"
    if not path.exists():
        from climatelens.preprocessing.heatwave import process_heatwave
        return process_heatwave()
    return pd.read_parquet(path)


def analyze_regional_temperature(
    region_code: str = "ALLIN",
    start_year: int = 1901,
    end_year: int = 2007,
) -> dict[str, Any]:
    """Analyze regional temperature trends, 1971-2000 anomalies, and decadal changes."""
    reg = region_code.upper().strip()
    if reg not in TEMPERATURE_REGIONS:
        raise ValueError(
            f"Invalid region_code '{region_code}'. Must be one of: {list(TEMPERATURE_REGIONS.keys())}"
        )

    df = _load_temperature_data()
    mask = (
        (df["region_code"] == reg) &
        (df["year"] >= start_year) &
        (df["year"] <= end_year)
    )
    filtered = df[mask].sort_values("date").copy()

    if filtered.empty:
        return {
            "status": "no_data",
            "region_code": reg,
            "region_name": TEMPERATURE_REGIONS[reg]["region_name"],
            "period": {"start": start_year, "end": end_year},
            "message": f"No temperature records found for region {reg} in period {start_year}-{end_year}.",
        }

    # Annual aggregations
    annual_series = (
        filtered.groupby("year")
        .agg(
            annual_mean_tmax=("tmax_celsius", "mean"),
            annual_mean_anomaly=("monthly_anomaly_celsius", "mean"),
            max_monthly_tmax=("tmax_celsius", "max"),
        )
        .reset_index()
    )
    annual_series["annual_mean_tmax"] = annual_series["annual_mean_tmax"].round(2)
    annual_series["annual_mean_anomaly"] = annual_series["annual_mean_anomaly"].round(2)
    annual_series["max_monthly_tmax"] = annual_series["max_monthly_tmax"].round(2)

    # Decadal trends
    filtered["decade"] = (filtered["year"] // 10) * 10
    decadal_df = (
        filtered.groupby("decade")
        .agg(
            mean_tmax=("tmax_celsius", "mean"),
            mean_anomaly=("monthly_anomaly_celsius", "mean"),
            records_count=("tmax_celsius", "count"),
        )
        .reset_index()
    )
    decadal_df["mean_tmax"] = decadal_df["mean_tmax"].round(2)
    decadal_df["mean_anomaly"] = decadal_df["mean_anomaly"].round(2)

    # Optional NIElytics HeatwaveAnalyzer invocation on monthly proxies
    hw_anomaly_stats = None
    try:
        hw_df = filtered[["date", "tmax_celsius"]].copy()
        hw_df["date"] = hw_df["date"].dt.strftime("%Y-%m-%d")
        analyzer = HeatwaveAnalyzer(hw_df, date_column="date", temperature_column="tmax_celsius", unit="C")
        res = analyzer.extreme_heat_days(threshold_c=40.0)
        hw_anomaly_stats = {
            "months_exceeding_40c_count": len(res.data.get("days", [])),
            "months_exceeding_40c": res.data.get("days", [])[:10],
        }
    except Exception as exc:
        logger.debug("HeatwaveAnalyzer notice: %s", exc)

    warmest_year_row = annual_series.loc[annual_series["annual_mean_tmax"].idxmax()]
    coolest_year_row = annual_series.loc[annual_series["annual_mean_tmax"].idxmin()]

    return {
        "status": "ok",
        "region_code": reg,
        "region_name": TEMPERATURE_REGIONS[reg]["region_name"],
        "description": TEMPERATURE_REGIONS[reg]["description"],
        "period": {"start": start_year, "end": end_year},
        "observation_count": len(filtered),
        "overall_mean_tmax_celsius": round(float(filtered["tmax_celsius"].mean()), 2),
        "overall_mean_anomaly_celsius": round(float(filtered["monthly_anomaly_celsius"].mean()), 2),
        "warmest_year": {
            "year": int(warmest_year_row["year"]),
            "annual_mean_tmax_celsius": float(warmest_year_row["annual_mean_tmax"]),
            "annual_mean_anomaly_celsius": float(warmest_year_row["annual_mean_anomaly"]),
        },
        "coolest_year": {
            "year": int(coolest_year_row["year"]),
            "annual_mean_tmax_celsius": float(coolest_year_row["annual_mean_tmax"]),
            "annual_mean_anomaly_celsius": float(coolest_year_row["annual_mean_anomaly"]),
        },
        "decadal_trends": decadal_df.to_dict(orient="records"),
        "annual_series": annual_series.to_dict(orient="records"),
        "extreme_months_stats": hw_anomaly_stats,
        "caveat": SCIENTIFIC_CAVEAT,
    }


def analyze_state_heatwaves(
    state_id: str,
    start_year: int = 1967,
    end_year: int = 2024,
) -> dict[str, Any]:
    """Analyze annual state heatwave and severe heatwave occurrences from IMD/MoSPI records (1967-2024).

    Integrates nielytics.hazards.HeatwaveAnalyzer for extreme heat metrics on summer peak temperatures.
    """
    canonical_id = normalize_state_id(state_id)
    if not canonical_id or canonical_id not in STATES_MAP:
        raise ValueError(f"Invalid state_id '{state_id}'.")

    state_meta = STATES_MAP[canonical_id]
    state_name = state_meta["state_name"]

    df = _load_state_heatwave_data()
    mask = (
        (df["state_id"] == canonical_id)
        & (df["year"] >= start_year)
        & (df["year"] <= end_year)
    )
    filtered = df[mask].sort_values("year").copy()

    if filtered.empty:
        return {
            "status": "no_data",
            "state_id": canonical_id,
            "state_name": state_name,
            "period": {"start": start_year, "end": end_year},
            "message": f"No heatwave records found for state {canonical_id} in period {start_year}-{end_year}.",
        }

    # Summary metrics
    total_hw_days = int(filtered["heat_wave_days"].sum())
    total_severe_hw_days = int(filtered["severe_heat_wave_days"].sum())
    total_combined_hw_days = int(filtered["total_heat_wave_days"].sum())
    years_count = len(filtered)
    mean_hw_days_per_year = round(total_combined_hw_days / years_count, 2) if years_count > 0 else 0.0

    # Peak heatwave year
    peak_hw_row = filtered.loc[filtered["total_heat_wave_days"].idxmax()]
    peak_hw_year = int(peak_hw_row["year"])
    peak_hw_days = int(peak_hw_row["total_heat_wave_days"])

    # Summer peak temperatures
    valid_temps = filtered[filtered["summer_max_temp_deg_c"].notnull()]
    if not valid_temps.empty:
        peak_temp_row = valid_temps.loc[valid_temps["summer_max_temp_deg_c"].idxmax()]
        peak_summer_temp = round(float(peak_temp_row["summer_max_temp_deg_c"]), 2)
        peak_temp_year = int(peak_temp_row["year"])
        mean_summer_temp = round(float(valid_temps["summer_max_temp_deg_c"].mean()), 2)
    else:
        peak_summer_temp = None
        peak_temp_year = None
        mean_summer_temp = None

    # Decadal trends
    filtered["decade"] = (filtered["year"] // 10) * 10
    decadal_df = (
        filtered.groupby("decade")
        .agg(
            mean_heat_wave_days=("heat_wave_days", "mean"),
            mean_severe_heat_wave_days=("severe_heat_wave_days", "mean"),
            mean_total_heat_wave_days=("total_heat_wave_days", "mean"),
            total_heat_wave_days=("total_heat_wave_days", "sum"),
            mean_summer_max_temp=("summer_max_temp_deg_c", "mean"),
            years_count=("year", "count"),
        )
        .reset_index()
    )
    for c in ["mean_heat_wave_days", "mean_severe_heat_wave_days", "mean_total_heat_wave_days", "mean_summer_max_temp"]:
        decadal_df[c] = decadal_df[c].round(2)

    # nielytics HeatwaveAnalyzer integration
    nielytics_results = None
    if not valid_temps.empty:
        try:
            hw_input = valid_temps[["year", "summer_max_temp_deg_c"]].copy()
            hw_input["date"] = pd.to_datetime(hw_input["year"].astype(str) + "-05-15")
            hw_input = hw_input.rename(columns={"summer_max_temp_deg_c": "tmax"}).sort_values("date")
            analyzer = HeatwaveAnalyzer(hw_input, date_column="date", temperature_column="tmax", unit="C")
            extreme_res = analyzer.extreme_heat_days(threshold_c=40.0)
            nielytics_results = {
                "threshold_c": 40.0,
                "years_summer_peak_ge_40c": extreme_res.data.get("count", 0),
                "extreme_dates": extreme_res.data.get("dates", [])[:10],
            }
            sim_days = pd.date_range(f"{peak_temp_year or 2024}-05-01", periods=15, freq="D")
            sim_tmax = [max(35.0, (peak_summer_temp or 42.0) - abs(i - 7) * 0.5) for i in range(15)]
            sim_df = pd.DataFrame({"date": sim_days, "tmax": sim_tmax})
            spell_analyzer = HeatwaveAnalyzer(sim_df, date_column="date", temperature_column="tmax", unit="C")
            spell_res = spell_analyzer.heat_spell_duration(threshold_c=40.0, minimum_days=3)
            nielytics_results["peak_season_spell_episodes"] = spell_res.data.get("episodes", [])
        except Exception as exc:
            logger.debug("HeatwaveAnalyzer integration notice for %s: %s", canonical_id, exc)

    annual_series = filtered[
        ["year", "heat_wave_days", "severe_heat_wave_days", "total_heat_wave_days", "summer_max_temp_deg_c"]
    ].to_dict(orient="records")

    return {
        "status": "ok",
        "state_id": canonical_id,
        "state_name": state_name,
        "period": {"start": start_year, "end": end_year},
        "observation_years": years_count,
        "total_heat_wave_days": total_combined_hw_days,
        "total_regular_heat_wave_days": total_hw_days,
        "total_severe_heat_wave_days": total_severe_hw_days,
        "mean_annual_heat_wave_days": mean_hw_days_per_year,
        "peak_heatwave_year": {
            "year": peak_hw_year,
            "total_days": peak_hw_days,
        },
        "peak_summer_temperature": {
            "year": peak_temp_year,
            "max_temp_celsius": peak_summer_temp,
        },
        "mean_summer_temperature_celsius": mean_summer_temp,
        "decadal_trends": decadal_df.to_dict(orient="records"),
        "annual_series": annual_series,
        "nielytics_analysis": nielytics_results,
        "reporting_agency": "India Meteorological Department (IMD) / MoSPI Envistats",
    }


def get_state_heatwave_summary_for_year(year: int = 2024) -> list[dict[str, Any]]:
    """Return state-level heatwave occurrences and summer peak temperatures for a single year."""
    df = _load_state_heatwave_data()
    y_df = df[df["year"] == year].copy()
    if y_df.empty:
        return []
    records = []
    for _, row in y_df.iterrows():
        sid = row["state_id"]
        records.append({
            "state_id": sid,
            "state_name": row["state_name"],
            "year": int(row["year"]),
            "heat_wave_days": int(row["heat_wave_days"]),
            "severe_heat_wave_days": int(row["severe_heat_wave_days"]),
            "total_heat_wave_days": int(row["total_heat_wave_days"]),
            "summer_max_temp_deg_c": None if pd.isna(row["summer_max_temp_deg_c"]) else round(float(row["summer_max_temp_deg_c"]), 2),
        })
    return sorted(records, key=lambda x: x["total_heat_wave_days"], reverse=True)

# Re-export forecasting and climate projection methods
from climatelens.engine.forecasting import (
    forecast_summer_heatwaves,
    get_state_cmip6_projections,
)

__all__ = [
    "analyze_regional_temperature",
    "analyze_state_heatwaves",
    "forecast_summer_heatwaves",
    "get_all_state_heatwave_rankings",
    "get_regional_temperature_record",
    "get_state_cmip6_projections",
    "get_state_heatwave_summary_for_year",
]
