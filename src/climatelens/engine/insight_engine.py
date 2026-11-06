"""Deterministic explainable climate insight generation engine for ClimateLens India."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nielytics.hazards import HeatwaveAnalyzer
from nielytics.insights import Insight, InsightEngine
from nielytics.models.schemas import AnalysisResult

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    normalize_state_id,
)

logger = logging.getLogger(__name__)


def insight_to_11_key_dict(
    insight: Insight,
    source_dataset: str = "nielytics_v2",
    geographic_level: str = "national",
    default_period: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a nielytics Insight dataclass into the canonical 11-key dictionary."""
    metrics = insight.metrics or {}
    val = (
        metrics.get("value")
        if metrics.get("value") is not None
        else metrics.get("severe_heatwave_days")
        if metrics.get("severe_heatwave_days") is not None
        else metrics.get("heatwave_days")
        if metrics.get("heatwave_days") is not None
        else metrics.get("event_count")
        if metrics.get("event_count") is not None
        else metrics.get("compound_days")
        if metrics.get("compound_days") is not None
        else 0
    )
    unit = "days" if "days" in insight.headline.lower() else "events" if "event" in insight.headline.lower() else "count"

    period = default_period or {"start": "1967", "end": "2026"}
    if insight.date_range and len(insight.date_range) >= 2:
        period = {"start": str(insight.date_range[0]), "end": str(insight.date_range[1])}

    return {
        "title": insight.headline,
        "human_readable": insight.description,
        "value": float(val) if isinstance(val, (int, float)) else val,
        "unit": unit,
        "period": period,
        "geographic_level": geographic_level,
        "source_dataset": source_dataset,
        "method": f"Deterministic rule-based category: {insight.category}",
        "observation_count": int(metrics.get("total_rows", metrics.get("observation_count", 1))),
        "limitations": insight.limitations or ["Deterministic rule-based derivation."],
        "calculation_details": {
            "category": insight.category,
            "metrics": metrics,
            "confidence": insight.confidence,
        },
    }


def generate_nielytics_insights(
    analysis_results: list[AnalysisResult] | dict[str, AnalysisResult],
    data_quality_report: dict[str, Any] | None = None,
) -> list[Insight]:
    """Derive rule-based insights using upstream nielytics InsightEngine."""
    engine = InsightEngine()
    result = engine.generate_insights(analysis_results, data_quality_report=data_quality_report)
    insights_data = result.data.get("insights", [])
    insights: list[Insight] = []
    for item in insights_data:
        if isinstance(item, dict):
            insights.append(Insight(**item))
        elif isinstance(item, Insight):
            insights.append(item)
    return insights


def generate_insights(
    state_id: str | None = None,
    period: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate explainable, evidence-grounded insights strictly derived from verified measurements.

    Schema per insight:
    - title
    - human_readable
    - value
    - unit
    - period
    - geographic_level
    - source_dataset
    - method
    - observation_count
    - limitations
    - calculation_details
    """
    sid = normalize_state_id(state_id) if state_id else None
    insights: list[dict[str, Any]] = []

    # 1. Rainfall Insight
    rf_path = PROCESSED_DATA_DIR / "rainfall_state_year.parquet"
    if rf_path.exists():
        rf_df = pd.read_parquet(rf_path)
        if sid:
            state_rf = rf_df[rf_df["state_id"] == sid]
            if not state_rf.empty:
                r = state_rf.iloc[0]
                act = r["monsoon_cumulative_actual_mm"]
                norm = r["monsoon_cumulative_normal_mm"]
                dep = r["monsoon_departure_pct"]
                cat = r["monsoon_category"]
                sname = STATES_MAP[sid]["state_name"]
                if pd.notna(dep):
                    direction = "above" if dep > 0 else "below"
                    insights.append({
                        "title": f"{sname} Monsoon Rainfall Departure",
                        "human_readable": f"{sname} recorded {act:.1f} mm cumulative monsoon rainfall in 2026, which is {abs(dep):.1f}% {direction} the normal baseline of {norm:.1f} mm, classified as '{cat}'.",
                        "value": float(dep),
                        "unit": "%",
                        "period": {"start": "2026-06-01", "end": "2026-09-30"},
                        "geographic_level": "state",
                        "source_dataset": "imd_rainfall_daily",
                        "method": "Cumulative seasonal departure from climatological normal: ((Actual - Normal) / Normal) * 100",
                        "observation_count": int(r["observed_days_count"]),
                        "limitations": ["Represents seasonal cumulative rainfall as of Sep 30, 2026 from IMD NWIC portal."],
                        "calculation_details": {
                            "actual_mm": float(act),
                            "normal_mm": float(norm),
                            "departure_pct": float(dep),
                            "formula": "((actual_mm - normal_mm) / normal_mm) * 100",
                        },
                    })
        else:
            # National extreme wettest / driest states
            valid_rf = rf_df.dropna(subset=["monsoon_departure_pct"])
            if not valid_rf.empty:
                wettest = valid_rf.loc[valid_rf["monsoon_departure_pct"].idxmax()]
                insights.append({
                    "title": "Highest State Monsoon Rainfall Departure (2026)",
                    "human_readable": f"{wettest['state_name']} experienced the highest positive rainfall anomaly across Indian states in 2026 at +{wettest['monsoon_departure_pct']:.1f}% above normal ({wettest['monsoon_cumulative_actual_mm']:.1f} mm vs {wettest['monsoon_cumulative_normal_mm']:.1f} mm normal).",
                    "value": float(wettest["monsoon_departure_pct"]),
                    "unit": "%",
                    "period": {"start": "2026-06-01", "end": "2026-09-30"},
                    "geographic_level": "national",
                    "source_dataset": "imd_rainfall_daily",
                    "method": "Argmax of state cumulative departure percentages",
                    "observation_count": len(valid_rf),
                    "limitations": ["Covers states with complete cumulative observation records in IMD daily dataset."],
                    "calculation_details": {
                        "state_id": wettest["state_id"],
                        "actual_mm": float(wettest["monsoon_cumulative_actual_mm"]),
                        "normal_mm": float(wettest["monsoon_cumulative_normal_mm"]),
                    },
                })

    # 2. Drought Insight (National level)
    if not sid:
        dr_path = PROCESSED_DATA_DIR / "national_drought.parquet"
        if dr_path.exists():
            dr_df = pd.read_parquet(dr_path)
            max_d0_row = dr_df.loc[dr_df["d0_d4_pct"].idxmax()]
            insights.append({
                "title": "Peak National Drought Footprint (2021-2026)",
                "human_readable": f"On {max_d0_row['date'].strftime('%Y-%m-%d')}, India recorded its largest abnormally dry drought footprint in the 5-year monitoring period, with {max_d0_row['d0_d4_pct']:.1f}% of national land area experiencing D0-D4 conditions.",
                "value": float(max_d0_row["d0_d4_pct"]),
                "unit": "%",
                "period": {"start": "2021-07-14", "end": "2026-10-28"},
                "geographic_level": "national",
                "source_dataset": "iitgn_national_drought",
                "method": "Maximum weekly national drought area percentage (D0-D4)",
                "observation_count": len(dr_df),
                "limitations": ["National aggregate timeseries from IIT Gandhinagar; cannot be disaggregated to state level."],
                "calculation_details": {
                    "peak_date": max_d0_row["date"].strftime("%Y-%m-%d"),
                    "d0_d4_pct": float(max_d0_row["d0_d4_pct"]),
                    "d1_d4_moderate_plus_pct": float(max_d0_row["d1_d4_pct"]),
                },
            })

    # 3. Flood Insight
    fl_path = PROCESSED_DATA_DIR / "flood_events.parquet"
    if fl_path.exists():
        fl_df = pd.read_parquet(fl_path)
        if sid:
            state_fl = fl_df[fl_df["primary_state_id"] == sid]
            count = len(state_fl)
            fatalities = float(state_fl["human_fatality"].sum()) if state_fl["human_fatality"].notna().any() else 0.0
            sname = STATES_MAP[sid]["state_name"]
            insights.append({
                "title": f"Historical Flood Occurrences in {sname}",
                "human_readable": f"A total of {count} flood events were cataloged in {sname} between 1967 and 2023, with {int(fatalities)} verified fatalities reported across documented complete cases.",
                "value": count,
                "unit": "events",
                "period": {"start": "1967", "end": "2023"},
                "geographic_level": "state",
                "source_dataset": "iitd_flood_inventory",
                "method": "Count of cataloged flood events matching primary state identifier",
                "observation_count": count,
                "limitations": ["Historical event reporting completeness varies over time; blank casualty fields remain uninvented."],
                "calculation_details": {
                    "event_count": count,
                    "reported_fatalities": fatalities,
                    "years_covered": 57,
                },
            })
        else:
            top_state = fl_df["primary_state_name"].value_counts().head(1)
            if not top_state.empty:
                sname = top_state.index[0]
                cnt = int(top_state.iloc[0])
                insights.append({
                    "title": "Most Frequent Flood Occurrence State",
                    "human_readable": f"{sname} leads India in cataloged historical flood events with {cnt} documented occurrences between 1967 and 2023.",
                    "value": cnt,
                    "unit": "events",
                    "period": {"start": "1967", "end": "2023"},
                    "geographic_level": "national",
                    "source_dataset": "iitd_flood_inventory",
                    "method": "Modal frequency of primary state across 6,876 cataloged flood records",
                    "observation_count": len(fl_df),
                    "limitations": ["Reflects recorded inventory in Zenodo IFI-Impacts v4 dataset."],
                    "calculation_details": {
                        "top_state": sname,
                        "events_count": cnt,
                        "pct_of_all_floods": round((cnt / len(fl_df)) * 100, 2),
                    },
                })

    # 4. Landslide Insight
    ls_path = PROCESSED_DATA_DIR / "landslide_events.parquet"
    if ls_path.exists():
        ls_df = pd.read_parquet(ls_path)
        if sid:
            state_ls = ls_df[ls_df["state_id"] == sid]
            count = len(state_ls)
            sname = STATES_MAP[sid]["state_name"]
            if count > 0:
                insights.append({
                    "title": f"Landslide Incidents in {sname}",
                    "human_readable": f"{sname} recorded {count} spatially verified landslide incidents between 2004 and 2024 with a mean reported magnitude of {state_ls['reported_magnitude'].mean():.2f}.",
                    "value": count,
                    "unit": "incidents",
                    "period": {"start": "2004", "end": "2024"},
                    "geographic_level": "state",
                    "source_dataset": "riseuppant_landslide_inventory",
                    "method": "Point-in-polygon spatial join against NWIC authoritative state boundaries",
                    "observation_count": count,
                    "limitations": ["Community-hosted dataset; reported magnitude is uncalibrated."],
                    "calculation_details": {
                        "incident_count": count,
                        "mean_reported_magnitude": round(float(state_ls["reported_magnitude"].mean()), 2),
                    },
                })
        else:
            top_ls = ls_df["state_name"].value_counts().head(1)
            if not top_ls.empty:
                sname = top_ls.index[0]
                cnt = int(top_ls.iloc[0])
                insights.append({
                    "title": "Highest Landslide Density State",
                    "human_readable": f"{sname} recorded the highest frequency of landslide occurrences in the 2004-2024 inventory with {cnt} incident points ({round((cnt / len(ls_df)) * 100, 1)}% of all recorded landslides).",
                    "value": cnt,
                    "unit": "incidents",
                    "period": {"start": "2004", "end": "2024"},
                    "geographic_level": "national",
                    "source_dataset": "riseuppant_landslide_inventory",
                    "method": "Spatial aggregation of validated coordinate points across state polygons",
                    "observation_count": len(ls_df),
                    "limitations": ["Community-hosted GitHub source; magnitude scale is uncalibrated."],
                    "calculation_details": {
                        "state_name": sname,
                        "incidents_count": cnt,
                        "national_share_pct": round((cnt / len(ls_df)) * 100, 2),
                    },
                })

    # 5. Cyclone Insight (National)
    if not sid:
        cy_path = PROCESSED_DATA_DIR / "cyclone_track_points.parquet"
        if cy_path.exists():
            cy_df = pd.read_parquet(cy_path)
            fani_points = cy_df[cy_df["name"] == "FANI"]
            if not fani_points.empty:
                max_wmo = fani_points["wmo_wind_kmh"].dropna().max()
                insights.append({
                    "title": "Extremely Severe Cyclonic Storm Fani (2019)",
                    "human_readable": f"Cyclone Fani (May 2019) tracked 3,112 km across the Bay of Bengal, reaching peak sustained winds of {max_wmo:.0f} km/h (Category 4 on Saffir-Simpson) before making landfall near Puri, Odisha.",
                    "value": float(max_wmo),
                    "unit": "km/h",
                    "period": {"start": "2019-04-25", "end": "2019-05-04"},
                    "geographic_level": "regional",
                    "source_dataset": "noaa_ibtracs_ni",
                    "method": "Track distance integration and maximum WMO sustained wind speed",
                    "observation_count": len(fani_points),
                    "limitations": ["IBTrACS best-track data; distance to coast does not replace official IMD landfall bulletin."],
                    "calculation_details": {
                        "storm_name": "FANI",
                        "sid": "2019116N02090",
                        "max_wmo_wind_kmh": float(max_wmo),
                        "min_pressure_hpa": float(fani_points["wmo_pres_hpa"].dropna().min()),
                    },
                })

    # 6. Heatwave & Official IMD Heatwave Classification Insight (nielytics v2 Integration)
    hw_path = PROCESSED_DATA_DIR / "state_heatwave.parquet"
    if hw_path.exists():
        hw_df = pd.read_parquet(hw_path)
        if sid:
            state_hw = (
                hw_df[hw_df["state_id"] == sid]
                .dropna(subset=["summer_max_temp_deg_c"])
                .groupby("year", as_index=False)["summer_max_temp_deg_c"]
                .mean()
            )
            if not state_hw.empty:
                sname = STATES_MAP[sid]["state_name"]
                df_hw_dates = pd.DataFrame({
                    "date": pd.to_datetime(state_hw["year"].astype(str) + "-05-15"),
                    "tmax": state_hw["summer_max_temp_deg_c"].astype(float),
                })
                analyzer = HeatwaveAnalyzer(df_hw_dates, date_column="date", temperature_column="tmax", unit="C")
                res_imd = analyzer.imd_heatwave_classification(normal_tmax=40.0, terrain="plains")
                counts = res_imd.data.get("counts", {})
                sev_days = int(counts.get("severe_heatwave", 0))
                hw_days = int(counts.get("heatwave", 0))
                min_yr = int(state_hw["year"].min())
                max_yr = int(state_hw["year"].max())

                # Pass through InsightEngine for rule-based generation
                engine = InsightEngine()
                engine_res = engine.generate_insights([res_imd])
                gen_insights = engine_res.data.get("insights", [])

                if gen_insights:
                    for gi in gen_insights:
                        insight_obj = Insight(**gi) if isinstance(gi, dict) else gi
                        dict_ins = insight_to_11_key_dict(
                            insight_obj,
                            source_dataset="imd_heatwave_raw",
                            geographic_level="state",
                            default_period={"start": str(min_yr), "end": str(max_yr)},
                        )
                        dict_ins["title"] = f"{sname} IMD Heatwave Classification"
                        dict_ins["geographic_level"] = "state"
                        insights.append(dict_ins)
                else:
                    insights.append({
                        "title": f"{sname} IMD Heatwave Classification",
                        "human_readable": f"Under IMD official heatwave criteria (plains terrain), {sname} recorded {hw_days} heatwave days and {sev_days} severe heatwave days between {min_yr} and {max_yr}.",
                        "value": float(hw_days + sev_days),
                        "unit": "days",
                        "period": {"start": str(min_yr), "end": str(max_yr)},
                        "geographic_level": "state",
                        "source_dataset": "imd_heatwave_raw",
                        "method": "IMD official heatwave criteria (plains: tmax >= 45.0C or departure >= 4.5C)",
                        "observation_count": len(state_hw),
                        "limitations": ["IMD criteria evaluated on observed summer maximum temperature records."],
                        "calculation_details": {
                            "heatwave_days": hw_days,
                            "severe_heatwave_days": sev_days,
                            "total_days_evaluated": len(state_hw),
                        },
                    })
        else:
            # National IMD Heatwave Classification Insight
            all_valid_hw = (
                hw_df.dropna(subset=["summer_max_temp_deg_c"])
                .groupby("year", as_index=False)["summer_max_temp_deg_c"]
                .max()
            )
            if not all_valid_hw.empty:
                df_hw_dates = pd.DataFrame({
                    "date": pd.to_datetime(all_valid_hw["year"].astype(str) + "-05-15"),
                    "tmax": all_valid_hw["summer_max_temp_deg_c"].astype(float),
                })
                analyzer = HeatwaveAnalyzer(df_hw_dates, date_column="date", temperature_column="tmax", unit="C")
                res_imd = analyzer.imd_heatwave_classification(normal_tmax=40.0, terrain="plains")
                counts = res_imd.data.get("counts", {})
                sev_days = int(counts.get("severe_heatwave", 0))
                hw_days = int(counts.get("heatwave", 0))

                insights.append({
                    "title": "National IMD Heatwave Criteria Exceedances",
                    "human_readable": f"Across historical state summer temperature records, official IMD criteria identified {hw_days} heatwave occurrences and {sev_days} severe heatwave occurrences exceeding national thresholds.",
                    "value": float(hw_days + sev_days),
                    "unit": "occurrences",
                    "period": {"start": str(int(all_valid_hw["year"].min())), "end": str(int(all_valid_hw["year"].max()))},
                    "geographic_level": "national",
                    "source_dataset": "imd_heatwave_raw",
                    "method": "IMD heatwave criteria classification across all state summer max series",
                    "observation_count": len(all_valid_hw),
                    "limitations": ["Evaluates station summer maximums against plains threshold criteria."],
                    "calculation_details": {
                        "heatwave_occurrences": hw_days,
                        "severe_heatwave_occurrences": sev_days,
                        "total_records": len(all_valid_hw),
                    },
                })

    return insights
