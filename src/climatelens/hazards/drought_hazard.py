"""National drought timeseries and severity transition analysis for ClimateLens India."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from climatelens.constants import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

SCIENTIFIC_CAVEAT = (
    "National-level drought area percentage from IIT Gandhinagar India Drought Monitor. "
    "Does NOT provide state-level or district-level drought observations."
)


def _load_drought_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "national_drought.parquet"
    if not path.exists():
        from climatelens.preprocessing.drought import process_drought
        return process_drought()
    return pd.read_parquet(path)


def analyze_national_drought(
    start_year: int = 2021,
    end_year: int = 2026,
) -> dict[str, Any]:
    """Analyze national drought area dynamics, peak severity weeks, and transitions."""
    df = _load_drought_data()
    mask = (df["year"] >= start_year) & (df["year"] <= end_year)
    filtered = df[mask].sort_values("date").copy()

    if filtered.empty:
        return {
            "status": "no_data",
            "period": {"start": start_year, "end": end_year},
            "message": f"No drought records found for period {start_year}-{end_year}.",
        }

    # Peak drought extent observations
    peak_d0_row = filtered.loc[filtered["d0_d4_pct"].idxmax()]
    peak_d1_row = filtered.loc[filtered["d1_d4_pct"].idxmax()]
    peak_d2_row = filtered.loc[filtered["d2_d4_pct"].idxmax()]
    peak_d3_row = filtered.loc[filtered["d3_d4_pct"].idxmax()]
    peak_d4_row = filtered.loc[filtered["d4_pct"].idxmax()]

    # Annual summary
    annual_summary = (
        filtered.groupby("year")
        .agg(
            mean_d0_d4_pct=("d0_d4_pct", "mean"),
            max_d0_d4_pct=("d0_d4_pct", "max"),
            mean_d1_d4_pct=("d1_d4_pct", "mean"),
            max_d1_d4_pct=("d1_d4_pct", "max"),
            observation_weeks=("d0_d4_pct", "count"),
        )
        .reset_index()
    )
    annual_summary["mean_d0_d4_pct"] = annual_summary["mean_d0_d4_pct"].round(2)
    annual_summary["max_d0_d4_pct"] = annual_summary["max_d0_d4_pct"].round(2)
    annual_summary["mean_d1_d4_pct"] = annual_summary["mean_d1_d4_pct"].round(2)
    annual_summary["max_d1_d4_pct"] = annual_summary["max_d1_d4_pct"].round(2)

    # Sharpest week-over-week expansions
    sharpest_expansions = (
        filtered.sort_values("weekly_change_d0_d4_pct", ascending=False)
        .head(5)[["date", "d0_d4_pct", "weekly_change_d0_d4_pct"]]
        .to_dict(orient="records")
    )
    for row in sharpest_expansions:
        row["date"] = row["date"].strftime("%Y-%m-%d")

    timeseries = []
    for _, r in filtered.iterrows():
        timeseries.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "no_drought_pct": float(r["no_drought_pct"]),
            "d0_d4_pct": float(r["d0_d4_pct"]),
            "d1_d4_pct": float(r["d1_d4_pct"]),
            "d2_d4_pct": float(r["d2_d4_pct"]),
            "d3_d4_pct": float(r["d3_d4_pct"]),
            "d4_pct": float(r["d4_pct"]),
            "rolling_4w_d0_d4_pct": float(r["rolling_4w_d0_d4_pct"]) if pd.notna(r["rolling_4w_d0_d4_pct"]) else None,
            "weekly_change_d0_d4_pct": float(r["weekly_change_d0_d4_pct"]) if pd.notna(r["weekly_change_d0_d4_pct"]) else None,
        })

    return {
        "status": "ok",
        "geographic_level": "national",
        "period": {"start": start_year, "end": end_year},
        "observation_weeks_count": len(filtered),
        "mean_d0_d4_pct": round(float(filtered["d0_d4_pct"].mean()), 2),
        "mean_d1_d4_pct": round(float(filtered["d1_d4_pct"].mean()), 2),
        "peak_abnormally_dry": {
            "date": peak_d0_row["date"].strftime("%Y-%m-%d"),
            "d0_d4_pct": float(peak_d0_row["d0_d4_pct"]),
        },
        "peak_moderate_drought": {
            "date": peak_d1_row["date"].strftime("%Y-%m-%d"),
            "d1_d4_pct": float(peak_d1_row["d1_d4_pct"]),
        },
        "peak_exceptional_drought": {
            "date": peak_d4_row["date"].strftime("%Y-%m-%d"),
            "d4_pct": float(peak_d4_row["d4_pct"]),
        },
        "annual_summary": annual_summary.to_dict(orient="records"),
        "sharpest_weekly_expansions": sharpest_expansions,
        "timeseries": timeseries,
        "caveat": SCIENTIFIC_CAVEAT,
    }
