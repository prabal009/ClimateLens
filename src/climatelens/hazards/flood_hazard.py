"""Historical flood event inventory analysis for ClimateLens India."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nielytics.hazards.flood import FloodAnalyzer

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    normalize_state_id,
)

logger = logging.getLogger(__name__)

SCIENTIFIC_CAVEAT = (
    "Historical flood inventory (IFI-Impacts, IIT Delhi). Missing impact or casualty fields "
    "are retained as null; zero casualties or damage are never imputed. Coordinates are absent in source."
)


def _load_flood_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "flood_events.parquet"
    if not path.exists():
        from climatelens.preprocessing.flood import process_flood
        return process_flood()
    return pd.read_parquet(path)


def analyze_flood_hazard(
    state_id: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    min_fatalities: float | None = None,
) -> dict[str, Any]:
    """Analyze historical flood event occurrences, casualty counts, and frequency trends."""
    df = _load_flood_data()

    sid = normalize_state_id(state_id) if state_id else None
    if state_id and not sid:
        raise ValueError(f"Invalid state_id '{state_id}'")

    filtered = df.copy()
    if sid:
        filtered = filtered[filtered["primary_state_id"] == sid]

    if start_year is not None:
        filtered = filtered[filtered["year"] >= start_year]
    if end_year is not None:
        filtered = filtered[filtered["year"] <= end_year]
    if min_fatalities is not None:
        filtered = filtered[filtered["human_fatality"] >= min_fatalities]

    if filtered.empty:
        return {
            "status": "no_data",
            "state_id": sid,
            "state_name": STATES_MAP[sid]["state_name"] if sid else "All India",
            "period": {"start": start_year, "end": end_year},
            "message": "No flood events match the specified filter criteria.",
        }

    # NIElytics FloodAnalyzer event frequency
    nielytics_flood_freq = None
    try:
        valid_dates_df = filtered[filtered["start_date"].notnull()][["start_date", "event_id"]].copy()
        valid_dates_df["start_date"] = valid_dates_df["start_date"].astype(str).str[:10]
        if not valid_dates_df.empty:
            dummy_env = pd.DataFrame({"date": [valid_dates_df["start_date"].iloc[0]], "rainfall": [10.0]})
            fa = FloodAnalyzer(dummy_env, date_column="date", rainfall_column="rainfall", unit="mm")
            freq_res = fa.flood_event_frequency(valid_dates_df, event_date_column="start_date")
            nielytics_flood_freq = freq_res.data.get("years", [])
    except Exception as exc:
        logger.debug("FloodAnalyzer flood_event_frequency notice: %s", exc)

    # Annual frequency series
    annual_counts = (
        filtered.groupby("year")
        .agg(
            events_count=("event_id", "count"),
            fatalities_sum=("human_fatality", lambda s: float(s.sum()) if s.notna().any() else None),
            displaced_sum=("human_displaced", lambda s: float(s.sum()) if s.notna().any() else None),
        )
        .reset_index()
    )
    annual_counts["year"] = annual_counts["year"].astype(int)

    # Top impacted states (if viewing national)
    top_states = []
    if not sid:
        top_states_df = (
            filtered.groupby(["primary_state_id", "primary_state_name"])
            .agg(
                events_count=("event_id", "count"),
                fatalities_recorded=("human_fatality", lambda s: float(s.sum()) if s.notna().any() else None),
            )
            .reset_index()
            .sort_values("events_count", ascending=False)
            .head(10)
        )
        top_states = top_states_df.to_dict(orient="records")

    # Major flood events
    major_events = (
        filtered.sort_values("human_fatality", ascending=False)
        .head(10)[["event_id", "year", "start_date", "primary_state_name", "districts", "human_fatality", "human_displaced", "main_cause"]]
        .to_dict(orient="records")
    )

    fatalities_col = filtered["human_fatality"].dropna()
    displaced_col = filtered["human_displaced"].dropna()

    return {
        "status": "ok",
        "state_id": sid,
        "state_name": STATES_MAP[sid]["state_name"] if sid else "All India",
        "period": {
            "start": int(filtered["year"].min()) if pd.notna(filtered["year"].min()) else None,
            "end": int(filtered["year"].max()) if pd.notna(filtered["year"].max()) else None,
        },
        "total_events_recorded": len(filtered),
        "total_fatalities_reported": float(fatalities_col.sum()) if not fatalities_col.empty else None,
        "total_displaced_reported": float(displaced_col.sum()) if not displaced_col.empty else None,
        "events_with_casualty_data_count": int(filtered["human_fatality"].notna().sum()),
        "annual_frequency": annual_counts.to_dict(orient="records"),
        "nielytics_frequency": nielytics_flood_freq,
        "top_affected_states": top_states,
        "major_events": major_events,
        "caveat": SCIENTIFIC_CAVEAT,
    }
