"""Rainfall data preprocessing and aggregation for ClimateLens India."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nielytics import Dataset

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    normalize_state_id,
    resolve_raw_file,
)
from climatelens.preprocessing.loader import get_raw_sheet_df
logger = logging.getLogger(__name__)


def clean_column_name(col: str) -> str:
    """Normalize raw column name by removing newlines and typos."""
    c = col.replace("\n", " ").replace("\r", " ").strip()
    c = c.replace("Departue", "Departure").replace("Acutual", "Actual")
    while "  " in c:
        c = c.replace("  ", " ")
    return c
def _parse_float_or_pct(val: Any) -> float:
    if pd.isna(val) or val is None or val == "":
        return np.nan
    try:
        val_str = str(val).replace("%", "").strip()
        if not val_str:
            return np.nan
        f = float(val_str)
        return f if np.isfinite(f) else np.nan
    except (ValueError, TypeError):
        return np.nan



def process_rainfall(output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process IMD state-wise daily rainfall dataset.

    Steps:
    1. Parse CSV and standardize column names.
    2. Categorize rows by geographic level (state, regional, national).
    3. Normalize state names to canonical state_id.
    4. Preserve missing actuals as np.nan (NEVER impute zero for missing data).
    5. Generate daily analytical records and state-level annual/seasonal summaries.
    6. Export rainfall_daily.parquet and rainfall_state_year.parquet.
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_df = get_raw_sheet_df("Rainfall_Raw")
        logger.info("Loaded %d rows from Rainfall_Raw", len(raw_df))
    except Exception as exc:
        csv_path = resolve_raw_file("rainfall")
        logger.info("Falling back to rainfall CSV from %s (%s)", csv_path, exc)
        raw_df = pd.read_csv(csv_path)

    ds = Dataset(raw_df)
    raw_df = ds.frame
    raw_df.columns = [clean_column_name(c) for c in raw_df.columns]
    records: list[dict[str, Any]] = []

    for _, row in raw_df.iterrows():
        raw_state = str(row["State"]).strip()
        date_str = str(row["Date"]).strip()

        # Determine geographic level and canonical entity ID
        if raw_state.startswith("COUNTRY :"):
            geo_level = "national"
            entity_id = "INDIA_NATIONAL"
            entity_name = "India (National)"
        elif raw_state.startswith("REGION :"):
            geo_level = "regional"
            region_name = raw_state.replace("REGION :", "").strip()
            entity_id = f"REGION_{region_name.replace(' ', '_').upper()}"
            entity_name = region_name.title()
        else:
            geo_level = "state"
            sid = normalize_state_id(raw_state)
            if not sid:
                logger.warning("Unrecognized state name in rainfall: '%s'", raw_state)
                continue
            entity_id = sid
            entity_name = STATES_MAP[sid]["state_name"]

        daily_act = _parse_float_or_pct(row.get("Daily Actual"))
        daily_norm = _parse_float_or_pct(row.get("Daily Normal"))
        daily_dep = _parse_float_or_pct(row.get("Daily Departure Per"))
        daily_cat = row.get("Daily Category")
        if pd.isna(daily_cat) or str(daily_cat).strip() in ("", "nan", "None"):
            daily_cat = None
        else:
            daily_cat = str(daily_cat).strip()

        records.append({
            "date": date_str,
            "entity_id": entity_id,
            "state_id": entity_id if geo_level == "state" else None,
            "entity_name": entity_name,
            "geographic_level": geo_level,
            "daily_actual_mm": daily_act,
            "daily_normal_mm": daily_norm,
            "daily_departure_pct": daily_dep,
            "daily_category": daily_cat,
            "weekly_actual_mm": _parse_float_or_pct(row.get("Weekly Actual")),
            "weekly_normal_mm": _parse_float_or_pct(row.get("Weekly Normal")),
            "weekly_departure_pct": _parse_float_or_pct(row.get("Weekly Departure Per")),
            "weekly_category": str(row.get("Weekly Category")).strip() if pd.notna(row.get("Weekly Category")) else None,
            "cumulative_actual_mm": _parse_float_or_pct(row.get("Cumulative Actual")),
            "cumulative_normal_mm": _parse_float_or_pct(row.get("Cumulative Normal")),
            "cumulative_departure_pct": _parse_float_or_pct(row.get("Cumulative Departure Per")),
            "cumulative_category": str(row.get("Cumulative Category")).strip() if pd.notna(row.get("Cumulative Category")) else None,
            "monthly_actual_mm": _parse_float_or_pct(row.get("Monthly Actual")),
            "monthly_normal_mm": _parse_float_or_pct(row.get("Monthly Normal")),
            "monthly_departure_pct": _parse_float_or_pct(row.get("Monthly Departure Per")),
            "monthly_category": str(row.get("Monthly Category")).strip() if pd.notna(row.get("Monthly Category")) else None,
        })

    daily_df = pd.DataFrame(records)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values(["entity_id", "date"]).reset_index(drop=True)

    # Export daily parquet
    daily_parquet_path = out_dir / "rainfall_daily.parquet"
    daily_df.to_parquet(daily_parquet_path, index=False)
    logger.info("Saved rainfall daily parquet to %s (%d rows)", daily_parquet_path, len(daily_df))

    # Build state-year and seasonal aggregations
    state_daily = daily_df[daily_df["geographic_level"] == "state"].copy()
    state_daily["year"] = state_daily["date"].dt.year

    summary_rows = []
    for (sid, yr), group in state_daily.groupby(["entity_id", "year"]):
        obs_count = len(group)
        acts = group["daily_actual_mm"].dropna()
        norms = group["daily_normal_mm"].dropna()

        # Cumulative monsoon value (as of Sep 30 or closest date)
        sep_rows = group[group["date"] == f"{yr}-09-30"]
        if sep_rows.empty:
            sep_rows = group[group["date"].dt.month == 9].sort_values("date").tail(1)

        monsoon_act = sep_rows.iloc[0]["cumulative_actual_mm"] if not sep_rows.empty else np.nan
        monsoon_norm = sep_rows.iloc[0]["cumulative_normal_mm"] if not sep_rows.empty else np.nan
        monsoon_dep = sep_rows.iloc[0]["cumulative_departure_pct"] if not sep_rows.empty else np.nan
        monsoon_cat = sep_rows.iloc[0]["cumulative_category"] if not sep_rows.empty else None

        summary_rows.append({
            "state_id": sid,
            "state_name": STATES_MAP[sid]["state_name"],
            "year": int(yr),
            "observed_days_count": obs_count,
            "daily_actual_mean_mm": round(float(acts.mean()), 2) if not acts.empty else np.nan,
            "daily_actual_sum_in_period_mm": round(float(acts.sum()), 2) if not acts.empty else np.nan,
            "daily_actual_max_mm": round(float(acts.max()), 2) if not acts.empty else np.nan,
            "heavy_rain_days_count": int((acts >= 64.5).sum()),  # IMD heavy rain >= 64.5mm
            "wet_days_count": int((acts >= 2.5).sum()),          # IMD rainy day >= 2.5mm
            "monsoon_cumulative_actual_mm": round(float(monsoon_act), 2) if pd.notna(monsoon_act) else np.nan,
            "monsoon_cumulative_normal_mm": round(float(monsoon_norm), 2) if pd.notna(monsoon_norm) else np.nan,
            "monsoon_departure_pct": round(float(monsoon_dep), 2) if pd.notna(monsoon_dep) else np.nan,
            "monsoon_category": monsoon_cat,
        })

    state_year_df = pd.DataFrame(summary_rows)
    state_year_df = state_year_df.sort_values(["state_id", "year"]).reset_index(drop=True)

    state_year_path = out_dir / "rainfall_state_year.parquet"
    state_year_df.to_parquet(state_year_path, index=False)
    logger.info("Saved rainfall state-year parquet to %s (%d rows)", state_year_path, len(state_year_df))

    return daily_df, state_year_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_rainfall()
