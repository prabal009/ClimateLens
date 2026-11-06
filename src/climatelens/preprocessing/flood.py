"""Flood inventory preprocessing and standardization for ClimateLens India."""

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

SCIENTIFIC_CAVEAT = (
    "Historical flood inventory (IFI-Impacts, IIT Delhi). Missing impact or casualty fields "
    "are retained as null; zero casualties or damage are never imputed. Coordinates are absent in source."
)


def _clean_str(val: Any) -> str | None:
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "null") else None


def _clean_numeric(val: Any) -> float | None:
    if pd.isna(val) or val is None or val == "":
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def process_flood(output_dir: Path | None = None) -> pd.DataFrame:
    """Parse IIT Delhi India Flood Inventory (1967-2023).

    Steps:
    1. Parse UTF-8-sig CSV (6,876 records).
    2. Parse Start Date and End Date using dayfirst '%d-%m-%Y %H:%M'.
    3. Reconcile comma-separated State values into canonical state_id lists.
    4. Designate primary_state_id (first listed state) and primary_state_name.
    5. Retain blank casualties/impacts as null without zero-fabrication.
    6. Export flood_events.parquet.
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_df = get_raw_sheet_df("Flood_Raw")
        logger.info("Loaded %d rows from Flood_Raw", len(raw_df))
        ds = Dataset(raw_df)
        raw_df = ds.frame.copy()
    except Exception as exc:
        csv_path = resolve_raw_file("flood")
        logger.info("Falling back to flood CSV from %s (%s)", csv_path, exc)
        raw_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    records: list[dict[str, Any]] = []

    for _, row in raw_df.iterrows():
        uei = _clean_str(row.get("UEI")) or f"UEI-FL-{len(records):05d}"
        raw_start = _clean_str(row.get("Start Date"))
        raw_end = _clean_str(row.get("End Date"))

        # Parse start and end timestamps
        start_ts = (
            pd.to_datetime(raw_start, format="%d-%m-%Y %H:%M", errors="coerce")
            if raw_start
            else pd.NaT
        )
        end_ts = (
            pd.to_datetime(raw_end, format="%d-%m-%Y %H:%M", errors="coerce")
            if raw_end
            else pd.NaT
        )

        start_date_str = start_ts.strftime("%Y-%m-%d") if pd.notna(start_ts) else None
        end_date_str = end_ts.strftime("%Y-%m-%d") if pd.notna(end_ts) else None
        year = int(start_ts.year) if pd.notna(start_ts) else None

        # State reconciliation
        raw_states_str = _clean_str(row.get("State"))
        states_affected: list[str] = []
        if raw_states_str:
            tokens = [t.strip() for t in raw_states_str.split(",")]
            for tok in tokens:
                sid = normalize_state_id(tok)
                if sid and sid not in states_affected:
                    states_affected.append(sid)

        primary_state_id = states_affected[0] if states_affected else None
        primary_state_name = (
            STATES_MAP[primary_state_id]["state_name"] if primary_state_id else None
        )

        records.append({
            "event_id": uei,
            "year": year,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "duration_days": _clean_numeric(row.get("Duration(Days)")),
            "primary_state_id": primary_state_id,
            "primary_state_name": primary_state_name,
            "states_affected": states_affected,
            "districts": _clean_str(row.get("Districts")),
            "location": _clean_str(row.get("Location")),
            "latitude": _clean_numeric(row.get("Latitude")),
            "longitude": _clean_numeric(row.get("Longitude")),
            "severity": _clean_numeric(row.get("Severity")),
            "area_affected": _clean_str(row.get("Area Affected")),
            "human_fatality": _clean_numeric(row.get("Human fatality")),
            "human_injured": _clean_numeric(row.get("Human injured")),
            "human_displaced": _clean_numeric(row.get("Human Displaced")),
            "animal_fatality": _clean_numeric(row.get("Animal Fatality")),
            "description_casualties": _clean_str(row.get("Description of Casualties/injured")),
            "extent_of_damage": _clean_str(row.get("Extent of damage ")),
            "main_cause": _clean_str(row.get("Main Cause")),
            "event_source": _clean_str(row.get("Event Source")),
            "caveat": SCIENTIFIC_CAVEAT,
        })

    df = pd.DataFrame(records)
    df = df.sort_values(["year", "start_date", "event_id"], na_position="last").reset_index(drop=True)

    parquet_path = out_dir / "flood_events.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved flood events parquet to %s (%d rows)", parquet_path, len(df))

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_flood()
