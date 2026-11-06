"""State-level heatwave days and summer peak temperature preprocessing for ClimateLens India."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from nielytics import Dataset

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    normalize_state_id,
)
from climatelens.preprocessing.loader import load_raw_dataset

logger = logging.getLogger("climatelens.heatwave")


def process_heatwave(output_dir: Path | None = None) -> pd.DataFrame:
    """Ingest and preprocess state-level heatwave records (1967-2024) from XeroFlux Heatwave_Raw.

    Returns:
        Cleaned pd.DataFrame written to state_heatwave.parquet.
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Heatwave_Raw via nielytics.Dataset loader...")
    ds: Dataset = load_raw_dataset("Heatwave_Raw", standardize=True)
    df = ds.frame.copy()

    # Column mapping & normalization
    col_map = {
        "state_ut": "raw_state_name",
        "year": "year",
        "heat_wave_days": "heat_wave_days",
        "severe_heat_wave_days": "severe_heat_wave_days",
        "total_heat_wave_days": "total_heat_wave_days",
        "summer_max_temp_deg_c": "summer_max_temp_deg_c",
        "reporting_agency": "reporting_agency",
    }
    df = df.rename(columns=col_map)

    # State normalization
    df["state_id"] = df["raw_state_name"].apply(normalize_state_id)
    if df["state_id"].isnull().any():
        unmapped = df[df["state_id"].isnull()]["raw_state_name"].unique().tolist()
        logger.warning("Unmapped state names in Heatwave_Raw: %s", unmapped)

    df["state_name"] = df["state_id"].apply(
        lambda sid: STATES_MAP[sid]["state_name"] if sid and sid in STATES_MAP else None
    )

    # Type casting
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["heat_wave_days"] = pd.to_numeric(df["heat_wave_days"], errors="coerce").fillna(0).astype(int)
    df["severe_heat_wave_days"] = (
        pd.to_numeric(df["severe_heat_wave_days"], errors="coerce").fillna(0).astype(int)
    )
    df["total_heat_wave_days"] = (
        pd.to_numeric(df["total_heat_wave_days"], errors="coerce").fillna(0).astype(int)
    )
    df["summer_max_temp_deg_c"] = pd.to_numeric(
        df["summer_max_temp_deg_c"], errors="coerce"
    ).astype(float)

    # Retain clean columns
    cols = [
        "state_id",
        "state_name",
        "year",
        "heat_wave_days",
        "severe_heat_wave_days",
        "total_heat_wave_days",
        "summer_max_temp_deg_c",
        "reporting_agency",
    ]
    df = df[cols].sort_values(["state_id", "year"]).reset_index(drop=True)

    parquet_path = out_dir / "state_heatwave.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info(
        "Saved %d state heatwave records (1967-2024) to %s",
        len(df),
        parquet_path,
    )

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_heatwave()
