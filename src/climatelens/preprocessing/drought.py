"""National drought timeseries preprocessing for ClimateLens India."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nielytics import Dataset

from climatelens.constants import PROCESSED_DATA_DIR, resolve_raw_file
from climatelens.preprocessing.loader import get_raw_sheet_df

logger = logging.getLogger(__name__)

SCIENTIFIC_CAVEAT = (
    "National-level drought area percentage from IIT Gandhinagar India Drought Monitor. "
    "Does NOT provide state-level or district-level drought observations."
)


def process_drought(output_dir: Path | None = None) -> pd.DataFrame:
    """Parse IIT Gandhinagar national drought area timeseries (2021-2026).

    Steps:
    1. Read whitespace-delimited columns.
    2. Format ISO date string YYYY-MM-DD.
    3. Validate 100% sum invariant and monotonic severity hierarchy.
    4. Compute 4-week rolling averages and week-over-week delta metrics.
    5. Export national_drought.parquet.
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_df = get_raw_sheet_df("Drought_Raw")
        logger.info("Loaded %d rows from Drought_Raw", len(raw_df))
        ds = Dataset(raw_df)
        raw_df = ds.frame.copy()
        for col in ["year", "month", "day", "value_4", "value_5", "value_6", "value_7", "value_8", "value_9"]:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
        raw_df = raw_df.rename(columns={
            "value_4": "no_drought_pct",
            "value_5": "d0_d4_pct",
            "value_6": "d1_d4_pct",
            "value_7": "d2_d4_pct",
            "value_8": "d3_d4_pct",
            "value_9": "d4_pct",
        })
    except Exception as exc:
        txt_path = resolve_raw_file("drought")
        logger.info("Falling back to drought text file from %s (%s)", txt_path, exc)
        raw_df = pd.read_csv(
            txt_path,
            sep=r"\s+",
            header=None,
            names=[
                "year",
                "month",
                "day",
                "no_drought_pct",
                "d0_d4_pct",
                "d1_d4_pct",
                "d2_d4_pct",
                "d3_d4_pct",
                "d4_pct",
            ],
        )

    # Validate sum invariant: no_drought + d0_d4 == 100.0 within 0.05%
    sum_diff = (raw_df["no_drought_pct"] + raw_df["d0_d4_pct"]) - 100.0
    if (sum_diff.abs() > 0.05).any():
        bad_idx = sum_diff.abs().idxmax()
        raise ValueError(
            f"Drought sum invariant violated at row {bad_idx}: "
            f"sum={raw_df.loc[bad_idx, 'no_drought_pct'] + raw_df.loc[bad_idx, 'd0_d4_pct']:.4f}"
        )

    # Validate monotonicity: 0 <= d4 <= d3_d4 <= d2_d4 <= d1_d4 <= d0_d4 <= 100.0
    tol = 1e-4
    if ((raw_df["d4_pct"] > raw_df["d3_d4_pct"] + tol) |
        (raw_df["d3_d4_pct"] > raw_df["d2_d4_pct"] + tol) |
        (raw_df["d2_d4_pct"] > raw_df["d1_d4_pct"] + tol) |
        (raw_df["d1_d4_pct"] > raw_df["d0_d4_pct"] + tol) |
        (raw_df["d0_d4_pct"] > 100.0 + tol) |
        (raw_df["d4_pct"] < 0.0 - tol)).any():
        raise ValueError("Drought severity hierarchy monotonicity violated")

    df = raw_df.copy()
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        df["month"].astype(str).str.zfill(2) + "-" +
        df["day"].astype(str).str.zfill(2)
    )
    df = df.sort_values("date").reset_index(drop=True)

    df["geographic_level"] = "national"
    df["entity_id"] = "INDIA_NATIONAL"
    df["entity_name"] = "India (National)"
    df["caveat"] = SCIENTIFIC_CAVEAT

    # Calculate 4-week rolling averages
    df["rolling_4w_d0_d4_pct"] = df["d0_d4_pct"].rolling(4, min_periods=1).mean().round(3)
    df["rolling_4w_d1_d4_pct"] = df["d1_d4_pct"].rolling(4, min_periods=1).mean().round(3)
    df["rolling_4w_d2_d4_pct"] = df["d2_d4_pct"].rolling(4, min_periods=1).mean().round(3)

    # Week-over-week change
    df["weekly_change_d0_d4_pct"] = df["d0_d4_pct"].diff().round(3)

    # Round percentages cleanly
    for col in ["no_drought_pct", "d0_d4_pct", "d1_d4_pct", "d2_d4_pct", "d3_d4_pct", "d4_pct"]:
        df[col] = df[col].round(3)

    parquet_path = out_dir / "national_drought.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved national drought parquet to %s (%d rows)", parquet_path, len(df))

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_drought()
