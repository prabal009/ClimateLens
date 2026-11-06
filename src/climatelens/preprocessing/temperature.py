"""Regional maximum temperature and historical climate trend preprocessing for ClimateLens India."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    TEMPERATURE_REGIONS,
    resolve_raw_file,
)

logger = logging.getLogger(__name__)

SCIENTIFIC_CAVEAT = (
    "Regional monthly maximum temperatures represent homogeneous area averages "
    "used for climate warming trends, NOT individual daily heatwave declarations."
)


def process_temperature(output_dir: Path | None = None) -> pd.DataFrame:
    """Parse IITM regional maximum temperature dataset (1901-2007).

    Steps:
    1. Parse NEW-TXREGION.TXT fixed-format lines for 8 climatic regions.
    2. Convert to tidy monthly timeseries: region_code, region_name, year, month, date, tmax_celsius.
    3. Calculate 1971-2000 climatological baseline normal per (region, month).
    4. Compute monthly temperature anomalies (tmax - baseline_normal).
    5. Compute rolling 12-month and 60-month mean anomalies.
    6. Export regional_temperature.parquet.
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = resolve_raw_file("temperature")
    logger.info("Loading temperature data from %s", txt_path)

    records: list[dict[str, Any]] = []

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or ":" in line_str:
                continue

            parts = line_str.split()
            if len(parts) < 13:
                continue

            tag = parts[0]
            region_code = tag[:5]
            year_str = tag[5:9]
            if not year_str.isdigit():
                continue

            year = int(year_str)
            if region_code not in TEMPERATURE_REGIONS:
                continue

            region_meta = TEMPERATURE_REGIONS[region_code]
            month_vals = [float(p) for p in parts[1:13]]

            for month_idx, val in enumerate(month_vals, start=1):
                date_str = f"{year:04d}-{month_idx:02d}-01"
                records.append({
                    "date": date_str,
                    "year": year,
                    "month": month_idx,
                    "region_code": region_code,
                    "region_name": region_meta["region_name"],
                    "tmax_celsius": round(val, 2),
                    "caveat": SCIENTIFIC_CAVEAT,
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region_code", "date"]).reset_index(drop=True)

    # Compute 1971-2000 climatological normals per (region_code, month)
    baseline_mask = (df["year"] >= 1971) & (df["year"] <= 2000)
    baseline_df = (
        df[baseline_mask]
        .groupby(["region_code", "month"])["tmax_celsius"]
        .mean()
        .reset_index()
        .rename(columns={"tmax_celsius": "baseline_normal_celsius"})
    )
    baseline_df["baseline_normal_celsius"] = baseline_df["baseline_normal_celsius"].round(2)

    df = df.merge(baseline_df, on=["region_code", "month"], how="left")
    df["monthly_anomaly_celsius"] = (df["tmax_celsius"] - df["baseline_normal_celsius"]).round(2)

    # Rolling averages per region (12-month and 60-month)
    df["rolling_12m_anomaly_celsius"] = (
        df.groupby("region_code")["monthly_anomaly_celsius"]
        .transform(lambda s: s.rolling(12, min_periods=6).mean())
        .round(2)
    )
    df["rolling_60m_anomaly_celsius"] = (
        df.groupby("region_code")["monthly_anomaly_celsius"]
        .transform(lambda s: s.rolling(60, min_periods=36).mean())
        .round(2)
    )

    parquet_path = out_dir / "regional_temperature.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved regional temperature parquet to %s (%d rows)", parquet_path, len(df))

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_temperature()
