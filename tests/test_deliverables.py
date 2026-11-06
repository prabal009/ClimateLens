"""Unit tests for deliverables, manifests, quality reports, and Excel workbook integrity."""

import json
from pathlib import Path
import openpyxl
import pandas as pd
import pytest

from climatelens.constants import PROCESSED_DATA_DIR


def test_processed_parquet_files_exist():
    expected_parquets = [
        "rainfall_daily.parquet",
        "rainfall_state_year.parquet",
        "regional_temperature.parquet",
        "national_drought.parquet",
        "flood_events.parquet",
        "landslide_events.parquet",
        "cyclone_track_points.parquet",
        "state_heatwave.parquet",
    ]
    for fname in expected_parquets:
        p = PROCESSED_DATA_DIR / fname
        assert p.exists(), f"Missing deliverable parquet: {fname}"
        df = pd.read_parquet(p)
        assert len(df) > 0, f"Deliverable parquet is empty: {fname}"


def test_manifest_and_quality_reports():
    manifest_p = PROCESSED_DATA_DIR / "manifest.json"
    quality_p = PROCESSED_DATA_DIR / "quality_report.json"

    assert manifest_p.exists()
    assert quality_p.exists()

    with open(manifest_p, "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["total_datasets"] == 7
    assert len(m["datasets"]) == 7

    with open(quality_p, "r", encoding="utf-8") as f:
        q = json.load(f)
    assert q["overall_status"] == "PASSED"
    assert len(q["audits"]) == 7


def test_master_excel_deliverable_integrity():
    excel_p = PROCESSED_DATA_DIR / "ClimateLens_India_Master_Dataset.xlsx"
    assert excel_p.exists(), "Master Excel workbook does not exist"

    size_bytes = excel_p.stat().st_size
    assert size_bytes < 100_000_000, f"Excel file exceeds 100MB: {size_bytes} bytes"
    assert size_bytes > 1_000_000, f"Excel file unexpectedly small: {size_bytes} bytes"

    # Verify all 11 required sheets exist and are readable
    wb = openpyxl.load_workbook(excel_p, read_only=True)
    expected_sheets = [
        "README",
        "Master Climate Data",
        "Heatwave Data (MoSPI)",
        "Landslide Data (NASA GLC)",
        "State Boundary Data",
        "Drought Data",
        "Monthly Max Temperature (IITM)",
        "Flood Data (IFI)",
        "Rainfall Data (IMD)",
        "Cyclone Data (IBTrACS)",
        "Manifest",
        "Data Dictionary",
    ]
    for sheet_name in expected_sheets:
        assert sheet_name in wb.sheetnames, f"Missing worksheet: '{sheet_name}' in master Excel"

    # Read Master Climate Data rows
    ws_master = wb["Master Climate Data"]
    rows = list(ws_master.iter_rows(values_only=True))
    assert len(rows) == 37  # Header + 36 states
    wb.close()
