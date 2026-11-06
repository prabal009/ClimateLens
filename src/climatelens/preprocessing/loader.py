"""Raw data loader for XeroFlux datasets with Parquet caching and nielytics Dataset profiling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
from nielytics import Dataset
from nielytics.harmonization import DatasetHarmonizer, HarmonizationSchema, StandardClimateRecord
from climatelens.constants import (
    PROJECT_ROOT,
    RAW_CACHE_DIR,
    RAW_DATA_DIR,
    RAW_DATA_SEARCH_PATHS,
)

logger = logging.getLogger("climatelens.loader")

SHEET_FILE_MAPPING: dict[str, str] = {
    "Drought_Raw": "drought_raw.parquet",
    "Heatwave_Raw": "heatwave_raw.parquet",
    "Flood_Raw": "flood_raw.parquet",
    "Landslide_Raw": "landslide_raw.parquet",
    "Rainfall_Raw": "rainfall_raw.parquet",
    "Cyclone_Raw": "cyclone_raw.parquet",
    "State_Attributes": "state_attributes.parquet",
}

EXPECTED_SHEETS = list(SHEET_FILE_MAPPING.keys())

HARMONIZATION_SCHEMAS: dict[str, HarmonizationSchema] = {
    "Heatwave_Raw": HarmonizationSchema(
        hazard="heatwave",
        metric="tmax",
        source_unit="C",
        target_unit="C",
        date_column="Year",
        value_column="Summer_Max_Temp_Deg_C",
        state_column="State_UT",
        temporal_resolution="annual",
        source="IMD / MoSPI",
    ),
    "Rainfall_Raw": HarmonizationSchema(
        hazard="flood",
        metric="rainfall",
        source_unit="mm",
        target_unit="mm",
        date_column="Date",
        value_column="Daily Actual",
        state_column="State",
        temporal_resolution="daily",
        source="IMD / NWIC",
    ),
    "Flood_Raw": HarmonizationSchema(
        hazard="flood",
        metric="fatalities",
        source_unit="count",
        target_unit="count",
        date_column="Start Date",
        value_column="Human fatality",
        state_column="State",
        district_column="Districts",
        latitude_column="Latitude",
        longitude_column="Longitude",
        temporal_resolution="event",
        source="IFI-Impacts",
    ),
    "Landslide_Raw": HarmonizationSchema(
        hazard="landslide",
        metric="events",
        source_unit="count",
        target_unit="count",
        date_column="event_date",
        value_column="events",
        state_column="admin_division_name",
        latitude_column="latitude",
        longitude_column="longitude",
        temporal_resolution="event",
        source="riseuppant / GitHub",
    ),
    "Cyclone_Raw": HarmonizationSchema(
        hazard="cyclone",
        metric="wind",
        source_unit="km/h",
        target_unit="km/h",
        date_column="ISO_TIME",
        value_column="wind",
        latitude_column="LAT",
        longitude_column="LON",
        temporal_resolution="event",
        source="IBTrACS",
    ),
}


def find_xeroflux_workbook(search_paths: list[Path] | None = None) -> Path:
    """Locate authoritative XeroFlux Dataset.xlsx or XeroFlux Datasets.xlsx."""
    candidates = [
        "XeroFlux Dataset.xlsx",
        "XeroFlux Datasets.xlsx",
        "xeroflux dataset.xlsx",
        "xeroflux datasets.xlsx",
    ]

    paths_to_check: list[Path] = []
    if search_paths:
        paths_to_check.extend(search_paths)
    paths_to_check.extend([PROJECT_ROOT, RAW_DATA_DIR])
    paths_to_check.extend(RAW_DATA_SEARCH_PATHS)

    seen: set[Path] = set()
    for directory in paths_to_check:
        d = Path(directory).resolve()
        if d in seen or not d.is_dir():
            continue
        seen.add(d)
        for name in candidates:
            target = d / name
            if target.is_file():
                logger.info("Found XeroFlux workbook at %s", target)
                return target

    raise FileNotFoundError(
        "Could not locate 'XeroFlux Dataset.xlsx' or 'XeroFlux Datasets.xlsx' in: "
        f"{[str(p) for p in seen]}"
    )


def extract_xeroflux_sheets(
    workbook_path: Path | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, Path]:
    """Extract all sheets from XeroFlux Dataset.xlsx into cached Parquet files in single pass."""
    out_dir = Path(cache_dir or RAW_CACHE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if all parquet cache files already exist
    all_cached = True
    output_paths: dict[str, Path] = {}
    for sheet_name, filename in SHEET_FILE_MAPPING.items():
        p = out_dir / filename
        output_paths[sheet_name] = p
        if not p.is_file() or p.stat().st_size == 0:
            all_cached = False

    if all_cached and not force_refresh:
        logger.info("All raw cache parquet files present in %s; skipping XLSX extraction", out_dir)
        return output_paths

    wb_path = workbook_path or find_xeroflux_workbook()
    logger.info("Opening %s (read_only=True) for single-pass extraction...", wb_path)

    wb = openpyxl.load_workbook(str(wb_path), read_only=True, data_only=True)
    available_sheets = set(wb.sheetnames)

    for sheet_name, filename in SHEET_FILE_MAPPING.items():
        if sheet_name not in available_sheets:
            logger.warning("Sheet '%s' not found in %s; skipping", sheet_name, wb_path)
            continue

        target_parquet = out_dir / filename
        if target_parquet.is_file() and not force_refresh and target_parquet.stat().st_size > 0:
            logger.info("Sheet '%s' already cached at %s", sheet_name, target_parquet)
            continue

        logger.info("Extracting worksheet '%s'...", sheet_name)
        ws = wb[sheet_name]
        row_iter = ws.iter_rows(values_only=True)
        try:
            raw_header = next(row_iter)
        except StopIteration:
            logger.warning("Worksheet '%s' is empty", sheet_name)
            continue

        # Clean header column names
        header = [
            str(col).strip() if col is not None and str(col).strip() != "" else f"col_{idx}"
            for idx, col in enumerate(raw_header)
        ]
        num_cols = len(header)

        data: list[list[Any]] = []
        for r in row_iter:
            row = list(r)
            if len(row) < num_cols:
                row.extend([None] * (num_cols - len(row)))
            elif len(row) > num_cols:
                row = row[:num_cols]
            data.append(row)

        df = pd.DataFrame(data, columns=header)

        # Convert all columns to string/object representation for raw cache fidelity
        for col in df.columns:
            # Replace whitespace-only strings with empty string or keep as string
            df[col] = df[col].astype(str).replace({"None": None, "nan": None, "<NA>": None})

        df.to_parquet(target_parquet, index=False, engine="pyarrow")
        logger.info(
            "Extracted '%s': %d rows, %d cols -> %s",
            sheet_name,
            len(df),
            len(df.columns),
            target_parquet.name,
        )

    wb.close()
    return output_paths


def get_raw_sheet_df(
    sheet_name: str,
    cache_dir: Path | None = None,
    workbook_path: Path | None = None,
) -> pd.DataFrame:
    """Retrieve raw DataFrame for a worksheet from cache (extracting if necessary)."""
    out_dir = Path(cache_dir or RAW_CACHE_DIR)
    if sheet_name not in SHEET_FILE_MAPPING:
        raise KeyError(f"Unknown sheet '{sheet_name}'. Expected one of {list(SHEET_FILE_MAPPING.keys())}")

    target_file = out_dir / SHEET_FILE_MAPPING[sheet_name]
    if not target_file.is_file() or target_file.stat().st_size == 0:
        extract_xeroflux_sheets(workbook_path=workbook_path, cache_dir=out_dir)

    return pd.read_parquet(target_file)


def load_raw_dataset(
    sheet_name: str,
    cache_dir: Path | None = None,
    workbook_path: Path | None = None,
    standardize: bool = True,
    profile: bool = True,
) -> Dataset:
    """Load worksheet as a nielytics.Dataset instance with optional profiling and standardization."""
    df = get_raw_sheet_df(sheet_name, cache_dir=cache_dir, workbook_path=workbook_path)
    ds = Dataset(df)

    if profile:
        profile_res = ds.profile()
        missing_res = ds.missing_report()
        logger.debug(
            "Profiled %s: %d rows, %d cols, memory: %d bytes, missing cols: %d",
            sheet_name,
            profile_res.get("rows", 0),
            profile_res.get("columns", 0),
            profile_res.get("memory_bytes", 0),
            len(missing_res),
        )

    if standardize:
        ds = ds.standardize_columns().clean_text()

    return ds


def load_all_raw_datasets(
    cache_dir: Path | None = None,
    workbook_path: Path | None = None,
    standardize: bool = True,
) -> dict[str, Dataset]:
    """Extract and load all worksheets as nielytics.Dataset instances."""
    extract_xeroflux_sheets(workbook_path=workbook_path, cache_dir=cache_dir)
    return {
        sheet: load_raw_dataset(sheet, cache_dir=cache_dir, workbook_path=workbook_path, standardize=standardize)
        for sheet in SHEET_FILE_MAPPING
    }


def harmonize_raw_dataset(
    sheet_name: str,
    cache_dir: Path | None = None,
    workbook_path: Path | None = None,
) -> pd.DataFrame:
    """Harmonize raw extracted worksheet records into StandardClimateRecord format using nielytics."""
    if sheet_name not in HARMONIZATION_SCHEMAS:
        raise KeyError(
            f"No harmonization schema registered for '{sheet_name}'. "
            f"Available schemas: {list(HARMONIZATION_SCHEMAS.keys())}"
        )

    df = get_raw_sheet_df(sheet_name, cache_dir=cache_dir, workbook_path=workbook_path).copy()

    # Apply dataset-specific column adapters if required
    if sheet_name == "Landslide_Raw" and "events" not in df.columns:
        df["events"] = 1.0
    elif sheet_name == "Cyclone_Raw" and "wind" not in df.columns:
        raw_wind = pd.to_numeric(df.get("USA_WIND"), errors="coerce").fillna(
            pd.to_numeric(df.get("WMO_WIND"), errors="coerce")
        )
        # Convert knots to km/h (1 knot = 1.852 km/h)
        df["wind"] = raw_wind * 1.852

    schema = HARMONIZATION_SCHEMAS[sheet_name]
    harmonizer = DatasetHarmonizer(schema=schema)
    analysis_result = harmonizer.harmonize(df)
    return DatasetHarmonizer.to_dataframe(analysis_result)
