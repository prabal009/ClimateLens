"""End-to-end data engineering and preprocessing pipeline runner for ClimateLens India."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from climatelens.constants import PROCESSED_DATA_DIR
from climatelens.preprocessing.boundaries import process_boundaries
from climatelens.preprocessing.cyclones import process_cyclones
from climatelens.preprocessing.drought import process_drought
from climatelens.preprocessing.flood import process_flood
from climatelens.preprocessing.heatwave import process_heatwave
from climatelens.preprocessing.landslide import process_landslide
from climatelens.preprocessing.manifest_builder import (
    generate_manifest,
    generate_quality_report,
)
from climatelens.preprocessing.master_excel import build_master_excel
from climatelens.preprocessing.rainfall import process_rainfall
from climatelens.preprocessing.temperature import process_temperature

logger = logging.getLogger("climatelens.pipeline")


def run_pipeline(output_dir: Path | None = None) -> dict[str, Any]:
    """Execute the full data transformation pipeline.

    Processes all 7 raw datasets into standardized Parquet, GeoJSON,
    metadata manifests, and the consolidated master Excel workbook.
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=== Starting ClimateLens India Data Preprocessing Pipeline ===")

    # Step 1: Boundaries
    t_step = time.time()
    logger.info("[1/10] Processing state boundaries and GeoJSON layers...")
    clean_gdf, simp_gdf = process_boundaries(out_dir)
    logger.info(" -> Boundaries done in %.2fs (36 features)", time.time() - t_step)

    # Step 2: Rainfall
    t_step = time.time()
    logger.info("[2/10] Processing IMD daily rainfall data...")
    rf_daily, rf_year = process_rainfall(out_dir)
    logger.info(" -> Rainfall done in %.2fs (%d daily, %d state-year rows)", time.time() - t_step, len(rf_daily), len(rf_year))

    # Step 3: Heatwave
    t_step = time.time()
    logger.info("[3/10] Processing IMD/MoSPI state-level heatwave days...")
    hw_df = process_heatwave(out_dir)
    logger.info(" -> Heatwave done in %.2fs (%d records)", time.time() - t_step, len(hw_df))

    # Step 4: Temperature
    t_step = time.time()
    logger.info("[4/10] Processing IITM regional maximum temperature data...")
    temp_df = process_temperature(out_dir)
    logger.info(" -> Temperature done in %.2fs (%d records)", time.time() - t_step, len(temp_df))

    # Step 5: Drought
    t_step = time.time()
    logger.info("[5/10] Processing IIT Gandhinagar national drought timeseries...")
    dr_df = process_drought(out_dir)
    logger.info(" -> Drought done in %.2fs (%d observations)", time.time() - t_step, len(dr_df))

    # Step 6: Flood
    t_step = time.time()
    logger.info("[6/10] Processing IIT Delhi flood events inventory...")
    fl_df = process_flood(out_dir)
    logger.info(" -> Flood done in %.2fs (%d events)", time.time() - t_step, len(fl_df))

    # Step 7: Landslide
    t_step = time.time()
    logger.info("[7/10] Processing NASA Global Landslide Catalog with spatial join...")
    ls_df = process_landslide(out_dir)
    logger.info(" -> Landslide done in %.2fs (%d records)", time.time() - t_step, len(ls_df))

    # Step 8: Cyclone
    t_step = time.time()
    logger.info("[8/10] Processing NOAA IBTrACS cyclone track points...")
    cy_df = process_cyclones(out_dir)
    logger.info(" -> Cyclone done in %.2fs (%d points, %d storms)", time.time() - t_step, len(cy_df), cy_df["sid"].nunique())

    # Step 9: Manifests & Quality Reports
    t_step = time.time()
    logger.info("[9/10] Building dataset manifest and quality report...")
    manifest = generate_manifest(out_dir)
    quality = generate_quality_report(out_dir)
    logger.info(" -> Manifests done in %.2fs", time.time() - t_step)

    # Step 10: Master Excel Workbook
    t_step = time.time()
    logger.info("[10/10] Generating consolidated master Excel workbook...")
    excel_path = build_master_excel(out_dir)
    logger.info(" -> Master Excel done in %.2fs (%.2f MB)", time.time() - t_step, excel_path.stat().st_size / 1024 / 1024)
    total_time = time.time() - t0
    logger.info("=== Pipeline completed successfully in %.2fs ===", total_time)

    return {
        "status": "success",
        "total_time_seconds": round(total_time, 2),
        "manifest": manifest,
        "quality_report": quality,
        "excel_path": str(excel_path),
        "excel_size_bytes": excel_path.stat().st_size,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_pipeline()
