"""Consolidated master Excel workbook generator for ClimateLens India.

Produces ClimateLens_India_Master_Dataset.xlsx (< 100MB) from XeroFlux datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from climatelens.constants import (
    DATASET_METADATA,
    PROCESSED_DATA_DIR,
    STATES_MAP,
    TEMPERATURE_REGIONS,
)

logger = logging.getLogger("climatelens.master_excel")

MAX_FILE_SIZE_BYTES = 100_000_000  # 100 MB strictly enforced limit


def build_master_excel(
    output_dir: Path | None = None,
    filename: str = "ClimateLens_India_Master_Dataset.xlsx",
) -> Path:
    """Build the official 12-sheet submission Excel workbook from XeroFlux primary data.

    Sheets:
    1. README
    2. Master Climate Data
    3. Heatwave Data (MoSPI)
    4. Landslide Data (NASA GLC)
    5. State Boundary Data
    6. Drought Data (IITM / MoES)
    7. Monthly Max Temperature (IITM)
    8. Flood Data (IFI)
    9. Rainfall Data (IMD)
    10. Cyclone Data (IBTrACS)
    11. Manifest
    12. Data Dictionary
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / filename

    logger.info("Generating master Excel workbook at %s", target_path)

    # Load processed datasets from disk (or generate on demand)
    b_path = out_dir / "state_boundaries.geojson"
    rf_daily_p = out_dir / "rainfall_daily.parquet"
    rf_year_p = out_dir / "rainfall_state_year.parquet"
    temp_p = out_dir / "regional_temperature.parquet"
    dr_p = out_dir / "national_drought.parquet"
    fl_p = out_dir / "flood_events.parquet"
    ls_p = out_dir / "landslide_events.parquet"
    cy_p = out_dir / "cyclone_track_points.parquet"
    hw_p = out_dir / "state_heatwave.parquet"

    # Ensure state heatwave exists
    if not hw_p.exists():
        from climatelens.preprocessing.heatwave import process_heatwave
        process_heatwave(out_dir)

    for p in [b_path, rf_daily_p, rf_year_p, temp_p, dr_p, fl_p, ls_p, cy_p, hw_p]:
        if not p.exists():
            raise FileNotFoundError(f"Prerequisite file {p} missing. Run preprocessing pipeline first.")

    gdf = gpd.read_file(b_path)
    rf_daily = pd.read_parquet(rf_daily_p)
    rf_year = pd.read_parquet(rf_year_p)
    temp = pd.read_parquet(temp_p)
    dr = pd.read_parquet(dr_p)
    fl = pd.read_parquet(fl_p)
    ls = pd.read_parquet(ls_p)
    cy = pd.read_parquet(cy_p)
    hw = pd.read_parquet(hw_p)

    # 1. Sheet: README
    readme_data = [
        {"Section": "Project Title", "Content": "ClimateLens India: Interactive Climate Disaster Intelligence Dashboard"},
        {"Section": "Competition", "Content": "NIErnaya 2026 — Round 2"},
        {"Section": "Team Role", "Content": "Teammate 1: Data Science, Climate Data Engineering & Analytical Foundation"},
        {"Section": "Primary Data Source", "Content": "XeroFlux Dataset.xlsx (Authoritative Phase 2 Raw Source)"},
        {"Section": "Core Analytics Library", "Content": "NIElytics (v2.0) by Prabal Pratap Singh"},
        {"Section": "Disaster Domains Covered", "Content": "5 Core Categories: Heatwaves, Rainfall, Floods, Drought, Cyclones, and Landslides"},
        {"Section": "Key Deliverables", "Content": "Parquet archives, validated GeoJSON layers, Plotly chart specs, unified adapter, and Master Excel"},
        {"Section": "Scientific Integrity Note", "Content": "Zero data fabrication. Missing values retained as N/A. National drought and regional temperatures strictly preserved at authoritative spatial resolutions."},
        {"Section": "Workbook Size Constraint", "Content": "Strictly under 100,000,000 bytes (100 MB). Compressed OpenPyXL XLSX storage."},
    ]
    readme_df = pd.DataFrame(readme_data)

    # 2. Sheet: Master Climate Data (Synthesis per 36 States/UTs)
    fl_summary = fl.groupby("primary_state_id").agg(
        flood_events_count=("event_id", "count"),
        total_fatalities_reported=("human_fatality", "sum"),
        fatalities_records_count=("human_fatality", lambda s: int(s.notna().sum())),
    ).reset_index()

    ls_summary = ls.groupby("state_id").agg(
        landslide_incidents_count=("event_id", "count"),
        mean_reported_magnitude=("reported_magnitude", lambda s: round(float(s.mean()), 2)),
    ).reset_index()

    hw_summary = hw.groupby("state_id").agg(
        total_heat_wave_days=("total_heat_wave_days", "sum"),
        peak_summer_temp=("summer_max_temp_deg_c", lambda s: round(float(s.dropna().max()), 2) if s.notna().any() else "N/A"),
    ).reset_index()

    rf_2026 = rf_year[rf_year["year"] == 2026].set_index("state_id")

    master_rows = []
    for sid, meta in sorted(STATES_MAP.items(), key=lambda x: x[1]["state_name"]):
        fl_match = fl_summary[fl_summary["primary_state_id"] == sid]
        ls_match = ls_summary[ls_summary["state_id"] == sid]
        hw_match = hw_summary[hw_summary["state_id"] == sid]

        monsoon_act = rf_2026.loc[sid, "monsoon_cumulative_actual_mm"] if sid in rf_2026.index else None
        monsoon_norm = rf_2026.loc[sid, "monsoon_cumulative_normal_mm"] if sid in rf_2026.index else None
        monsoon_dep = rf_2026.loc[sid, "monsoon_departure_pct"] if sid in rf_2026.index else None
        monsoon_cat = rf_2026.loc[sid, "monsoon_category"] if sid in rf_2026.index else None

        total_hw = int(hw_match.iloc[0]["total_heat_wave_days"]) if not hw_match.empty else 0
        peak_t = hw_match.iloc[0]["peak_summer_temp"] if not hw_match.empty else "N/A"

        master_rows.append({
            "State ID": sid,
            "State Name": meta["state_name"],
            "State Code": meta["state_code"],
            "Census STCode": meta["stcode"],
            "Geographic Region": meta["region"],
            "Entity Type": meta["type"],
            "Monsoon Actual Rainfall 2026 (mm)": monsoon_act if pd.notna(monsoon_act) else "N/A",
            "Monsoon Normal Rainfall (mm)": monsoon_norm if pd.notna(monsoon_norm) else "N/A",
            "Monsoon Rainfall Departure (%)": monsoon_dep if pd.notna(monsoon_dep) else "N/A",
            "Monsoon Rainfall Category": monsoon_cat if pd.notna(monsoon_cat) else "N/A",
            "Total Recorded Heatwave Days (1967-2024)": total_hw,
            "Peak Summer Max Temperature (°C)": peak_t,
            "Historical Flood Events Count (1967-2023)": int(fl_match.iloc[0]["flood_events_count"]) if not fl_match.empty else 0,
            "Total Recorded Flood Fatalities": int(fl_match.iloc[0]["total_fatalities_reported"]) if not fl_match.empty and pd.notna(fl_match.iloc[0]["total_fatalities_reported"]) else "N/A",
            "Historical Landslide Incidents Count (NASA GLC)": int(ls_match.iloc[0]["landslide_incidents_count"]) if not ls_match.empty else 0,
            "Mean Landslide Severity Score": ls_match.iloc[0]["mean_reported_magnitude"] if not ls_match.empty else "N/A",
        })
    master_df = pd.DataFrame(master_rows)

    # 3. Sheet: Heatwave Data (MoSPI)
    hw_export = hw.copy()
    hw_export.columns = [c.replace("_", " ").title() for c in hw_export.columns]
    hw_export = hw_export.fillna("N/A")

    # 4. Sheet: Landslide Data (NASA GLC)
    ls_export = ls.copy()
    ls_export.columns = [c.replace("_", " ").title() for c in ls_export.columns]
    ls_export = ls_export.fillna("N/A")

    # 5. Sheet: State Boundary Data
    b_rows = []
    for _, row in gdf.iterrows():
        sid = row["state_id"]
        meta = STATES_MAP[sid]
        minx, miny, maxx, maxy = row["geometry"].bounds
        b_rows.append({
            "State ID": sid,
            "State Name": meta["state_name"],
            "State Code": meta["state_code"],
            "Census STCode": meta["stcode"],
            "Entity Type": meta["type"],
            "Region": meta["region"],
            "Bounding Box Min Lon": round(minx, 4),
            "Bounding Box Min Lat": round(miny, 4),
            "Bounding Box Max Lon": round(maxx, 4),
            "Bounding Box Max Lat": round(maxy, 4),
            "CRS": "EPSG:4326 (WGS84)",
            "Source Agency": str(row.get("source_agency", "Survey of India")),
        })
    b_df = pd.DataFrame(b_rows)

    # 6. Sheet: Drought Data
    dr_export = dr.copy()
    dr_export["date"] = dr_export["date"].dt.strftime("%Y-%m-%d")
    dr_export.columns = [c.replace("_", " ").title() for c in dr_export.columns]
    dr_export = dr_export.fillna("N/A")

    # 7. Sheet: Monthly Maximum Temperature (IITM)
    temp_export = temp.copy()
    temp_export["date"] = temp_export["date"].dt.strftime("%Y-%m-%d")
    temp_export.columns = [c.replace("_", " ").title() for c in temp_export.columns]
    temp_export = temp_export.fillna("N/A")

    # 8. Sheet: Flood Data (IFI)
    fl_export = fl.drop(columns=["states_affected"]).copy()
    fl_export.columns = [c.replace("_", " ").title() for c in fl_export.columns]
    fl_export = fl_export.fillna("N/A")

    # 9. Sheet: Rainfall Data (IMD)
    rf_export = rf_daily.copy()
    rf_export["date"] = rf_export["date"].dt.strftime("%Y-%m-%d")
    rf_export.columns = [c.replace("_", " ").title() for c in rf_export.columns]
    rf_export = rf_export.fillna("N/A")

    # 10. Sheet: Cyclone Data (IBTrACS)
    cy_export = cy.copy()
    cy_export.columns = [c.replace("_", " ").title() for c in cy_export.columns]
    cy_export = cy_export.fillna("N/A")

    # 11. Sheet: Manifest
    manifest_rows = []
    for key, m in DATASET_METADATA.items():
        manifest_rows.append({
            "Dataset Key": key,
            "Source Name": m.get("name", "N/A"),
            "Provider": m.get("provider", "N/A"),
            "Citation": m.get("citation", "N/A"),
            "Coverage": m.get("coverage", "N/A"),
            "Time Period": m.get("time_period", "N/A"),
            "Direct URL": m.get("url", "N/A"),
            "Scientific Limitations": "; ".join(m.get("limitations", [])),
        })
    manifest_df = pd.DataFrame(manifest_rows)

    # 12. Sheet: Data Dictionary
    dictionary_rows = [
        {"Worksheet": "Master Climate Data", "Column Name": "State ID", "Data Type": "String", "Unit": "N/A", "Description": "Canonical uppercase standardized identifier for 36 Indian States/UTs."},
        {"Worksheet": "Master Climate Data", "Column Name": "Total Recorded Heatwave Days", "Data Type": "Integer", "Unit": "Days", "Description": "Cumulative summer heatwave days from IMD/MoSPI Envistats (1967-2024)."},
        {"Worksheet": "Master Climate Data", "Column Name": "Peak Summer Max Temperature", "Data Type": "Float", "Unit": "°C", "Description": "Highest recorded summer maximum temperature recorded for state station network."},
        {"Worksheet": "Master Climate Data", "Column Name": "Monsoon Actual Rainfall 2026 (mm)", "Data Type": "Float", "Unit": "mm", "Description": "Season cumulative actual rainfall as of September 30, 2026 (IMD)."},
        {"Worksheet": "Master Climate Data", "Column Name": "Historical Flood Events Count", "Data Type": "Integer", "Unit": "Events", "Description": "Total recorded flood occurrences in primary state (1967-2023, IIT Delhi)."},
        {"Worksheet": "Master Climate Data", "Column Name": "Historical Landslide Incidents Count", "Data Type": "Integer", "Unit": "Incidents", "Description": "Total recorded landslide occurrences spatially intersecting state polygon (NASA GLC)."},
        {"Worksheet": "Heatwave Data (MoSPI)", "Column Name": "Heat Wave Days", "Data Type": "Integer", "Unit": "Days", "Description": "Official summer season recorded heatwave days from IMD / MoSPI Envistats."},
        {"Worksheet": "Landslide Data (NASA GLC)", "Column Name": "Landslide Trigger", "Data Type": "String", "Unit": "N/A", "Description": "Trigger mechanism reported in NASA Global Landslide Catalog (e.g. rain, downpour, monsoonal rain)."},
        {"Worksheet": "Drought Data", "Column Name": "No Drought Pct", "Data Type": "Float", "Unit": "%", "Description": "National land area percentage under normal or wet moisture conditions (IIT Gandhinagar)."},
        {"Worksheet": "Drought Data", "Column Name": "D0 D4 Pct", "Data Type": "Float", "Unit": "%", "Description": "National land area percentage experiencing abnormally dry or worse conditions (D0-D4)."},
        {"Worksheet": "Monthly Max Temperature (IITM)", "Column Name": "Tmax Celsius", "Data Type": "Float", "Unit": "°C", "Description": "Monthly mean maximum surface temperature for the homogeneous climatic region (IITM Pune)."},
        {"Worksheet": "Flood Data (IFI)", "Column Name": "Human Fatality", "Data Type": "Float", "Unit": "Count", "Description": "Verified reported flood fatalities (unverified/missing kept as N/A, never zero)."},
        {"Worksheet": "Rainfall Data (IMD)", "Column Name": "Daily Actual Mm", "Data Type": "Float", "Unit": "mm", "Description": "Daily observed rainfall across IMD recording stations (missing kept as N/A)."},
        {"Worksheet": "Cyclone Data (IBTrACS)", "Column Name": "Wmo Wind Kmh", "Data Type": "Float", "Unit": "km/h", "Description": "WMO agency sustained wind converted from knots (1 knot = 1.852 km/h)."},
    ]
    dict_df = pd.DataFrame(dictionary_rows)

    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        readme_df.to_excel(writer, sheet_name="README", index=False)
        master_df.to_excel(writer, sheet_name="Master Climate Data", index=False)
        hw_export.to_excel(writer, sheet_name="Heatwave Data (MoSPI)", index=False)
        ls_export.to_excel(writer, sheet_name="Landslide Data (NASA GLC)", index=False)
        b_df.to_excel(writer, sheet_name="State Boundary Data", index=False)
        dr_export.to_excel(writer, sheet_name="Drought Data", index=False)
        temp_export.to_excel(writer, sheet_name="Monthly Max Temperature (IITM)", index=False)
        fl_export.to_excel(writer, sheet_name="Flood Data (IFI)", index=False)
        rf_export.to_excel(writer, sheet_name="Rainfall Data (IMD)", index=False)
        cy_export.to_excel(writer, sheet_name="Cyclone Data (IBTrACS)", index=False)
        manifest_df.to_excel(writer, sheet_name="Manifest", index=False)
        dict_df.to_excel(writer, sheet_name="Data Dictionary", index=False)

    size_bytes = target_path.stat().st_size
    logger.info("Master Excel successfully generated: %d bytes (%.2f MB)", size_bytes, size_bytes / 1024 / 1024)

    if size_bytes >= MAX_FILE_SIZE_BYTES:
        raise ValueError(f"Master Excel size {size_bytes} exceeds strict 100MB limit ({MAX_FILE_SIZE_BYTES} bytes)")

    return target_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_master_excel()
