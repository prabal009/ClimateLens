"""ClimateLens India - Analytical constants, metadata, and canonical spatial mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nielytics.geospatial.india_names import normalize_district_name, normalize_state_name

# Directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_CACHE_DIR = DATA_DIR / "raw_cache"

# Fallback search paths for raw datasets
RAW_DATA_SEARCH_PATHS = [
    RAW_DATA_DIR,
    Path("/run/media/prabal/SANDISK E/le moksh"),
    Path("/home/prabal/Desktop/nielytics raw dataset"),
]

# 36 Canonical States / Union Territories of India (aligned with NWIC GeoJSON 2025)
# Key: canonical state_id (uppercase with underscores)
STATES_MAP: dict[str, dict[str, Any]] = {
    "ANDAMAN_AND_NICOBAR_ISLANDS": {
        "state_id": "ANDAMAN_AND_NICOBAR_ISLANDS",
        "state_name": "Andaman and Nicobar Islands",
        "state_code": "AN",
        "stcode": "35",
        "type": "Union Territory",
        "region": "Islands",
    },
    "ANDHRA_PRADESH": {
        "state_id": "ANDHRA_PRADESH",
        "state_name": "Andhra Pradesh",
        "state_code": "AP",
        "stcode": "28",
        "type": "State",
        "region": "South",
    },
    "ARUNACHAL_PRADESH": {
        "state_id": "ARUNACHAL_PRADESH",
        "state_name": "Arunachal Pradesh",
        "state_code": "AR",
        "stcode": "12",
        "type": "State",
        "region": "Northeast",
    },
    "ASSAM": {
        "state_id": "ASSAM",
        "state_name": "Assam",
        "state_code": "AS",
        "stcode": "18",
        "type": "State",
        "region": "Northeast",
    },
    "BIHAR": {
        "state_id": "BIHAR",
        "state_name": "Bihar",
        "state_code": "BR",
        "stcode": "10",
        "type": "State",
        "region": "East",
    },
    "CHANDIGARH": {
        "state_id": "CHANDIGARH",
        "state_name": "Chandigarh",
        "state_code": "CH",
        "stcode": "04",
        "type": "Union Territory",
        "region": "North",
    },
    "CHHATTISGARH": {
        "state_id": "CHHATTISGARH",
        "state_name": "Chhattisgarh",
        "state_code": "CG",
        "stcode": "22",
        "type": "State",
        "region": "Central",
    },
    "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU": {
        "state_id": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
        "state_name": "Dadra & Nagar Haveli and Daman & Diu",
        "state_code": "DH",
        "stcode": "38",
        "type": "Union Territory",
        "region": "West",
    },
    "DELHI": {
        "state_id": "DELHI",
        "state_name": "Delhi",
        "state_code": "DL",
        "stcode": "07",
        "type": "Union Territory",
        "region": "North",
    },
    "GOA": {
        "state_id": "GOA",
        "state_name": "Goa",
        "state_code": "GA",
        "stcode": "30",
        "type": "State",
        "region": "West",
    },
    "GUJARAT": {
        "state_id": "GUJARAT",
        "state_name": "Gujarat",
        "state_code": "GJ",
        "stcode": "24",
        "type": "State",
        "region": "West",
    },
    "HARYANA": {
        "state_id": "HARYANA",
        "state_name": "Haryana",
        "state_code": "HR",
        "stcode": "06",
        "type": "State",
        "region": "North",
    },
    "HIMACHAL_PRADESH": {
        "state_id": "HIMACHAL_PRADESH",
        "state_name": "Himachal Pradesh",
        "state_code": "HP",
        "stcode": "02",
        "type": "State",
        "region": "North",
    },
    "JAMMU_AND_KASHMIR": {
        "state_id": "JAMMU_AND_KASHMIR",
        "state_name": "Jammu and Kashmir",
        "state_code": "JK",
        "stcode": "01",
        "type": "Union Territory",
        "region": "North",
    },
    "JHARKHAND": {
        "state_id": "JHARKHAND",
        "state_name": "Jharkhand",
        "state_code": "JH",
        "stcode": "20",
        "type": "State",
        "region": "East",
    },
    "KARNATAKA": {
        "state_id": "KARNATAKA",
        "state_name": "Karnataka",
        "state_code": "KA",
        "stcode": "29",
        "type": "State",
        "region": "South",
    },
    "KERALA": {
        "state_id": "KERALA",
        "state_name": "Kerala",
        "state_code": "KL",
        "stcode": "32",
        "type": "State",
        "region": "South",
    },
    "LADAKH": {
        "state_id": "LADAKH",
        "state_name": "Ladakh",
        "state_code": "LK",
        "stcode": "37",
        "type": "Union Territory",
        "region": "North",
    },
    "LAKSHADWEEP": {
        "state_id": "LAKSHADWEEP",
        "state_name": "Lakshadweep",
        "state_code": "LD",
        "stcode": "31",
        "type": "Union Territory",
        "region": "Islands",
    },
    "MADHYA_PRADESH": {
        "state_id": "MADHYA_PRADESH",
        "state_name": "Madhya Pradesh",
        "state_code": "MP",
        "stcode": "23",
        "type": "State",
        "region": "Central",
    },
    "MAHARASHTRA": {
        "state_id": "MAHARASHTRA",
        "state_name": "Maharashtra",
        "state_code": "MH",
        "stcode": "27",
        "type": "State",
        "region": "West",
    },
    "MANIPUR": {
        "state_id": "MANIPUR",
        "state_name": "Manipur",
        "state_code": "MN",
        "stcode": "14",
        "type": "State",
        "region": "Northeast",
    },
    "MEGHALAYA": {
        "state_id": "MEGHALAYA",
        "state_name": "Meghalaya",
        "state_code": "ML",
        "stcode": "17",
        "type": "State",
        "region": "Northeast",
    },
    "MIZORAM": {
        "state_id": "MIZORAM",
        "state_name": "Mizoram",
        "state_code": "MZ",
        "stcode": "15",
        "type": "State",
        "region": "Northeast",
    },
    "NAGALAND": {
        "state_id": "NAGALAND",
        "state_name": "Nagaland",
        "state_code": "NL",
        "stcode": "13",
        "type": "State",
        "region": "Northeast",
    },
    "ODISHA": {
        "state_id": "ODISHA",
        "state_name": "Odisha",
        "state_code": "OD",
        "stcode": "21",
        "type": "State",
        "region": "East",
    },
    "PUDUCHERRY": {
        "state_id": "PUDUCHERRY",
        "state_name": "Puducherry",
        "state_code": "PY",
        "stcode": "34",
        "type": "Union Territory",
        "region": "South",
    },
    "PUNJAB": {
        "state_id": "PUNJAB",
        "state_name": "Punjab",
        "state_code": "PB",
        "stcode": "03",
        "type": "State",
        "region": "North",
    },
    "RAJASTHAN": {
        "state_id": "RAJASTHAN",
        "state_name": "Rajasthan",
        "state_code": "RJ",
        "stcode": "08",
        "type": "State",
        "region": "North",
    },
    "SIKKIM": {
        "state_id": "SIKKIM",
        "state_name": "Sikkim",
        "state_code": "SK",
        "stcode": "11",
        "type": "State",
        "region": "Northeast",
    },
    "TAMIL_NADU": {
        "state_id": "TAMIL_NADU",
        "state_name": "Tamil Nadu",
        "state_code": "TN",
        "stcode": "33",
        "type": "State",
        "region": "South",
    },
    "TELANGANA": {
        "state_id": "TELANGANA",
        "state_name": "Telangana",
        "state_code": "TS",
        "stcode": "36",
        "type": "State",
        "region": "South",
    },
    "TRIPURA": {
        "state_id": "TRIPURA",
        "state_name": "Tripura",
        "state_code": "TR",
        "stcode": "16",
        "type": "State",
        "region": "Northeast",
    },
    "UTTAR_PRADESH": {
        "state_id": "UTTAR_PRADESH",
        "state_name": "Uttar Pradesh",
        "state_code": "UP",
        "stcode": "09",
        "type": "State",
        "region": "North",
    },
    "UTTARAKHAND": {
        "state_id": "UTTARAKHAND",
        "state_name": "Uttarakhand",
        "state_code": "UK",
        "stcode": "05",
        "type": "State",
        "region": "North",
    },
    "WEST_BENGAL": {
        "state_id": "WEST_BENGAL",
        "state_name": "West Bengal",
        "state_code": "WB",
        "stcode": "19",
        "type": "State",
        "region": "East",
    },
}

# Aliases dictionary for fuzzy / raw name normalization to canonical state_id
STATE_NAME_ALIASES: dict[str, str] = {
    # Direct matches
    "ANDAMAN & NICOBAR ISLAND": "ANDAMAN_AND_NICOBAR_ISLANDS",
    "ANDAMAN & NICOBAR ISLANDS": "ANDAMAN_AND_NICOBAR_ISLANDS",
    "ANDAMAN AND NICOBAR ISLANDS": "ANDAMAN_AND_NICOBAR_ISLANDS",
    "ANDAMAN & NICOBAR (UT)": "ANDAMAN_AND_NICOBAR_ISLANDS",
    "ANDAMAN & NICOBAR": "ANDAMAN_AND_NICOBAR_ISLANDS",
    "ANDHRA PRADESH": "ANDHRA_PRADESH",
    "ARUNANCHAL PRADESH": "ARUNACHAL_PRADESH",
    "ARUNACHAL PRADESH": "ARUNACHAL_PRADESH",
    "ASSAM": "ASSAM",
    "BIHAR": "BIHAR",
    "CHANDIGARH": "CHANDIGARH",
    "CHANDIGARH (UT)": "CHANDIGARH",
    "CHHATTISGARH": "CHHATTISGARH",
    "CHATTISGARH": "CHHATTISGARH",
    "DADRA & NAGAR HAVELLI AND DAMAN & DIU": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
    "DADRA & NAGAR HAVELI AND DAMAN & DIU (UT)": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
    "DADRA & NAGAR HAVELI": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
    "DAMAN & DIU": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
    "DAMAN & DIU (UT)": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
    "DADAR&NAGAR HAVELI": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
    "DADAR & NAGAR HAVELI": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
    "DAMAN": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
    "DIU": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
    "NEW DELHI": "DELHI",
    "PARTS OF MAHARASHTRA": "MAHARASHTRA",
    "EAST RAJASTHAN": "RAJASTHAN",
    "MADRAS": "TAMIL_NADU",
    "DAMAN AND DIU": "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU",
    "DELHI": "DELHI",
    "DELHI (UT)": "DELHI",
    "NCT OF DELHI": "DELHI",
    "DELHI (NCT)": "DELHI",
    "DELHI NCT": "DELHI",
    "GOA": "GOA",
    "KASHMIR": "JAMMU_AND_KASHMIR",
    "BENGAL": "WEST_BENGAL",
    "MEGHĀLAYA": "MEGHALAYA",
    "NĀGĀLAND": "NAGALAND",
    "TAMIL NĀDU": "TAMIL_NADU",
    "KARNĀTAKA": "KARNATAKA",
    "ARUNĀCHAL PRADESH": "ARUNACHAL_PRADESH",
    "GUJARĀT": "GUJARAT",
    "STATE OF ODISHA": "ODISHA",
    "NCT": "DELHI",
    "GUJARAT": "GUJARAT",
    "HARYANA": "HARYANA",
    "HIMACHAL PRADESH": "HIMACHAL_PRADESH",
    "JAMMU & KASHMIR": "JAMMU_AND_KASHMIR",
    "JAMMU & KASHMIR (UT)": "JAMMU_AND_KASHMIR",
    "JAMMU AND KASHMIR": "JAMMU_AND_KASHMIR",
    "JHARKHAND": "JHARKHAND",
    "KARNATAKA": "KARNATAKA",
    "KERALA": "KERALA",
    "LADAKH": "LADAKH",
    "LADAKH (UT)": "LADAKH",
    "LAKSHADWEEP": "LAKSHADWEEP",
    "LAKSHADWEEP (UT)": "LAKSHADWEEP",
    "MADHYA PRADESH": "MADHYA_PRADESH",
    "MAHARASHTRA": "MAHARASHTRA",
    "MANIPUR": "MANIPUR",
    "MEGHALAYA": "MEGHALAYA",
    "MIZORAM": "MIZORAM",
    "NAGALAND": "NAGALAND",
    "ODISHA": "ODISHA",
    "ORISSA": "ODISHA",
    "PUDUCHERRY": "PUDUCHERRY",
    "PUDUCHERRY (UT)": "PUDUCHERRY",
    "PONDICHERRY": "PUDUCHERRY",
    "PUNJAB": "PUNJAB",
    "RAJASTHAN": "RAJASTHAN",
    "SIKKIM": "SIKKIM",
    "TAMIL NADU": "TAMIL_NADU",
    "TAMILNADU": "TAMIL_NADU",
    "TELANGANA": "TELANGANA",
    "TRIPURA": "TRIPURA",
    "UTTAR PRADESH": "UTTAR_PRADESH",
    "UTTARAKHAND": "UTTARAKHAND",
    "UTTARANCHAL": "UTTARAKHAND",
    "WEST BENGAL": "WEST_BENGAL",
}

# 2-letter state codes to canonical state_id
CODE_TO_STATE_ID: dict[str, str] = {v["state_code"]: k for k, v in STATES_MAP.items()}

# Census stcode to canonical state_id
STCODE_TO_STATE_ID: dict[str, str] = {v["stcode"]: k for k, v in STATES_MAP.items()}


def normalize_state_id(raw_name: str | None) -> str | None:
    """Normalize arbitrary raw state string or code to canonical state_id."""
    if not raw_name or not isinstance(raw_name, str):
        return None
    cleaned = raw_name.strip().upper()
    if cleaned in STATES_MAP:
        return cleaned
    if cleaned in CODE_TO_STATE_ID:
        return CODE_TO_STATE_ID[cleaned]
    if cleaned in STCODE_TO_STATE_ID:
        return STCODE_TO_STATE_ID[cleaned]

    # Primary resolution pass using nielytics.geospatial.india_names
    try:
        norm_name = normalize_state_name(raw_name)
        if norm_name:
            candidate = norm_name.strip().upper().replace(" & ", "_AND_").replace(" AND ", "_AND_").replace(" ", "_")
            if candidate in STATES_MAP:
                return candidate
            if candidate in STATE_NAME_ALIASES:
                return STATE_NAME_ALIASES[candidate]
            norm_upper = norm_name.strip().upper()
            if norm_upper in STATES_MAP:
                return norm_upper
            if norm_upper in STATE_NAME_ALIASES:
                return STATE_NAME_ALIASES[norm_upper]
    except Exception:
        pass

    # Fallback alias lookup
    if cleaned in STATE_NAME_ALIASES:
        return STATE_NAME_ALIASES[cleaned]
    # Strip common suffixes like " (UT)" or " STATE"
    simplified = cleaned.replace(" (UT)", "").replace(" STATE", "").strip()
    if simplified in STATE_NAME_ALIASES:
        return STATE_NAME_ALIASES[simplified]
    return None


def normalize_district_id(raw_district: str | None, state: str | None = None) -> str | None:
    """Normalize arbitrary district name using nielytics geospatial resolution."""
    if not raw_district or not isinstance(raw_district, str):
        return None
    try:
        return normalize_district_name(raw_district, state=state)
    except Exception:
        return raw_district.strip()

# IITM Temperature Regions (NEW-TXREGION.TXT)
TEMPERATURE_REGIONS: dict[str, dict[str, str]] = {
    "ALLIN": {
        "region_code": "ALLIN",
        "region_name": "All-India",
        "description": "Entire country geographical average",
    },
    "WHIND": {
        "region_code": "WHIND",
        "region_name": "Western Himalayas",
        "description": "Jammu & Kashmir, Himachal Pradesh, Uttarakhand hills",
    },
    "NWIND": {
        "region_code": "NWIND",
        "region_name": "Northwest India",
        "description": "Punjab, Haryana, Rajasthan, Delhi, Western UP",
    },
    "NCIND": {
        "region_code": "NCIND",
        "region_name": "North Central India",
        "description": "Uttar Pradesh plains, Madhya Pradesh",
    },
    "NEIND": {
        "region_code": "NEIND",
        "region_name": "Northeast India",
        "description": "Assam, Arunachal Pradesh, Meghalaya, Nagaland, Manipur, Mizoram, Tripura",
    },
    "IPIND": {
        "region_code": "IPIND",
        "region_name": "Interior Peninsular India",
        "description": "Maharashtra, Telangana, Karnataka, Andhra Pradesh interior",
    },
    "ECIND": {
        "region_code": "ECIND",
        "region_name": "East Coast India",
        "description": "Odisha, Coastal Andhra Pradesh, Tamil Nadu coast",
    },
    "WCIND": {
        "region_code": "WCIND",
        "region_name": "West Coast India",
        "description": "Coastal Maharashtra, Goa, Coastal Karnataka, Kerala",
    },
}

# Drought Severity Definitions (IIT Gandhinagar Drought Monitor)
DROUGHT_CATEGORIES = {
    "no_drought_pct": {
        "code": "None",
        "label": "No Drought / Normal to Wet",
        "description": "Normal or wetter moisture conditions",
    },
    "d0_d4_pct": {
        "code": "D0-D4",
        "label": "Abnormally Dry or Worse",
        "description": "Percent of national area experiencing D0 or higher drought",
    },
    "d1_d4_pct": {
        "code": "D1-D4",
        "label": "Moderate Drought or Worse",
        "description": "Percent of national area experiencing D1 or higher drought",
    },
    "d2_d4_pct": {
        "code": "D2-D4",
        "label": "Severe Drought or Worse",
        "description": "Percent of national area experiencing D2 or higher drought",
    },
    "d3_d4_pct": {
        "code": "D3-D4",
        "label": "Extreme Drought or Worse",
        "description": "Percent of national area experiencing D3 or higher drought",
    },
    "d4_pct": {
        "code": "D4",
        "label": "Exceptional Drought",
        "description": "Percent of national area experiencing exceptional drought conditions",
    },
}

# IMD Rainfall Departure Categories
RAINFALL_CATEGORIES = {
    "Large Excess": {"code": "LE", "min_departure": 60.0, "max_departure": None, "color": "#1e3a8a"},
    "Excess": {"code": "E", "min_departure": 20.0, "max_departure": 59.0, "color": "#3b82f6"},
    "Normal": {"code": "N", "min_departure": -19.0, "max_departure": 19.0, "color": "#10b981"},
    "Deficient": {"code": "D", "min_departure": -59.0, "max_departure": -20.0, "color": "#f59e0b"},
    "Large Deficient": {"code": "LD", "min_departure": -99.0, "max_departure": -60.0, "color": "#ef4444"},
    "No Rain": {"code": "NR", "min_departure": -100.0, "max_departure": -100.0, "color": "#78716c"},
}

# Provenance and metadata for all 7 raw datasets
DATASET_METADATA: dict[str, dict[str, Any]] = {
    "boundaries": {
        "source_id": "nwic_state_boundary",
        "name": "India State Boundary GeoJSON",
        "raw_file": "state_nwic_geojson.zip",
        "internal_file": "state_NWIC.GeoJSON",
        "provider": "National Water Informatics Centre (NWIC), Ministry of Jal Shakti, Government of India",
        "citation": "National Water Informatics Centre. (2025). State Boundary [Geospatial dataset]. National Water Data Portal, Government of India.",
        "url": "https://nwdp.nwic.gov.in/dataset/dd960900-34cc-4486-a95c-83200d2f2c7b",
        "coverage": "All 36 Indian States and Union Territories",
        "crs": "EPSG:4326",
        "time_period": "2025",
        "limitations": ["Official Survey of India boundaries; topological simplification at 0.01 deg used for web display."],
    },
    "rainfall": {
        "source_id": "imd_rainfall_daily",
        "name": "IMD State-Wise Daily Rainfall",
        "raw_file": "rainfall_statewise_daily_imd.csv",
        "provider": "India Meteorological Department (IMD) / NWIC",
        "citation": "India Meteorological Department. (n.d.). Rainfall Daily IMD - StateWise Daily Rainfall [CSV dataset]. National Water Data Portal, Government of India.",
        "url": "https://nwdp.nwic.gov.in/dataset/rainfall-daily-imd",
        "coverage": "36 Indian States/UTs + Regional summaries + Country aggregate",
        "time_period": "2022-05-31 to 2026-11-01 (75 snapshot dates)",
        "units": "mm",
        "limitations": [
            "Dataset represents snapshot daily observations from NWIC portal, not unbroken multi-decade daily time series.",
            "Missing rainfall actuals are retained as null/NaN; never imputed as 0.0 mm.",
        ],
    },
    "temperature": {
        "source_id": "iitm_regional_temperature",
        "name": "Historical Regional Maximum Temperature",
        "raw_file": "heatwave",
        "canonical_file": "NEW-TXREGION.TXT",
        "provider": "Indian Institute of Tropical Meteorology (IITM Pune) / IMD",
        "citation": "Indian Institute of Tropical Meteorology. (2012). IITM Indian Regional Monthly Surface Air Temperature Dataset: All-India and Regional Maximum Temperatures (1901-2007) [TXT dataset]. IITM Pune.",
        "url": "https://www.tropmet.res.in/data/data-archival/txtn/NEW-TXREGION.TXT",
        "coverage": "All-India and 7 Homogeneous Climatic Regions",
        "time_period": "1901 to 2007 (monthly)",
        "units": "degrees Celsius",
        "limitations": [
            "Monthly mean maximum temperatures are regional climate indicators, NOT daily heatwave event records.",
            "Do NOT present monthly maximums as daily heatwave declarations.",
        ],
    },
    "drought": {
        "source_id": "iitgn_national_drought",
        "name": "India Drought Area Time Series",
        "raw_file": "drought",
        "canonical_file": "India_Drought_Area_Timeseries.txt",
        "provider": "Water and Climate Lab, IIT Gandhinagar (India Drought Monitor)",
        "citation": "Water and Climate Lab, IIT Gandhinagar. (n.d.). India Drought Monitor - National Drought Area Time Series [TXT dataset]. Indian Institute of Technology Gandhinagar.",
        "url": "https://indiadroughtmonitor.in/data/India_Drought_Area_Timeseries.txt",
        "coverage": "National (entire India geographical area)",
        "time_period": "2021-07-14 to 2026-10-28 (277 weekly observations)",
        "units": "percent of national land area (%)",
        "limitations": [
            "Strictly national-level drought percentages; does NOT provide state-level or district-level drought data.",
            "Do NOT manufacture state-level drought observations from this national series.",
        ],
    },
    "flood": {
        "source_id": "iitd_flood_inventory",
        "name": "India Flood Inventory-Impacts (IFI-Impacts)",
        "raw_file": "India_Flood_Inventory_v3.csv",
        "provider": "HydroSense Lab, Indian Institute of Technology Delhi (Saharia et al.)",
        "citation": "Saharia, M. (2025). India Flood Inventory-Impacts (IFI-Impacts), 1967-2023, Version 4 [Dataset]. Zenodo. doi:10.5281/zenodo.16994648",
        "url": "https://zenodo.org/records/16994648",
        "coverage": "India flood events across states and districts",
        "time_period": "1967 to 2023 (6,876 event records)",
        "limitations": [
            "Historical event inventory with casualty and impact data from official reports and media records.",
            "Blank impact/casualty fields are retained as null; zero casualties or damage are never imputed.",
            "Recorded historical events do not represent exhaustive future flood risk.",
        ],
    },
    "landslide": {
        "source_id": "riseuppant_landslide_inventory",
        "name": "India Landslide Dataset",
        "raw_file": "landslide",
        "canonical_file": "landslide_data_india.csv",
        "provider": "riseuppant (GitHub Disaster Management)",
        "citation": "riseuppant. (n.d.). India Landslide Dataset [CSV dataset]. GitHub repository Disaster_Management.",
        "url": "https://raw.githubusercontent.com/riseuppant/Disaster_Management/main/landslide_data_india.csv",
        "coverage": "India landslide incident points (4,500 records)",
        "time_period": "2004 to 2024",
        "limitations": [
            "Community-hosted dataset; original scientific provenance is unverified.",
            "Magnitude scale is uncalibrated and undocumented; used for spatial frequency and density only.",
        ],
    },
    "cyclone": {
        "source_id": "noaa_ibtracs_ni",
        "name": "IBTrACS North Indian Ocean Cyclone Tracks",
        "raw_file": "ibtracs.NI.list.v04r01.csv",
        "provider": "NOAA National Centers for Environmental Information (NCEI)",
        "citation": "Gahtan, J., Knapp, K. R., Schreck, C. J., Diamond, H. J., Kossin, J. P., & Kruk, M. C. (2024). International Best Track Archive for Climate Stewardship (IBTrACS), Version 4r01 [Dataset]. NOAA NCEI.",
        "url": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv",
        "coverage": "North Indian Ocean (Bay of Bengal and Arabian Sea)",
        "time_period": "1842 to 2024",
        "units": "wind in kts and km/h, pressure in hPa/mb, coordinates in WGS84",
        "limitations": [
            "Covers entire North Indian Ocean basin; not every storm tracked made landfall in India.",
            "Track intersections with coastlines do not constitute confirmed official landfall without meteorological verification.",
        ],
    },
}


def resolve_raw_file(filename_or_key: str) -> Path:
    """Resolve a raw data file from search paths, supporting aliases."""
    # Check if key is in DATASET_METADATA
    target_names = [filename_or_key]
    if filename_or_key in DATASET_METADATA:
        meta = DATASET_METADATA[filename_or_key]
        target_names = [
            meta.get("canonical_file"),
            meta.get("raw_file"),
            meta.get("internal_file"),
        ]
        target_names = [t for t in target_names if t]

    for search_dir in RAW_DATA_SEARCH_PATHS:
        for name in target_names:
            p = search_dir / name
            if p.exists() and p.is_file():
                return p
    raise FileNotFoundError(
        f"Could not locate raw dataset '{filename_or_key}'. Searched in: {[str(p) for p in RAW_DATA_SEARCH_PATHS]}"
    )
