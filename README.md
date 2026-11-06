# ClimateLens India

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-88%20passed-brightgreen.svg)]()
[![Zero Warnings](https://img.shields.io/badge/warnings-0%20deprecation-success.svg)]()
[![Powered by NIElytics v2](https://img.shields.io/badge/powered%20by-NIElytics%20v2.0-orange.svg)](https://github.com/prabal009/nielytics)

**ClimateLens India** is an analytical engine, geospatial pipeline, and climate risk platform covering all 36 Indian States and Union Territories across 6 historical hazard domains (1967–2026).

Built as the data science and climate analytics foundation for **NIErnaya 2026 Round 2**, ClimateLens India integrates deeply with upstream **`nielytics` v2.0** to provide standardized climate data records, operational temperature forecasting, multi-decadal CMIP6 climate projections, multi-hazard composite risk scoring, and interactive Plotly visual specifications.

---

## Key Features

1. **Operational Short-Term Temperature Forecasting (1 to 14 Days):**
   - Direct multi-step L2 Ridge Regression with calibrated 80% and 95% uncertainty prediction intervals.
   - Daily heatwave exceedance probability ($P(T_{max} \ge 40^\circ\text{C})$) categorized into Low, Moderate, and High risk.
2. **Multi-Decadal Climate Projections (2030, 2040, 2050):**
   - IPCC AR6 South Asia CMIP6 regional downscaled projections across **SSP1-2.6**, **SSP2-4.5**, and **SSP5-8.5**.
   - Historical Theil-Sen robust non-parametric trend extrapolation over 57 years of recorded summer observations (1967–2024).
3. **Cross-Hazard Correlation & Composite Multi-Hazard Index:**
   - Antecedent rainfall correlation against documented flood and landslide event catalogs.
   - 4-hazard composite risk index (heatwave intensity, flood frequency, landslide density, rainfall volume) across all 36 States/UTs.
4. **Automated Rule-Based Climate Insights:**
   - Detects frequency spikes, compound heat-drought episodes, and decade-over-decade anomalies via `nielytics.InsightEngine`.
5. **Geospatial NWIC Alignment (EPSG:4326):**
   - Point-in-polygon spatial indexing against official National Water Informatics Centre (NWIC) state boundaries.
6. **Unified Web Adapter & Chart Engine:**
   - Zero-dependency JSON responses (`{"status": "ok", "data": {...}, "meta": {...}}`) with no raw NumPy types, unquoted NaNs, or pandas Timestamps.
   - Ready-to-render Plotly chart specifications (`adapter.get_hazard_chart`).

---

## Repository Structure

```
ClimateLens/
├── data/
│   ├── processed/                # Production deliverables
│   │   ├── *.parquet             # 8 analysis-ready Parquet tables
│   │   ├── *.geojson             # NWIC state boundaries (full + simplified) & landslides
│   │   ├── manifest.json         # Dataset provenance & SHA256 hashes
│   │   ├── quality_report.json   # 7-dataset quality audit report
│   │   └── ClimateLens_India_Master_Dataset.xlsx  # 12-sheet official master workbook
│   └── raw/                      # Authoritative raw source archives
├── docs/
│   ├── BACKEND_INTEGRATION.md    # API contract for Backend & Frontend teammates
│   └── DATA_DICTIONARY.md        # Comprehensive metadata dictionary
├── src/
│   └── climatelens/
│       ├── adapter.py            # Unified backend API bridge (12 endpoints)
│       ├── constants.py          # 36 canonical States/UTs metadata & aliases
│       ├── engine/               # Forecasting, CMIP6, multi-hazard, insights
│       ├── geospatial/           # Point-in-polygon spatial join & boundary index
│       ├── hazards/              # Hazard modules & Plotly chart builders
│       └── preprocessing/        # End-to-end data pipelines & master Excel builder
├── tests/                        # 88 automated unit, deliverable & v2 tests
├── understand.md                 # Complete architectural & operational guide
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/prabal009/ClimateLens.git
cd ClimateLens

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run Preprocessing Pipeline

Transforms all raw datasets, performs geospatial joins, and builds the Master Excel workbook:

```bash
python -m climatelens.preprocessing.pipeline
```
*(Executes in ~35 seconds with 0 runtime warnings)*

### 3. Run Automated Tests

```bash
# Run full suite (88 tests)
pytest tests/ -v

# Verify strict zero-deprecation-warning compliance
pytest tests/ -v -W error::DeprecationWarning
```

---

## API Adapter Usage (for Backend & Frontend Developers)

Backend frameworks (FastAPI, Flask, Django) can directly import and serve `climatelens.adapter`:

```python
from climatelens import adapter

# 1. 7-Day Operational Temperature Forecast
forecast = adapter.get_temperature_forecast("MAHARASHTRA", horizon=7)

# 2. CMIP6 Climate Projections (2030, 2040, 2050)
projections = adapter.get_climate_projections("MAHARASHTRA")

# 3. Multi-Hazard Composite Risk Assessment
risk = adapter.get_multi_hazard_assessment("MAHARASHTRA")

# 4. Ready-to-render Plotly Chart Spec
chart = adapter.get_hazard_chart(hazard="forecasting", chart_type="timeseries", state_id="MAHARASHTRA")
# chart["data"] contains Plotly {"data": [...], "layout": {...}} ready for Plotly.newPlot()

# 5. Consolidated State Risk Profile
report = adapter.get_report_data("MAHARASHTRA")
```

For detailed API schemas, parameter types, and endpoint tables, see **[understand.md](understand.md)** and **[docs/BACKEND_INTEGRATION.md](docs/BACKEND_INTEGRATION.md)**.

---

## License

Developed for **NIErnaya 2026 Round 2**. Powered by **NIElytics v2.0** by Prabal Pathak.
