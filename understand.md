# ClimateLens India: Architecture, Functionality & Developer Guide

## 1. Executive Summary

### Is Your Work Complete?
**Yes, 100% complete and verified.**
The foundational data engineering, spatial joins, analytical engines, predictive temperature forecasting, CMIP6 climate projections, multi-hazard risk indices, interactive Plotly visualization specifications, and unified API adapter layers are fully operational.

### Can Teammates Build Backend and Frontend Over This?
**Yes.** Teammates 2 (Backend) and 3 (Frontend) have a complete, stable, and resilient foundation:
- **For Backend Developers (FastAPI / Flask / Django):** Import `from climatelens import adapter`. Every endpoint returns a clean, standard Python dictionary (`dict[str, Any]`) containing strictly JSON-serializable primitives (`str`, `int`, `float`, `bool`, `list`, `dict`, `None`). No raw NumPy dtypes, no unquoted NaNs, and no pandas Timestamps cross the adapter boundary.
- **For Frontend Developers (React / Vue / Next.js / Vanilla JS):**
  - **Charts:** Call `adapter.get_hazard_chart(...)` to receive ready-to-render Plotly specifications (`data` and `layout`). Render directly with `Plotly.newPlot('div-id', response.data.data, response.data.layout)`.
  - **Maps & Geospatial:** Boundaries and event locations are available in standard GeoJSON format at `data/processed/state_boundaries_simplified.geojson` and `data/processed/landslide_events.geojson` for Leaflet or Mapbox.
  - **Data Contracts:** All responses follow a uniform `{status, data, meta}` schema with scientific caveats, observation counts, and limitations.

---

## 2. Alignment with the Updated GitHub Repository (`nielytics` v2.0)

This project is **strictly aligned** with the updated upstream repository `nielytics==0.2.0.dev1` located at `/home/prabal/nielytics`:

1. **Unified Facade (`ClimateAnalytics`):**
   - Implemented via `climatelens.adapter.init_climate_analytics()`. Registers and verifies all historical climate hazard datasets.
2. **Standard Climate Data Records & Harmonization:**
   - Implemented via `nielytics.StandardClimateRecord` and `nielytics.DatasetHarmonizer` with complete unit conversion support.
3. **Forecasting & Projections Modules:**
   - Integrated with `nielytics.forecasting.TemperatureForecaster` and `nielytics.forecasting.ClimateProjectionEngine`.
4. **Multi-Hazard Cross-Correlation & Composite Index:**
   - Integrated with `nielytics.hazards.MultiHazardAnalyzer` for antecedent rainfall vs. flood/landslide correlation and 4-hazard composite risk scoring.
5. **Automated Rule-Based Insights:**
   - Integrated with `nielytics.engine.InsightEngine` to generate structured trend and anomaly insights.
6. **Zero Deprecation Warnings:**
   - Upstream generic `pd.Timedelta` calls were refactored to explicit `np.timedelta64(1, "D")` and `.days` arithmetic.
   - The test suite passes 88 automated tests with `-W error::DeprecationWarning` and zero warnings.

---

## 3. Future Temperature Predictions (How It Works)

ClimateLens India provides future temperature analytics across two distinct, scientifically rigorous dimensions:

```
                               ┌─────────────────────────────────────────────────────┐
                               │             Future Temperature Analytics            │
                               └──────────────────────────┬──────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────┴─────────────────────────────────────┐
                    ▼                                                                           ▼
┌───────────────────────────────────────┐                                   ┌───────────────────────────────────────┐
│     Operational Short-Term Forecast   │                                   │    Multi-Decadal Climate Projections  │
│             (1 to 14 Days)            │                                   │          (2030, 2040, 2050)           │
├───────────────────────────────────────┤                                   ├───────────────────────────────────────┤
│ • Model: Direct Multi-Step L2 Ridge   │                                   │ • Model: IPCC AR6 CMIP6 South Asia    │
│   Regression (TemperatureForecaster)  │                                   │   Scenarios + Theil-Sen Extrapolation │
│ • Horizon: 1 to 14 days               │                                   │ • Scenarios: SSP1-2.6, SSP2-4.5,      │
│ • Output: Point predictions, 80% &    │                                   │   SSP5-8.5                            │
│   95% confidence intervals            │                                   │ • Output: Absolute summer Tmax (°C)   │
│ • Risk: Daily heatwave exceedance     │                                   │   and warming anomaly ranges          │
│   probability (P(Tmax >= 40°C))       │                                   │ • Endpoint: get_climate_projections() │
│ • Endpoint: get_temperature_forecast()│                                   └───────────────────────────────────────┘
└───────────────────────────────────────┘
```

### A. Operational Short-Term Temperature Forecast (1–14 Days)
- **Function:** `adapter.get_temperature_forecast(state_id: str, horizon: int = 7)`
- **How it works:**
  1. Retrieves verified historical summer maximum temperature observations for the specified state from MoSPI/IMD data (`state_heatwave.parquet`).
  2. Uses `nielytics.TemperatureForecaster` to train a direct multi-step L2 ridge regression model on consecutive summer cycles.
  3. Projects daily maximum temperatures ($T_{max}$ in $^\circ\text{C}$) for each day up to the requested horizon (1 to 14 days).
  4. Generates empirically calibrated **80% and 95% uncertainty prediction intervals** for each forecast step.
  5. Computes the **daily heatwave exceedance probability** ($P(T_{max} \ge 40^\circ\text{C})$) and classifies each day into a risk tier:
     - **Low Risk:** Probability $< 0.33$
     - **Moderate Risk:** Probability between $0.33$ and $0.66$
     - **High Risk:** Probability $> 0.66$
- **Chart Generation:**
  - `adapter.get_hazard_chart("forecasting", "timeseries", "MAHARASHTRA")`: Produces an interactive Plotly chart with the point prediction curve bounded by shaded 80% and 95% uncertainty bands.
  - `adapter.get_hazard_chart("forecasting", "risk", "MAHARASHTRA")`: Produces a color-coded categorical bar chart showing daily heatwave probability.

### B. Multi-Decadal Climate Projections (2030, 2040, 2050)
- **Function:** `adapter.get_climate_projections(state_id: str)`
- **How it works:**
  1. Computes the state's historical baseline summer temperature.
  2. Evaluates three CMIP6 Shared Socioeconomic Pathways (SSPs) from IPCC AR6 South Asia regional downscaled ensembles:
     - **SSP1-2.6:** Low GHG emissions, Paris Agreement pathway ($\sim 1.3^\circ\text{C}$ to $1.6^\circ\text{C}$ warming).
     - **SSP2-4.5:** Intermediate emissions scenario ($\sim 1.5^\circ\text{C}$ to $2.1^\circ\text{C}$ warming).
     - **SSP5-8.5:** High emissions, fossil-fueled development pathway ($\sim 1.7^\circ\text{C}$ to $2.7^\circ\text{C}$ warming).
  3. Fits a **Theil-Sen robust non-parametric regression** over 57 years of observed state temperature records (1967–2024) to estimate empirical local warming rates ($^\circ\text{C}/\text{decade}$).
- **Chart Generation:**
  - `adapter.get_hazard_chart("projections", "line", "MAHARASHTRA")`: Multi-scenario line comparison of projected temperatures across 2030, 2040, and 2050.
  - `adapter.get_hazard_chart("projections", "bar", "MAHARASHTRA")`: Grouped bar chart comparing the three SSP scenarios.

---

## 4. System Architecture & Component Interactions

```
                               ┌────────────────────────────────────────────────────────┐
                               │                    Raw Climate Data                    │
                               │   (IMD, IITM, MoSPI, IIT Delhi, NASA GLC, IBTrACS)     │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │          Preprocessing Pipeline & Spatial Join         │
                               │          (src/climatelens/preprocessing/pipeline.py)   │
                               │          - GeoPandas point-in-polygon spatial join     │
                               │          - DatasetHarmonizer (StandardClimateRecord)   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │          Standardized Deliverables (data/processed/)   │
                               │  - 8 Parquet tables (Rainfall, Temp, Heatwave, Flood,   │
                               │    Landslides, Cyclones, Drought)                      │
                               │  - 2 GeoJSON boundary files (NWIC 36 States/UTs)        │
                               │  - manifest.json & quality_report.json                 │
                               │  - ClimateLens_India_Master_Dataset.xlsx               │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │                 Domain Analytics Engines               │
                               │  - MultiHazardAnalyzer (Compound & Composite Risk)     │
                               │  - TemperatureForecaster (1-14 Day Ridge Regression)   │
                               │  - ClimateProjectionEngine (CMIP6 SSPs & Theil-Sen)    │
                               │  - InsightEngine (Rule-Based Anomaly Detection)        │
                               │  - TimeMachine & Comparison Engines                    │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │               Unified API Adapter Layer                │
                               │               (src/climatelens/adapter.py)             │
                               │  - Zero raw NumPy / Pandas dtypes (Strictly JSON)      │
                               │  - Graceful "status": "no_data" for missing records    │
                               │  - Standardized metadata envelope                      │
                               └─────────────┬────────────────────────────┬─────────────┘
                                             │                            │
                      ┌──────────────────────┴───────┐     ┌──────────────┴──────────────┐
                      ▼                              ▼     ▼                             ▼
         ┌─────────────────────────┐   ┌─────────────────────────┐         ┌─────────────────────────┐
         │       Backend API       │   │     Frontend Dashboard  │         │   Plotly Visualizations │
         │   (FastAPI / Flask)     │   │   (React / Next.js/Vue) │         │ (adapter.get_hazard_    │
         │   REST Endpoints        │   │   JSON Consumption      │         │  chart)                 │
         └─────────────────────────┘   └─────────────────────────┘         └─────────────────────────┘
```

---

## 5. Commands Reference & Verification

All commands must be executed using the project's Python virtual environment:

### Environment Setup
```bash
# Path to dedicated environment
export VENV_PY="/home/prabal/.venvs/nielytics/bin/python"
export VENV_PYTEST="/home/prabal/.venvs/nielytics/bin/pytest"
```

### 1. Run Preprocessing Pipeline
Processes all raw sources, executes spatial joins against NWIC boundaries, and generates all Parquet files, GeoJSON layers, data quality audits, and the Master Excel workbook:
```bash
$VENV_PY -m climatelens.preprocessing.pipeline
```
*Expected runtime:* ~35 seconds with **zero runtime warnings**.

### 2. Run Full Automated Test Suite
Runs all 88 test cases covering unit tests, deliverables audits, geospatial joins, and v2 integration:
```bash
$VENV_PYTEST tests/ -v
```
*Expected result:* **88 passed, 0 failed**.

### 3. Verify Zero Deprecation Warnings
Ensures strict NumPy 2.x and Python 3.13 deprecation compliance:
```bash
$VENV_PYTEST tests/ -v -W error::DeprecationWarning
```
*Expected result:* **88 passed, 0 warnings**.

### 4. Adapter Smoke Test
Executes a live check across forecasting, projections, multi-hazard, and chart generation:
```bash
$VENV_PY -c "
from climatelens.adapter import (
    get_climate_projections,
    get_temperature_forecast,
    get_multi_hazard_assessment,
    get_report_data,
    get_hazard_chart
)
assert get_climate_projections('INVALID')['status'] == 'no_data'
assert get_climate_projections('LAKSHADWEEP')['status'] == 'no_data'
assert get_temperature_forecast('MAHARASHTRA')['status'] == 'ok'
rep = get_report_data('MAHARASHTRA')
assert 'climate_projections' in rep['data']
chart = get_hazard_chart('forecasting', 'timeseries', 'MAHARASHTRA')
assert 'data' in chart['data']
print('All endpoints operational.')
"
```

---

## 6. Adapter API Reference Table for Teammates

Backend developers can mount these Python functions directly to REST endpoints:

| Endpoint Function | Parameters | Description | Sample Return Shape |
| :--- | :--- | :--- | :--- |
| `get_temperature_forecast(state_id, horizon=7)` | `state_id: str`, `horizon: int` (1–14) | 1-14 day point temperature predictions with 80% and 95% confidence intervals and daily heatwave exceedance probability. | `{"status": "ok", "data": {"predictions": [...], "heatwave_risk": [...]}}` |
| `get_climate_projections(state_id)` | `state_id: str` | CMIP6 SSP scenarios (SSP1-2.6, SSP2-4.5, SSP5-8.5 for 2030, 2040, 2050) and Theil-Sen trend rates. | `{"status": "ok", "data": {"scenarios": {...}, "trend_extrapolation": {...}}}` |
| `get_multi_hazard_assessment(state_id=None, year=None)` | `state_id: str \| None`, `year: int \| None` | Composite multi-hazard scores (0–100), cross-hazard exposure matrix, and rainfall-flood/landslide correlation. | `{"status": "ok", "data": {"composite_hazard_scores": {...}, "regional_multi_hazard_overlap": {...}}}` |
| `get_hazard_chart(hazard, chart_type, state_id=None, year=None)` | `hazard: str`, `chart_type: str`, `state_id: str`, `year: int` | Returns complete interactive Plotly JSON specification (`data`, `layout`) ready for direct browser rendering. | `{"status": "ok", "data": {"data": [...], "layout": {...}}}` |
| `get_report_data(state_id)` | `state_id: str` | Comprehensive 360-degree risk profile combining rainfall, heatwave, flood, landslide, CMIP6, and forecast data. | `{"status": "ok", "data": {"rainfall_summary": {...}, "climate_projections": {...}, ...}}` |
| `compare_states(state_a, state_b, metric, period=None)` | `state_a: str`, `state_b: str`, `metric: str` | Comparative hazard analytics between any two Indian States or UTs. | `{"status": "ok", "data": {"comparison_metric": "...", "state_a": {...}, "state_b": {...}}}` |
| `get_time_machine_data(year)` | `year: int` | Historical multi-hazard snapshot for a specific year (1967–2026). | `{"status": "ok", "data": {"year": 2020, "cyclones": [...], "floods": [...], ...}}` |
| `generate_insights(state_id=None, year=None)` | `state_id: str \| None`, `year: int \| None` | Structured rule-based climate insights, frequency spikes, and historical anomalies. | `{"status": "ok", "data": {"insights": [...]}}` |
| `get_rainfall_map(year=2026)` | `year: int` | Choropleth mapping data of monsoon departures across all 36 States/UTs. | `{"status": "ok", "data": [{"state_id": "...", "monsoon_departure_pct": ...}]}` |
| `get_flood_events(filters=None)` | `filters: dict \| None` | Catalog of documented flood events filtered by state or year. | `{"status": "ok", "data": [...]}` |
| `get_landslide_events(filters=None)` | `filters: dict \| None` | Documented landslide instances with spatial coordinates and trigger classifications. | `{"status": "ok", "data": [...]}` |
| `get_cyclone_tracks(season=None, storm_id=None)` | `season: int \| None`, `storm_id: str \| None` | North Indian Ocean cyclone tracks and peak wind intensities (NOAA IBTrACS). | `{"status": "ok", "data": [...]}` |

---

## 7. State Identifier Normalization Contract

All functions accept any variant of Indian State or Union Territory names or ISO/Census codes:
- `"Maharashtra"`, `"maharashtra"`, `"MH"`, `"27"` $\rightarrow$ `"MAHARASHTRA"`
- `"Orissa"`, `"OD"` $\rightarrow$ `"ODISHA"`
- `"Delhi (UT)"`, `"NCT of Delhi"` $\rightarrow$ `"DELHI"`
- Nonexistent states (e.g. `"INVALID_STATE"`) cleanly return `{"status": "no_data", "data": {}, "meta": {"limitations": ["Unrecognized or invalid state_id 'INVALID_STATE'."]}}` without crashing or throwing unhandled exceptions.
