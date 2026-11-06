# ClimateLens India: Backend Integration Guide & API Contract

**Role:** Teammate 1 (Data Science, Geospatial Engineering & Analytics Foundation)  
**Target Consumer:** Teammate 2 & 3 (Backend API & Frontend Dashboard Developers)  
**Package:** `climatelens`  
**Core Library:** `nielytics>=0.2.0.dev1` (v2.0)  

---

## 1. Quickstart & Installation

The analytics adapter is pre-installed in the Python 3.13 virtual environment (`/home/prabal/.venvs/nielytics/bin/python`). To import in your FastAPI, Flask, or Django application:

```python
from climatelens import adapter
```

All functions return **standard Python dictionaries** containing **strictly JSON-serializable primitives** (`int`, `float`, `str`, `bool`, `list`, `dict`, or `None`). There are **no raw numpy types, no pandas Timestamps, and no unquoted NaNs**. You can return them directly in FastAPI routes:

```python
from fastapi import FastAPI
from climatelens import adapter

app = FastAPI(title="ClimateLens India API")

@app.get("/api/rainfall/map")
def get_rainfall_map(year: int = 2026):
    return adapter.get_rainfall_map(year=year)

@app.get("/api/compare")
def compare_states(state_a: str, state_b: str, metric: str, year: int = 2026):
    return adapter.compare_states(state_a, state_b, metric, period={"start": year, "end": year})

@app.get("/api/time-machine/{year}")
def get_time_machine(year: int):
    return adapter.get_time_machine_data(year=year)
```

---

## 2. Shared Application Data Contract

Every adapter endpoint complies with the shared response schema:

```json
{
  "status": "ok",
  "data": [],
  "meta": {
    "metric": "rainfall_departure_category",
    "unit": "category / %",
    "geographic_level": "state",
    "time_resolution": "season",
    "period": {"start": "2026-06-01", "end": "2026-09-30"},
    "source_ids": ["imd_rainfall_daily"],
    "method": "Cumulative seasonal departure from normal: ((Actual - Normal) / Normal) * 100",
    "observation_count": 36,
    "limitations": [
      "Represents seasonal cumulative rainfall as of September 30 from IMD NWIC portal."
    ]
  }
}
```

### Error / Unavailable Data Handling
When data is unavailable or when a request violates scientific constraints (e.g. attempting to compare national drought at the state level), the response returns:
```json
{
  "status": "no_data",
  "data": [],
  "meta": {
    "metric": "state_comparison_national_drought",
    "limitations": ["Drought monitor data is exclusively national-level; state-wise disaggregation is scientifically prohibited."]
  }
}
```
**Never substitute `0.0` for missing or unavailable data.** Check `response["status"] == "ok"` before rendering.

---

## 3. Complete Function Reference (15 Methods)

### A. Geospatial & Map Layers

#### `get_available_metadata() -> dict`
- **Purpose:** Full catalog audit of all 7 datasets, file sizes, citations, date spans, and bounding boxes.
- **Sample Return:**
```json
{
  "status": "ok",
  "data": {
    "total_datasets": 7,
    "datasets": { ... }
  },
  "meta": { "metric": "metadata_manifest", ... }
}
```

#### `get_state_boundaries(simplified: bool = True) -> dict`
- **Purpose:** Validated GeoJSON `FeatureCollection` containing all 36 Indian States and Union Territories in EPSG:4326.
- **Parameters:**
  - `simplified` (`bool`, default `True`): Tolerance `0.01` degrees (~760 KB payload for fast web display). Pass `simplified=False` for Survey of India 38 MB high-res polygons.
- **Feature Properties:** `state_id`, `state_name`, `state_code`, `stcode`, `type`, `region`, `source_agency`.

---

### B. Rainfall & Precipitation

#### `get_rainfall_map(year: int = 2026) -> dict`
- **Purpose:** State choropleth departure categories, values, and official IMD color hex codes.
- **Parameters:** `year` (`int`, default `2026`).
- **Data Array Fields:** `state_id`, `state_name`, `state_code`, `monsoon_actual_mm`, `monsoon_normal_mm`, `monsoon_departure_pct`, `monsoon_category`, `color`.

#### `get_state_rainfall_trend(state_id: str, start_year: int = 2022, end_year: int = 2026) -> dict`
- **Purpose:** Daily observations, departure percentages, and cumulative seasonal totals for a state.
- **Parameters:**
  - `state_id` (`str`): Canonical state ID or any common variant (e.g. `'MAHARASHTRA'`, `'Maharashtra'`, `'MH'`, `'27'`).
  - `start_year`, `end_year` (`int`).

---

### C. Drought & Regional Temperature

#### `get_national_drought_trend(start_year: int = 2021, end_year: int = 2026) -> dict`
- **Purpose:** 277 weekly national drought observations from IIT Gandhinagar India Drought Monitor.
- **Data Fields:** `no_drought_pct`, `d0_d4_pct` (abnormally dry+), `d1_d4_pct` (moderate+), `d2_d4_pct` (severe+), `d3_d4_pct` (extreme+), `d4_pct` (exceptional), `rolling_4w_d0_d4_pct`, `weekly_change_d0_d4_pct`.
- **Constraint:** Strictly national resolution.

#### `get_regional_temperature_trend(region: str = "ALLIN", start_year: int = 1901, end_year: int = 2007) -> dict`
- **Purpose:** Long-term maximum temperature warming trends and 1971-2000 climatological anomalies across 8 regions.
- **Valid Regions:** `ALLIN` (All-India), `WHIND` (Western Himalayas), `NWIND` (Northwest), `NCIND` (North Central), `NEIND` (Northeast), `IPIND` (Interior Peninsula), `ECIND` (East Coast), `WCIND` (West Coast).
- **Constraint:** Monthly regional indicators; not daily heatwave declarations.

---

### D. Floods, Landslides & Cyclones

#### `get_flood_events(filters: dict | None = None) -> dict`
- **Filters:** `state_id` (`str`), `start_year` (`int`), `end_year` (`int`), `min_fatalities` (`float`).
- **Data Payload:** Total flood events, verified casualties/displacements, annual frequency, top affected states, and major event table.

#### `get_landslide_events(filters: dict | None = None) -> dict`
- **Filters:** `state_id` (`str`), `year` (`int`).
- **Data Payload:** Total incident count, Western Ghats vs Himalayan hotspot distribution, annual frequency, and up to 500 coordinate points with `reported_magnitude` (uncalibrated scale).

#### `get_cyclone_list(year: int | None = None) -> dict`
- **Filters:** `year` (`int`, optional).
- **Data Payload:** List of North Indian Ocean storms with `storm_id`, `name`, `season`, `start_time`, `end_time`, `max_wmo_wind_kmh`, `min_pressure_hpa`, `min_distance_to_coast_km`.

#### `get_cyclone_track(storm_id: str) -> dict`
- **Parameters:** `storm_id` (`str`, e.g. `'2019116N02090'` or `'FANI'`).
- **Data Payload:** Successive track points with coordinates, `wmo_wind_kmh`, `usa_wind_kmh`, `central_pressure_hpa`, `dist2land_km`, `is_interpolated: False`.

---

### E. Showcase Engines

#### `get_time_machine_data(year: int) -> dict`
- **Purpose:** Full multi-hazard chronological disaster feed for year `year` (e.g. 2019).
- **Payload Structure:**
  - `data.cyclones`: Ordered tracks suitable for animation replay.
  - `data.floods`: Cataloged flood occurrences with start dates and casualties.
  - `data.landslides`: Coordinate incident points.
  - `data.rainfall`: Seasonal state departures.

#### `compare_states(state_a: str, state_b: str, metric: str, period: dict) -> dict`
- **Parameters:**
  - `state_a`, `state_b`: State names or codes (e.g. `'MAHARASHTRA'`, `'GUJARAT'`).
  - `metric`: `'annual_rainfall'`, `'rainfall_anomaly'`, `'flood_event_count'`, `'landslide_event_count'`.
  - `period`: `{"start": int, "end": int}`.
- **Return Payload:** Aligned values, absolute difference, percentage difference (`((val_b - val_a) / val_a) * 100`), higher state indicator, and zero-denominator safety note.
- **Prohibited Guardrails:** Rejects `national_drought` and `regional_temperature` with `status: "no_data"`.

#### `generate_insights(state_id: str | None = None, period: dict | None = None) -> dict`
- **Purpose:** Deterministic, evidence-based insights derived from real measurements. No AI API keys needed.
- **Schema per insight:** `title`, `human_readable`, `value`, `unit`, `period`, `geographic_level`, `source_dataset`, `method`, `observation_count`, `limitations`, `calculation_details`.

#### `investigate_multi_hazard(filters: dict | None = None) -> dict`
- **Purpose:** Explores descriptive co-occurrences (e.g. coastal cyclones with simultaneous inland floods). Clearly notes `causation_established: False`.

#### `get_report_data(state_id: str, period: dict | None = None) -> dict`
- **Purpose:** Consolidated multi-hazard climate risk profile for a single state (rainfall, floods, landslides, and insights).

---

## 4. State ID Normalization Rules

The adapter handles all Indian State / UT variations automatically. Examples:
- `'Arunanchal Pradesh'` -> `'ARUNACHAL_PRADESH'`
- `'Dadra & Nagar Havelli and Daman & Diu'` -> `'DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU'`
- `'Daman & Diu (UT)'` -> `'DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU'`
- `'Orissa'` -> `'ODISHA'`
- `'Pondicherry'` -> `'PUDUCHERRY'`
- `'Uttaranchal'` -> `'UTTARAKHAND'`
- `'DL'` / `'07'` / `'New Delhi'` -> `'DELHI'`
- `'MH'` / `'27'` -> `'MAHARASHTRA'`
