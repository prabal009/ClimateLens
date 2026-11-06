# ClimateLens India: Data Dictionary & Scientific Provenance

**Round:** NIErnaya 2026 Round 2  
**Dataset Count:** 7 Historical Climate Disaster Datasets  
**Processed Format:** Parquet, GeoJSON (EPSG:4326), OpenPyXL XLSX (< 100MB)  

---

## 1. Datasets Overview

| # | Dataset Key | Raw File | Processed Deliverable | Provider / Citation | Coverage | Rows / Records |
|---|---|---|---|---|---|---|
| 1 | `boundaries` | `state_nwic_geojson.zip` | `state_boundaries.geojson`<br>`state_boundaries_simplified.geojson` | National Water Informatics Centre (NWIC), 2025 | All 36 Indian States & UTs | 36 features |
| 2 | `rainfall` | `rainfall_statewise_daily_imd.csv` | `rainfall_daily.parquet`<br>`rainfall_state_year.parquet` | India Meteorological Department (IMD) / NWIC | 36 States + National / Regional | 2,961 daily<br>37 state-year |
| 3 | `temperature`| `heatwave` (`NEW-TXREGION.TXT`) | `regional_temperature.parquet` | Indian Institute of Tropical Meteorology (IITM Pune) | 8 Homogeneous Climatic Regions | 10,272 monthly |
| 4 | `drought` | `drought` (`India_Drought_Area_Timeseries.txt`) | `national_drought.parquet` | Water & Climate Lab, IIT Gandhinagar | National India Land Area | 277 weekly |
| 5 | `flood` | `India_Flood_Inventory_v3.csv` | `flood_events.parquet` | HydroSense Lab, IIT Delhi (Saharia et al., 2025) | Indian States & Districts | 6,876 events |
| 6 | `landslide` | `landslide` (`landslide_data_india.csv`) | `landslide_events.parquet` | riseuppant, GitHub Disaster Management | Indian Incident Points | 4,500 points |
| 7 | `cyclone` | `ibtracs.NI.list.v04r01.csv` | `cyclone_track_points.parquet` | NOAA NCEI IBTrACS v04r01 (Gahtan et al., 2024) | North Indian Ocean (NI) | 57,880 points<br>(1,860 storms) |

---

## 2. Processed Files Schemas & Field Definitions

### A. `state_boundaries.geojson` & `state_boundaries_simplified.geojson`
- **CRS:** `EPSG:4326` (WGS84, coordinates in `[longitude, latitude]` order).
- **Properties:**
  - `state_id` (`String`): Canonical uppercase state identifier (e.g. `'MAHARASHTRA'`).
  - `state_name` (`String`): Official display name (e.g. `'Maharashtra'`).
  - `state_code` (`String`): 2-letter ISO / NWIC code (e.g. `'MH'`).
  - `stcode` (`String`): 2-digit census code (e.g. `'27'`).
  - `type` (`String`): `'State'` or `'Union Territory'`.
  - `region` (`String`): `'North'`, `'South'`, `'East'`, `'West'`, `'Central'`, `'Northeast'`, `'Islands'`.
  - `source_agency` (`String`): Survey of India (SOI).

---

### B. `rainfall_daily.parquet` & `rainfall_state_year.parquet`
- **Daily Table:**
  - `date` (`datetime64[ns]`): Date of observation (`YYYY-MM-DD`).
  - `entity_id` (`String`): State ID, `INDIA_NATIONAL`, or `REGION_*`.
  - `state_id` (`String | None`): Canonical state ID if geographic_level is state.
  - `entity_name` (`String`): Clean display name.
  - `geographic_level` (`String`): `'state'`, `'regional'`, or `'national'`.
  - `daily_actual_mm` (`Float64`): Observed daily rainfall in mm (retains `NaN` for missing observations; no zero imputation).
  - `daily_normal_mm` (`Float64`): IMD climatological normal daily rainfall in mm.
  - `daily_departure_pct` (`Float64`): Departure percentage `((Actual - Normal) / Normal) * 100`.
  - `daily_category` (`String`): IMD category (`Large Excess`, `Excess`, `Normal`, `Deficient`, `Large Deficient`, `No Rain`).
  - `cumulative_actual_mm`, `cumulative_normal_mm`, `cumulative_departure_pct`, `cumulative_category`: Seasonal cumulative rainfall metrics.

- **State-Year Table:**
  - `state_id` (`String`), `state_name` (`String`), `year` (`Int64`).
  - `monsoon_cumulative_actual_mm` (`Float64`): Cumulative southwest monsoon rainfall as of September 30.
  - `monsoon_cumulative_normal_mm` (`Float64`): Climatological southwest monsoon normal baseline.
  - `monsoon_departure_pct` (`Float64`): Departure from normal baseline.
  - `monsoon_category` (`String`): Seasonal category code (`LE`, `E`, `N`, `D`, `LD`, `NR`).
  - `observed_days_count` (`Int64`): Count of available observation snapshot days.
  - `heavy_rain_days_count` (`Int64`): Days where actual rainfall exceeded 64.5 mm (IMD threshold).
  - `wet_days_count` (`Int64`): Days where actual rainfall exceeded 2.5 mm.

---

### C. `regional_temperature.parquet`
- **Columns:**
  - `date` (`datetime64[ns]`): First day of month (`YYYY-MM-01`).
  - `year` (`Int64`), `month` (`Int64`).
  - `region_code` (`String`): One of 8 IITM regions (`ALLIN`, `WHIND`, `NWIND`, `NCIND`, `NEIND`, `IPIND`, `ECIND`, `WCIND`).
  - `region_name` (`String`): Full region name (e.g. `'Western Himalayas'`).
  - `tmax_celsius` (`Float64`): Monthly mean maximum surface air temperature in °C.
  - `baseline_normal_celsius` (`Float64`): 1971-2000 WMO standard 30-year climatological normal mean for that (region, month).
  - `monthly_anomaly_celsius` (`Float64`): `tmax_celsius - baseline_normal_celsius`.
  - `rolling_12m_anomaly_celsius` (`Float64`): 12-month rolling mean temperature anomaly in °C.
  - `rolling_60m_anomaly_celsius` (`Float64`): 60-month (5-year) rolling mean temperature anomaly in °C.
  - `caveat` (`String`): Explicit warning that regional monthly means represent climate warming trends, not daily heatwave declarations.

---

### D. `national_drought.parquet`
- **Columns:**
  - `date` (`datetime64[ns]`): Date of weekly report (`YYYY-MM-DD`).
  - `year` (`Int64`), `month` (`Int64`), `day` (`Int64`).
  - `no_drought_pct` (`Float64`): Percent of national area under normal or wet moisture conditions.
  - `d0_d4_pct` (`Float64`): Percent of national area experiencing abnormally dry or worse conditions.
  - `d1_d4_pct` (`Float64`): Moderate drought or worse area percentage.
  - `d2_d4_pct` (`Float64`): Severe drought or worse area percentage.
  - `d3_d4_pct` (`Float64`): Extreme drought or worse area percentage.
  - `d4_pct` (`Float64`): Exceptional drought area percentage.
  - `rolling_4w_d0_d4_pct` (`Float64`): 4-week moving average of D0-D4 area percentage.
  - `weekly_change_d0_d4_pct` (`Float64`): Week-over-week difference in D0-D4 area percentage.
- **Invariants:**
  - `no_drought_pct + d0_d4_pct == 100.0%` within `0.05%` rounding tolerance.
  - `0 <= d4 <= d3_d4 <= d2_d4 <= d1_d4 <= d0_d4 <= 100.0%`.
  - Strictly national resolution.

---

### E. `flood_events.parquet`
- **Columns:**
  - `event_id` (`String`): Universal Event Identifier (UEI, e.g. `'UEI-IMD-FL-1967-0001'`).
  - `year` (`Float64`): Year of event start (1967-2023).
  - `start_date` (`String | None`): ISO date `YYYY-MM-DD`.
  - `end_date` (`String | None`): ISO date `YYYY-MM-DD`.
  - `duration_days` (`Float64`): Duration of flood event in days.
  - `primary_state_id` (`String | None`): Canonical primary state ID.
  - `primary_state_name` (`String | None`): Official state name.
  - `districts` (`String | None`): Reported affected districts.
  - `human_fatality` (`Float64 | None`): Verified fatalities (blank in source retained as `None`; no zero-imputation).
  - `human_injured` (`Float64 | None`): Reported injuries.
  - `human_displaced` (`Float64 | None`): Reported population displacements.
  - `main_cause` (`String | None`): Meteorological or hydrologic trigger.
  - `extent_of_damage` (`String | None`): Qualitative impact notes.
  - `latitude`, `longitude`: Retained as null (coordinates absent in original source).

---

### F. `landslide_events.parquet`
- **Columns:**
  - `event_id` (`String`): Unique identifier (e.g. `'LS-2006-0000'`).
  - `year` (`Int64`): Year of incident (2004-2024).
  - `state_id` (`String`): Authoritative state ID assigned via spatial join against NWIC boundary polygon.
  - `state_name` (`String`): Official state name.
  - `location` (`String`): Community-reported location text.
  - `latitude` (`Float64`): WGS84 latitude degrees north [10.85, 34.15].
  - `longitude` (`Float64`): WGS84 longitude degrees east [73.85, 94.56].
  - `reported_magnitude` (`Float64`): Raw reported score (1.0 to 10.0, uncalibrated scale).
  - `caveat` (`String`): Note stating community provenance and uncalibrated magnitude.

---

### G. `cyclone_track_points.parquet`
- **Columns:**
  - `sid` (`String`): IBTrACS unique storm identifier (e.g. `'2019116N02090'`).
  - `season` (`Int64`): Year of cyclone season (1842-2026).
  - `name` (`String`): Official storm name (e.g. `'FANI'`, `'AMPHAN'`, or `'UNNAMED'`).
  - `iso_time` (`String`): UTC timestamp `YYYY-MM-DD HH:MM:SS`.
  - `basin` (`String`): `'NI'` (North Indian Ocean).
  - `subbasin` (`String | None`): `'BB'` (Bay of Bengal) or `'AS'` (Arabian Sea).
  - `latitude`, `longitude` (`Float64`): WGS84 coordinates.
  - `wmo_wind_kts` (`Float64 | None`): WMO agency 10-minute maximum sustained wind in knots.
  - `wmo_wind_kmh` (`Float64 | None`): WMO sustained wind converted to km/h (`kts * 1.852`).
  - `usa_wind_kts` (`Float64 | None`): JTWC 1-minute sustained wind in knots.
  - `usa_wind_kmh` (`Float64 | None`): JTWC wind converted to km/h (`kts * 1.852`).
  - `wmo_pres_hpa`, `usa_pres_hpa` (`Float64 | None`): Minimum central atmospheric pressure in hPa.
  - `dist2land_km` (`Float64 | None`): Distance to nearest land in km.
  - `landfall` (`Int64 | None`): IBTrACS landfall proximity indicator.
  - `is_interpolated` (`Bool`): Strictly `False` for all real observations.

---

## 3. Scientific Policies & Integrity Guardrails

1. **Zero Data Fabrication:** Missing measurements are retained as `NaN` / `None`. Under no circumstances are missing rainfall days imputed as `0.0 mm`, nor are unrecorded flood casualties marked as `0`.
2. **Geographical Resolution Preservation:** The IIT Gandhinagar drought time series is exclusively national; the IITM temperature data is published across 8 homogeneous climatic regions. The adapter will reject any attempt to compare or plot these datasets at the state level with `status: "no_data"`.
3. **Cyclone Landfall Clarification:** Track proximity to the coastline does not constitute confirmed official IMD landfall without meteorological verification.
4. **Landslide Magnitude Disclaimer:** The community-sourced landslide magnitude scale is uncalibrated and is used solely for relative spatial density and event frequency counts.
