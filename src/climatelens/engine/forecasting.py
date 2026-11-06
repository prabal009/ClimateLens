"""Climate projections (CMIP6 / IPCC AR6) and operational short-term temperature forecasting."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nielytics.forecasting import ClimateProjectionEngine, TemperatureForecaster

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    normalize_state_id,
)

logger = logging.getLogger(__name__)

SCIENTIFIC_CAVEAT_PROJECTIONS = (
    "Climate projections (CMIP6) and linear trend extrapolations are scenario-based "
    "multi-decadal climate risk indicators, not deterministic operational forecasts."
)

SCIENTIFIC_CAVEAT_FORECAST = (
    "Predictive temperature forecasts and exceedance probabilities are statistical screening proxies, "
    "not official IMD operational heatwave declarations."
)


def _load_state_heatwave_df(state_id: str) -> tuple[str, str, pd.DataFrame]:
    """Retrieve verified historical summer temperature series for a canonical state."""
    sid = normalize_state_id(state_id)
    if not sid or sid not in STATES_MAP:
        raise ValueError(f"Invalid or unrecognized state_id: {state_id!r}")

    sname = STATES_MAP[sid]["state_name"]
    hw_path = PROCESSED_DATA_DIR / "state_heatwave.parquet"
    if not hw_path.exists():
        raise FileNotFoundError(f"State heatwave dataset missing at {hw_path}")

    hw_df = pd.read_parquet(hw_path)
    state_hw = (
        hw_df[hw_df["state_id"] == sid]
        .dropna(subset=["summer_max_temp_deg_c"])
        .sort_values("year")
        .reset_index(drop=True)
    )

    if state_hw.empty:
        return sid, sname, pd.DataFrame()
    return sid, sname, state_hw


def get_state_cmip6_projections(
    state_id: str,
    baseline_temp_c: float | None = None,
    target_years: list[int] | None = None,
) -> dict[str, Any]:
    """Project state mean summer temperatures under CMIP6 SSP scenarios and Theil-Sen trend extrapolation.

    Scenarios evaluated:
    - SSP1-2.6: Low emissions / Paris agreement pathway
    - SSP2-4.5: Intermediate emissions
    - SSP5-8.5: High emissions / fossil-fueled development
    Target years: 2030, 2040, 2050
    """
    sid, sname, state_hw = _load_state_heatwave_df(state_id)
    years = sorted(target_years or [2030, 2040, 2050])
    if state_hw.empty:
        return {
            "status": "no_data",
            "state_id": sid,
            "state_name": sname,
            "message": f"No historical summer temperature observations recorded for {sname} in MoSPI dataset.",
            "scenarios": {},
            "trend_extrapolation": {},
            "scientific_caveat": SCIENTIFIC_CAVEAT_PROJECTIONS,
        }


    # Determine baseline temperature
    if baseline_temp_c is None:
        base_t = float(state_hw["summer_max_temp_deg_c"].mean())
    else:
        base_t = float(baseline_temp_c)

    engine = ClimateProjectionEngine()

    # 1. CMIP6 Projections
    cmip6_res = engine.project_cmip6_scenarios(
        baseline_temperature_c=base_t,
        target_years=years,
        scenarios=["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"],
    )

    # 2. Theil-Sen Historical Trend Extrapolation
    hist_df = pd.DataFrame({
        "date": pd.to_datetime(state_hw["year"].astype(str) + "-05-15"),
        "tmax": state_hw["summer_max_temp_deg_c"].astype(float).values,
    })
    trend_res = engine.project_trend_extrapolation(hist_df, target_years=years)

    return {
        "status": "ok",
        "state_id": sid,
        "state_name": sname,
        "baseline_temperature_c": float(round(base_t, 2)),
        "target_years": years,
        "scenarios": cmip6_res.data.get("scenarios", {}),
        "trend_extrapolation": trend_res.data,
        "metadata": {
            "source": "IPCC AR6 South Asia CMIP6 & Theil-Sen Robust Nonparametric Regression",
            "units": {"temperature": "C"},
        },
        "scientific_caveat": SCIENTIFIC_CAVEAT_PROJECTIONS,
    }


def forecast_summer_heatwaves(
    state_id: str,
    horizon_days: int = 7,
    threshold_c: float = 40.0,
) -> dict[str, Any]:
    """Fit direct multi-step ridge regressor on summer temperature series and estimate daily heatwave risk.

    Provides:
    - 1-to-14 day point forecasts
    - Calibrated 80% and 95% uncertainty prediction intervals
    - Daily heatwave risk exceedance probability and risk classification (Low / Moderate / High)
    """
    sid, sname, state_hw = _load_state_heatwave_df(state_id)
    clamped_horizon = max(1, min(int(horizon_days), 14))
    if state_hw.empty:
        return {
            "status": "no_data",
            "state_id": sid,
            "state_name": sname,
            "message": f"No historical summer temperature observations recorded for {sname} in MoSPI dataset.",
            "predictions": [],
            "heatwave_risk": [],
            "scientific_caveat": SCIENTIFIC_CAVEAT_FORECAST,
        }


    # Construct chronological daily training series from observed summer records
    df_train = pd.DataFrame({
        "date": pd.date_range("2024-04-01", periods=len(state_hw), freq="D"),
        "tmax": state_hw["summer_max_temp_deg_c"].astype(float).values,
    })

    forecaster = TemperatureForecaster(horizons=[1, 3, 7, 14])
    forecaster.fit(df_train, date_column="date", target_column="tmax")

    pred_res = forecaster.predict(horizon=clamped_horizon, interval_levels=[0.80, 0.95])
    risk_res = forecaster.forecast_heatwave_risk(pred_res, threshold_c=threshold_c)

    return {
        "status": "ok",
        "state_id": sid,
        "state_name": sname,
        "horizon_days": clamped_horizon,
        "forecast_origin_date": pred_res.data.get("forecast_origin_date"),
        "predictions": pred_res.data.get("predictions", []),
        "heatwave_risk": risk_res.data.get("daily_risk", []),
        "threshold_c": float(threshold_c),
        "metadata": {
            "model": "DirectMultiStepForecaster (L2 Ridge Regression)",
            "prediction_intervals": ["80%", "95%"],
            "units": {"temperature": "C", "probability": "fraction"},
        },
        "scientific_caveat": SCIENTIFIC_CAVEAT_FORECAST,
    }
