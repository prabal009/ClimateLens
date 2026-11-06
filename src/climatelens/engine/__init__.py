"""Showcase analytical engines for ClimateLens India."""

from climatelens.engine.comparison import compare_states
from climatelens.engine.insight_engine import generate_insights
from climatelens.engine.multi_hazard import investigate_multi_hazard
from climatelens.engine.time_machine import get_time_machine_data
from climatelens.engine.forecasting import forecast_summer_heatwaves, get_state_cmip6_projections

__all__ = [
    "get_time_machine_data",
    "compare_states",
    "generate_insights",
    "investigate_multi_hazard",
    "forecast_summer_heatwaves",
    "get_state_cmip6_projections",
]
