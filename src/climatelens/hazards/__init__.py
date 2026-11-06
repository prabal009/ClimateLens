"""Hazard analysis modules for ClimateLens India."""

from climatelens.hazards.cyclone_hazard import analyze_cyclone_hazard, get_storm_track
from climatelens.hazards.drought_hazard import analyze_national_drought
from climatelens.hazards.flood_hazard import analyze_flood_hazard
from climatelens.hazards.heat_hazard import analyze_regional_temperature
from climatelens.hazards.landslide_hazard import analyze_landslide_hazard
from climatelens.hazards.rainfall_hazard import (
    analyze_state_rainfall,
    get_rainfall_choropleth,
    get_state_rainfall_timeseries,
)

__all__ = [
    "analyze_state_rainfall",
    "get_rainfall_choropleth",
    "get_state_rainfall_timeseries",
    "analyze_regional_temperature",
    "analyze_national_drought",
    "analyze_flood_hazard",
    "analyze_landslide_hazard",
    "analyze_cyclone_hazard",
    "get_storm_track",
]
