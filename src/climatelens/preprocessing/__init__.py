"""ClimateLens India data preprocessing pipeline modules."""

from climatelens.preprocessing.boundaries import process_boundaries
from climatelens.preprocessing.cyclones import process_cyclones
from climatelens.preprocessing.drought import process_drought
from climatelens.preprocessing.flood import process_flood
from climatelens.preprocessing.landslide import process_landslide
from climatelens.preprocessing.manifest_builder import (
    generate_manifest,
    generate_quality_report,
)
from climatelens.preprocessing.master_excel import build_master_excel
from climatelens.preprocessing.rainfall import process_rainfall
from climatelens.preprocessing.temperature import process_temperature

__all__ = [
    "process_boundaries",
    "process_rainfall",
    "process_temperature",
    "process_drought",
    "process_flood",
    "process_landslide",
    "process_cyclones",
    "generate_manifest",
    "generate_quality_report",
    "build_master_excel",
    "run_pipeline",
]

def __getattr__(name: str):
    if name == "run_pipeline":
        from climatelens.preprocessing.pipeline import run_pipeline
        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
