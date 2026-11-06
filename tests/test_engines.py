"""Unit tests for showcase engines: time machine, comparison, insight, and multi-hazard."""

import pytest

from climatelens.engine.comparison import compare_states
from climatelens.engine.insight_engine import generate_insights
from climatelens.engine.multi_hazard import investigate_multi_hazard
from climatelens.engine.time_machine import get_time_machine_data


def test_time_machine_replay_2019():
    tm = get_time_machine_data(2019)
    assert tm["status"] == "ok"
    assert tm["year"] == 2019
    data = tm["data"]
    assert len(data["cyclones"]) > 0
    assert len(data["floods"]) > 0

    # Verify Fani is in 2019 cyclones
    fani = [c for c in data["cyclones"] if c["name"] == "FANI"]
    assert len(fani) == 1
    assert len(fani[0]["track"]) == 71
    # Real observations check
    assert fani[0]["track"][0]["is_interpolated"] is False


def test_state_comparison_calculations():
    # Valid comparison: Maharashtra vs Gujarat annual rainfall 2026
    res = compare_states("MAHARASHTRA", "GUJARAT", "annual_rainfall", {"start": 2026, "end": 2026})
    assert res["status"] == "ok"
    assert "comparison" in res
    assert "difference_b_minus_a" in res["comparison"]
    assert "percent_difference" in res["comparison"]
    assert isinstance(res["comparison"]["percent_difference"], float)


def test_state_comparison_scientific_guardrails():
    # Comparing national drought or regional temperature at state level is prohibited
    res = compare_states("MAHARASHTRA", "GUJARAT", "national_drought", {"start": 2023, "end": 2023})
    assert res["status"] == "no_data"
    assert "scientific_rationale" in res

    res_temp = compare_states("MAHARASHTRA", "GUJARAT", "regional_temperature", {"start": 2000, "end": 2000})
    assert res_temp["status"] == "no_data"


def test_state_comparison_zero_denominator():
    # Comparison when State A has 0 occurrences (e.g. state with 0 landslides)
    # Lakshadweep has 0 landslides in this dataset
    res = compare_states("LAKSHADWEEP", "UTTARAKHAND", "landslide_event_count", {"start": 2004, "end": 2024})
    assert res["status"] == "ok"
    assert res["comparison"]["percent_difference"] is None
    assert "undefined due to zero baseline" in res["comparison"]["note"].lower()


def test_insight_engine_schema_and_determinism():
    # National insights
    insights = generate_insights()
    assert len(insights) >= 4
    for ins in insights:
        assert "title" in ins
        assert "human_readable" in ins
        assert "value" in ins
        assert "unit" in ins
        assert "period" in ins
        assert "geographic_level" in ins
        assert "source_dataset" in ins
        assert "method" in ins
        assert "observation_count" in ins
        assert "limitations" in ins
        assert "calculation_details" in ins

    # State-specific insights (Kerala)
    kerala_insights = generate_insights(state_id="KERALA")
    assert len(kerala_insights) >= 2
    for ins in kerala_insights:
        assert ins["geographic_level"] == "state"


def test_multi_hazard_explorer():
    res = investigate_multi_hazard({"year": 2019})
    assert res["status"] == "ok"
    assert "associations" in res
    assert "scientific_caveat" in res
    for a in res["associations"]:
        assert a["causation_established"] is False
