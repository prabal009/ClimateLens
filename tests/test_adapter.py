"""Unit tests for backend adapter contract compliance and JSON serializability."""

import json
import pytest

from climatelens import adapter


ADAPTER_CALLS = [
    ("get_available_metadata", lambda: adapter.get_available_metadata()),
    ("get_state_boundaries", lambda: adapter.get_state_boundaries(simplified=True)),
    ("get_rainfall_map", lambda: adapter.get_rainfall_map(2026)),
    ("get_state_rainfall_trend", lambda: adapter.get_state_rainfall_trend("MAHARASHTRA")),
    ("get_national_drought_trend", lambda: adapter.get_national_drought_trend(2021, 2026)),
    ("get_regional_temperature_trend", lambda: adapter.get_regional_temperature_trend("ALLIN", 1901, 2007)),
    ("get_flood_events", lambda: adapter.get_flood_events({"state_id": "KERALA"})),
    ("get_landslide_events", lambda: adapter.get_landslide_events({"state_id": "UTTARAKHAND"})),
    ("get_cyclone_list", lambda: adapter.get_cyclone_list(2019)),
    ("get_cyclone_track", lambda: adapter.get_cyclone_track("FANI")),
    ("get_time_machine_data", lambda: adapter.get_time_machine_data(2019)),
    ("compare_states", lambda: adapter.compare_states("MAHARASHTRA", "GUJARAT", "annual_rainfall", {"start": 2026, "end": 2026})),
    ("generate_insights", lambda: adapter.generate_insights(state_id="KERALA")),
    ("investigate_multi_hazard", lambda: adapter.investigate_multi_hazard({"year": 2019})),
    ("get_report_data", lambda: adapter.get_report_data("ODISHA")),
]


@pytest.mark.parametrize("name,fn", ADAPTER_CALLS)
def test_adapter_contract_schema_and_json_safety(name, fn):
    res = fn()
    assert isinstance(res, dict), f"{name} did not return a dict"
    assert "status" in res, f"{name} missing 'status'"
    assert res["status"] in ("ok", "no_data"), f"{name} returned unexpected status '{res['status']}'"
    assert "data" in res, f"{name} missing 'data'"
    assert "meta" in res, f"{name} missing 'meta'"

    meta = res["meta"]
    assert "metric" in meta, f"{name} meta missing 'metric'"
    assert "unit" in meta, f"{name} meta missing 'unit'"
    assert "geographic_level" in meta, f"{name} meta missing 'geographic_level'"
    assert "time_resolution" in meta, f"{name} meta missing 'time_resolution'"
    assert "period" in meta, f"{name} meta missing 'period'"
    assert "source_ids" in meta, f"{name} meta missing 'source_ids'"
    assert "method" in meta, f"{name} meta missing 'method'"
    assert "observation_count" in meta, f"{name} meta missing 'observation_count'"
    assert "limitations" in meta, f"{name} meta missing 'limitations'"
    assert isinstance(meta["limitations"], list), f"{name} limitations must be a list"

    # Test full JSON serialization without error
    try:
        serialized = json.dumps(res)
        assert len(serialized) > 0
    except Exception as exc:
        pytest.fail(f"{name} output failed JSON serialization: {exc}")


def test_adapter_invalid_input_handling():
    # Invalid state_id in rainfall trend
    res = adapter.get_state_rainfall_trend("InvalidStateXYZ")
    assert res["status"] == "no_data"

    # Invalid state_id in comprehensive report
    res_rep = adapter.get_report_data("InvalidStateXYZ")
    assert res_rep["status"] == "no_data"

    # Invalid region in temperature trend
    res_temp = adapter.get_regional_temperature_trend("INVALID_REGION")
    assert res_temp["status"] == "no_data"

    # Future year with no cyclone data
    res_cy = adapter.get_time_machine_data(2099)
    assert res_cy["status"] == "no_data"
