"""Unit tests for constants, file resolution, and state normalization."""

from pathlib import Path
import pytest

from climatelens.constants import (
    DATASET_METADATA,
    STATES_MAP,
    TEMPERATURE_REGIONS,
    normalize_state_id,
    resolve_raw_file,
)


def test_states_map_completeness():
    assert len(STATES_MAP) == 36
    for sid, meta in STATES_MAP.items():
        assert meta["state_id"] == sid
        assert isinstance(meta["state_name"], str) and meta["state_name"]
        assert len(meta["state_code"]) == 2
        assert len(meta["stcode"]) == 2
        assert meta["type"] in ("State", "Union Territory")
        assert meta["region"] in ("North", "South", "East", "West", "Central", "Northeast", "Islands")


def test_state_normalization_variations():
    # Standard exact names
    assert normalize_state_id("MAHARASHTRA") == "MAHARASHTRA"
    assert normalize_state_id("Kerala") == "KERALA"

    # Common spelling variations and aliases
    assert normalize_state_id("ARUNANCHAL PRADESH") == "ARUNACHAL_PRADESH"
    assert normalize_state_id("Dadra & Nagar Havelli and Daman & Diu") == "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU"
    assert normalize_state_id("DADRA & NAGAR HAVELI AND DAMAN & DIU (UT)") == "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU"
    assert normalize_state_id("Daman & Diu") == "DADRA_AND_NAGAR_HAVELI_AND_DAMAN_AND_DIU"
    assert normalize_state_id("PONDICHERRY") == "PUDUCHERRY"
    assert normalize_state_id("ORISSA") == "ODISHA"
    assert normalize_state_id("UTTARANCHAL") == "UTTARAKHAND"
    assert normalize_state_id("New Delhi") == "DELHI"
    assert normalize_state_id("Madras") == "TAMIL_NADU"

    # Census and abbreviation codes
    assert normalize_state_id("MH") == "MAHARASHTRA"
    assert normalize_state_id("27") == "MAHARASHTRA"
    assert normalize_state_id("DL") == "DELHI"
    assert normalize_state_id("07") == "DELHI"

    # Invalid / None handling
    assert normalize_state_id(None) is None
    assert normalize_state_id("") is None
    assert normalize_state_id("NonexistentState123") is None


def test_temperature_regions_completeness():
    assert len(TEMPERATURE_REGIONS) == 8
    assert "ALLIN" in TEMPERATURE_REGIONS
    for reg, meta in TEMPERATURE_REGIONS.items():
        assert meta["region_code"] == reg
        assert isinstance(meta["region_name"], str)


def test_dataset_metadata_and_file_resolution():
    assert len(DATASET_METADATA) == 7
    for key in DATASET_METADATA:
        path = resolve_raw_file(key)
        assert path.exists(), f"Raw file for {key} does not exist at {path}"
        assert path.is_file()
        assert path.stat().st_size > 0
