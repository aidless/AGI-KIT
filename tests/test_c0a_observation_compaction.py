from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from are.simulation.apps.apartment_listing import Apartment
from scripts.run_gaia2_canonical import NativeToolEngine


def make_apartment(index: int, *, saved: bool, price: int | float) -> Apartment:
    return Apartment(
        name=f"Flat {index}",
        location=f"District {index % 7}",
        zip_code=f"94{index:03d}",
        price=price,
        bedrooms=1 + index % 3,
        bathrooms=1 + index % 2,
        property_type="apartment",
        square_footage=500 + index,
        furnished_status="furnished",
        floor_level="3",
        pet_policy="cats allowed",
        lease_term="12 months",
        apartment_id=f"{index:032x}",
        amenities=["elevator"],
        saved=saved,
    )


def render_observation(apartments: list[Apartment]) -> str:
    state = {apartment.apartment_id: apartment for apartment in apartments}
    return "Observation: " + repr(state)


def test_compaction_preserves_decision_fields_for_real_are_repr() -> None:
    apartments = [
        make_apartment(1, saved=True, price=2500),
        make_apartment(2, saved=False, price=2500.0),
    ]
    summary = NativeToolEngine._compact_tool_response(render_observation(apartments))

    for apartment in apartments:
        assert apartment.apartment_id in summary
        assert f"location='{apartment.location}'" in summary
        assert f"price={apartment.price}" in summary
        assert f"saved={apartment.saved}" in summary

    assert "STATE SUMMARY (all apartments; decision fields preserved)" in summary


def test_transform_levels_are_semantically_distinct() -> None:
    apartment = make_apartment(7, saved=True, price=2500.0)
    raw = render_observation([apartment]) + (" extra" * 1000)

    raw_engine = NativeToolEngine(lambda: {}, "http://127.0.0.1:9", "test", observation_transform="raw_truncate_3500")
    legacy_engine = NativeToolEngine(lambda: {}, "http://127.0.0.1:9", "test", observation_transform="legacy_id_name")
    decision_engine = NativeToolEngine(lambda: {}, "http://127.0.0.1:9", "test", observation_transform="decision_fields")

    raw_summary = raw_engine._transform_observation(raw)
    legacy_summary = legacy_engine._transform_observation(raw)
    decision_summary = decision_engine._transform_observation(raw)

    assert len(raw_summary) == 3500
    assert apartment.location not in legacy_summary
    assert f"price={apartment.price}" not in legacy_summary
    assert f"location='{apartment.location}'" in decision_summary
    assert f"price={apartment.price}" in decision_summary
    assert f"saved={apartment.saved}" in decision_summary


def test_compaction_preserves_all_125_real_are_apartments() -> None:
    apartments = [
        make_apartment(index, saved=(index % 5 == 0), price=1800 + index / 10)
        for index in range(1, 126)
    ]
    summary = NativeToolEngine._compact_tool_response(render_observation(apartments))

    assert summary.count(" | name=") == 125
    for apartment in apartments:
        assert apartment.apartment_id in summary
        assert f"location='{apartment.location}'" in summary
        assert f"price={apartment.price}" in summary
        assert f"saved={apartment.saved}" in summary


if __name__ == "__main__":
    test_compaction_preserves_decision_fields_for_real_are_repr()
    test_transform_levels_are_semantically_distinct()
    test_compaction_preserves_all_125_real_are_apartments()
    print("C0A-08 compaction regression tests passed")
