"""Tests for the algorithm registry.

The registry is the foundation for the plugin system (v0.5) and the
metadata-driven GUI (v0.3.0).  It must stay a pure lookup table:
no knowledge of AlgorithmConfig, runner, or any consumer.
"""
from __future__ import annotations

import pytest

from wds_optimizer.algorithms.registry import (
    ALGORITHM_REGISTRY,
    get_algorithm,
    list_algorithms,
    register,
)


def test_builtin_algorithms_registered():
    """The four algorithms from v0.2.1 are all present."""
    names = list_algorithms()
    for expected in ("GA", "PSO", "DE", "SADE"):
        assert expected in names, (
            f"Missing built-in algorithm {expected!r}"
        )


def test_every_entry_has_run_and_meta():
    for name, entry in ALGORITHM_REGISTRY.items():
        assert callable(entry["run"]), f"{name}: run is not callable"
        assert isinstance(entry["meta"], dict), f"{name}: meta is not a dict"


def test_short_ids_are_unique():
    short_ids = [e["meta"]["short_id"] for e in ALGORITHM_REGISTRY.values()]
    assert len(short_ids) == len(set(short_ids)), (
        f"Duplicate short_ids: {short_ids}"
    )


def test_meta_has_required_keys():
    for name, entry in ALGORITHM_REGISTRY.items():
        meta = entry["meta"]
        for key in ("short_id", "display_name", "category",
                    "paper_faithful", "description", "references", "params"):
            assert key in meta, f"{name}: meta missing key {key!r}"


def test_all_params_have_required_keys():
    for name, entry in ALGORITHM_REGISTRY.items():
        for p in entry["meta"]["params"]:
            for key in ("key", "name", "display_name",
                        "type", "default", "description"):
                assert key in p, (
                    f"{name}: param {p.get('name', '?')} missing {key!r}"
                )


def test_choice_params_have_choices():
    for name, entry in ALGORITHM_REGISTRY.items():
        for p in entry["meta"]["params"]:
            if p["type"] == "choice":
                assert p.get("choices"), (
                    f"{name}: choice param {p['name']} has no choices"
                )
                assert p["default"] in p["choices"], (
                    f"{name}: {p['name']} default not in choices"
                )


def test_lookup_by_short_id():
    for name in list_algorithms():
        entry = get_algorithm(name)
        assert entry["meta"]["short_id"] == name


def test_lookup_by_display_name():
    entry = get_algorithm("GA")
    display_name = entry["meta"]["display_name"]
    entry2 = get_algorithm(display_name)
    assert entry2 is entry


def test_unknown_algorithm_raises():
    with pytest.raises(KeyError, match="Unknown algorithm"):
        get_algorithm("NOT_AN_ALGORITHM")


def test_register_rejects_duplicate():
    with pytest.raises(ValueError, match="already registered"):
        register(
            "GA", run=lambda *a, **k: None,
            meta=ALGORITHM_REGISTRY["GA"]["meta"],
        )


def test_register_rejects_mismatched_short_id():
    with pytest.raises(ValueError, match="does not match"):
        register(
            "FAKE", run=lambda *a, **k: None,
            meta={
                "short_id": "OTHER",
                "display_name": "X", "category": "X",
                "paper_faithful": False, "description": "X",
                "references": [], "params": [],
            },
        )


def test_register_rejects_unknown_param_type():
    with pytest.raises(ValueError, match="unknown type"):
        register(
            "FAKE", run=lambda *a, **k: None,
            meta={
                "short_id": "FAKE",
                "display_name": "Fake", "category": "X",
                "paper_faithful": False, "description": "X",
                "references": [],
                "params": [{
                    "key": "x", "name": "x", "display_name": "x",
                    "type": "unknown", "default": 0, "description": "x",
                }],
            },
        )