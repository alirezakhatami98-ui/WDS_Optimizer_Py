"""Algorithm registry — single source of truth for available algorithms.

Each algorithm module exports:

    run_<name>(problem, config, *, progress_callback, cancel_callback,
               meta) -> (score, position, convergence, feasible)
    ALGORITHM_META = {...}

This registry collects them and provides lookup by short_id or
display_name.  Adding a new algorithm (including user plugins in v0.5)
means dropping a new module in algorithms/ and calling ``register()``.

Design invariants (checked in tests/test_algorithm_registry.py):
- Every registered entry has both "run" and "meta".
- ``meta["short_id"]`` is unique and matches the registry key.
- Every param has all required keys.
- Param types are one of float / int / choice / bool.
- ``choice`` params have a non-empty "choices" list.

The registry deliberately does NOT know about AlgorithmConfig, runner,
GUI, or any consumer.  Consumers subscribe; the registry stays a plain
lookup table.  This keeps the plugin path open: a user plugin only
needs to declare a run function and META — no changes to this file.
"""
from __future__ import annotations

from typing import Callable

from .ga import ALGORITHM_META as GA_META, run_ga
from .pso import ALGORITHM_META as PSO_META, run_pso
from .de import ALGORITHM_META as DE_META, run_de
from .sade import ALGORITHM_META as SADE_META, run_sade


_REQUIRED_META_KEYS = (
    "short_id", "display_name", "category", "paper_faithful",
    "description", "references", "params",
)
_REQUIRED_PARAM_KEYS = (
    "key", "name", "display_name", "type", "default", "description",
)
_VALID_PARAM_TYPES = ("float", "int", "choice", "bool")


def _validate_meta(meta: dict) -> None:
    if not isinstance(meta, dict):
        raise ValueError("ALGORITHM_META must be a dict")
    for key in _REQUIRED_META_KEYS:
        if key not in meta:
            raise ValueError(
                f"ALGORITHM_META[{meta.get('short_id', '?')!r}] "
                f"missing required key: {key!r}"
            )
    if not isinstance(meta["params"], list):
        raise ValueError(
            f"ALGORITHM_META[{meta['short_id']!r}].params must be a list"
        )
    seen_keys = set()
    for p in meta["params"]:
        if not isinstance(p, dict):
            raise ValueError(
                f"ALGORITHM_META[{meta['short_id']!r}] param entries "
                f"must be dicts"
            )
        for key in _REQUIRED_PARAM_KEYS:
            if key not in p:
                raise ValueError(
                    f"ALGORITHM_META[{meta['short_id']!r}].params"
                    f"[{p.get('name', '?')!r}] missing key: {key!r}"
                )
        if p["type"] not in _VALID_PARAM_TYPES:
            raise ValueError(
                f"ALGORITHM_META[{meta['short_id']!r}] param "
                f"{p['name']!r} has unknown type {p['type']!r}"
            )
        if p["type"] == "choice":
            choices = p.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"ALGORITHM_META[{meta['short_id']!r}] choice "
                    f"param {p['name']!r} must have a non-empty "
                    f"'choices' list"
                )
            if p["default"] not in choices:
                raise ValueError(
                    f"ALGORITHM_META[{meta['short_id']!r}] choice "
                    f"param {p['name']!r}: default {p['default']!r} "
                    f"is not in choices {choices!r}"
                )
        if p["key"] in seen_keys:
            raise ValueError(
                f"ALGORITHM_META[{meta['short_id']!r}] duplicate "
                f"param key {p['key']!r}"
            )
        seen_keys.add(p["key"])


ALGORITHM_REGISTRY: dict[str, dict] = {}


def register(short_id: str, *, run: Callable, meta: dict) -> None:
    """Register an algorithm.

    Raises if the meta is malformed, if the short_id does not match
    ``meta['short_id']``, or if the id is already registered.
    """
    _validate_meta(meta)
    if meta["short_id"] != short_id:
        raise ValueError(
            f"Registry key {short_id!r} does not match "
            f"meta['short_id'] {meta['short_id']!r}"
        )
    if short_id in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Algorithm {short_id!r} is already registered"
        )
    ALGORITHM_REGISTRY[short_id] = {"run": run, "meta": meta}


def get_algorithm(name: str) -> dict:
    """Look up an algorithm by registry key or display name."""
    if name in ALGORITHM_REGISTRY:
        return ALGORITHM_REGISTRY[name]
    for entry in ALGORITHM_REGISTRY.values():
        if entry["meta"]["display_name"] == name:
            return entry
    raise KeyError(
        f"Unknown algorithm {name!r}.  Available: "
        f"{sorted(ALGORITHM_REGISTRY.keys())}"
    )


def list_algorithms() -> list[str]:
    """Registry keys in registration order."""
    return list(ALGORITHM_REGISTRY.keys())


# ---- Built-in registrations ----

register("GA",   run=run_ga,   meta=GA_META)
register("PSO",  run=run_pso,  meta=PSO_META)
register("DE",   run=run_de,   meta=DE_META)
register("SADE", run=run_sade, meta=SADE_META)

# Paper-faithful variants (v0.3)
from .de_suribabu_2010 import (  # noqa: E402
    ALGORITHM_META as DE_SURIBABU_META,
    run_de_suribabu_2010,
)
register(
    "DE_suribabu_2010",
    run=run_de_suribabu_2010,
    meta=DE_SURIBABU_META,
)