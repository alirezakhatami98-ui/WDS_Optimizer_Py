"""Sweep parameter catalogue.

The list of sweepable parameters is dynamic: built from
ALGORITHM_REGISTRY (for algorithm params) plus a small set of global
parameters (population, generations, constraint values, policy).

Backward compatibility: legacy keys (GA_Pc, PSO_w, ...) from
pre-v0.6 snapshots still work — see LEGACY_MAP below.
"""
from __future__ import annotations

from typing import Any

from ..algorithms.registry import ALGORITHM_REGISTRY
from ..optimization.policies import CONSTRAINT_POLICIES


# Legacy short names from v0.5.x and earlier.  Mapped to v0.6
# namespaced names.  Kept so old snapshots and old user habits
# still work.
LEGACY_MAP: dict[str, str] = {
    "GA_Pc": "GA.ga_pc",
    "GA_Pm": "GA.ga_pm",
    "GA_Selection": "GA.ga_sel",
    "PSO_w": "PSO.pso_w",
    "PSO_c1": "PSO.pso_c1",
    "PSO_c2": "PSO.pso_c2",
    "DE_F": "DE.de_f",
    "DE_CR": "DE.de_cr",
    "DE_Strategy": "DE.de_strategy",
    "SADE_Fmin": "SADE.sade_fmin",
    "SADE_Fmax": "SADE.sade_fmax",
    "SADE_CRmin": "SADE.sade_crmin",
    "SADE_CRmax": "SADE.sade_crmax",
    "SADE_Conv": "SADE.sade_conv",
    "SADE_Tol": "SADE.sade_tol",
    "SADE_Stag": "SADE.sade_stag",
    # Pmin / Vmax already legacy-friendly (same name).
}


def normalize_param_name(key: str) -> str:
    """Return the canonical name for a sweep parameter key.

    Handles legacy names like ``GA_Pc`` transparently.  Unknown keys
    are returned unchanged.
    """
    return LEGACY_MAP.get(key, key)


# ---------------------------------------------------------------------
#  Global parameters (always sweepable)
# ---------------------------------------------------------------------

# Each entry: (display_label, kind, extra)
#   kind = "int" | "float" | "choice"
#   extra = {min, max, step} for numbers
#         = {choices} for choice
_GLOBAL_PARAMS: dict[str, dict] = {
    "NS": {
        "label": "Population / Swarm size (NS)",
        "kind": "int",
        "min": 4, "max": 500, "step": 1,
    },
    "MaxGen": {
        "label": "Max generations / iterations",
        "kind": "int",
        "min": 5, "max": 5000, "step": 1,
    },
    "n_runs": {
        "label": "Runs per sweep point",
        "kind": "int",
        "min": 1, "max": 50, "step": 1,
    },
    "seed": {
        "label": "Experiment seed",
        "kind": "int",
        "min": 0, "max": 2**31 - 1, "step": 1,
    },
    "constraint_policy": {
        "label": "Constraint policy",
        "kind": "choice",
        "choices": tuple(CONSTRAINT_POLICIES.keys()),
    },
    # Constraint values are entered in DISPLAY units (psi / m /
    # ft/s / m/s).  The backend converts to SI before applying.
    # Sweeping a constraint value implicitly enables that constraint.
    "Pmin": {
        "label": "Pmin (value)  — enables the constraint",
        "kind": "float",
        "min": 0.0, "max": 1000.0, "step": 1.0,
        "unit": "pressure",
    },
    "Pmax": {
        "label": "Pmax (value)  — enables the constraint",
        "kind": "float",
        "min": 0.0, "max": 1000.0, "step": 1.0,
        "unit": "pressure",
    },
    "Vmin": {
        "label": "Vmin (value)  — enables the constraint",
        "kind": "float",
        "min": 0.0, "max": 50.0, "step": 0.1,
        "unit": "velocity",
    },
    "Vmax": {
        "label": "Vmax (value)  — enables the constraint",
        "kind": "float",
        "min": 0.1, "max": 50.0, "step": 0.1,
        "unit": "velocity",
    },
}


# ---------------------------------------------------------------------
#  Dynamic catalogue builder
# ---------------------------------------------------------------------

def build_sweep_parameters() -> dict[str, dict]:
    """Return the full catalogue of sweepable parameters.

    Structure: ``{name: {"label": str, "kind": "int"|"float"|"choice",
    ...extra}}``.

    - Global params are always present.
    - Algorithm params come from each registered algorithm's META,
      namespaced as ``<ALGO>.<key>`` (e.g. ``"GA.ga_pc"``).
    - Plugin algorithms' params are included automatically.
    """
    catalog: dict[str, dict] = {}

    # ---- Global ----
    for name, info in _GLOBAL_PARAMS.items():
        catalog[name] = dict(info)

    # ---- Algorithm (dynamic from META) ----
    for short_id, entry in ALGORITHM_REGISTRY.items():
        meta = entry["meta"]
        algo_label = meta.get("display_name", short_id)
        for p in meta["params"]:
            param_key = f"{short_id}.{p['key']}"
            entry_info: dict[str, Any] = {
                "label": f"{algo_label}: {p['display_name']}",
                "kind": p["type"],
                "algorithm": short_id,
                "param_key": p["key"],
            }
            for field in ("min", "max", "step", "choices"):
                if field in p:
                    entry_info[field] = p[field]
            catalog[param_key] = entry_info

    return catalog


# Public, lazily rebuilt.  Existing code that does
# `from sweep.config import SWEEP_PARAMETERS` still works.
# Note: this is a snapshot at import time — if plugins load *after*
# import, they will NOT appear here.  Callers who need a fresh view
# should call build_sweep_parameters() directly.  The GUI does this
# on every render.
SWEEP_PARAMETERS = build_sweep_parameters()


# ---------------------------------------------------------------------
#  Value parser
# ---------------------------------------------------------------------

def parse_sweep_values(raw: str) -> list:
    """Parse a comma-separated values string.

    Strings are kept as strings (for choice params); numbers are
    parsed.  Blank entries are dropped.  Order is preserved.
    """
    if not raw.strip():
        raise ValueError("values list is empty")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise ValueError("values list is empty")
    out: list = []
    for t in tokens:
        try:
            # Integers first (so "1" stays 1, not 1.0)
            out.append(int(t))
            continue
        except ValueError:
            pass
        try:
            out.append(float(t))
            continue
        except ValueError:
            out.append(t)   # keep as string (for choice params)
    return out