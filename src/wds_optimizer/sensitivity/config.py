"""Sensitivity analysis parameter catalogue.

The list of analyzable parameters is dynamic: built from
ALGORITHM_REGISTRY (for algorithm params) plus a small set of global
parameters (population, generations, constraint values, policy).

The user-facing term is "sensitivity analysis"; the mechanical
operation of varying one parameter is a "sweep".  See also
wds_optimizer/sensitivity/runner.py and, in v0.8.1+,
wds_optimizer/sensitivity/batch.py.

Backward compatibility: legacy short names (GA_Pc, PSO_w, ...) from
pre-v0.6 snapshots still work — see LEGACY_MAP below.
"""
from __future__ import annotations

from typing import Any

from ..algorithms.registry import ALGORITHM_REGISTRY
from ..optimization.policies import CONSTRAINT_POLICIES


# Legacy short names from v0.5.x and earlier.  Mapped to v0.6+
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
}


def normalize_param_name(key: str) -> str:
    """Return the canonical name for a sensitivity parameter key.

    Handles legacy names like ``GA_Pc`` transparently.  Unknown keys
    are returned unchanged.
    """
    return LEGACY_MAP.get(key, key)


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
    # Constraint values are entered in DISPLAY units.  The runner
    # converts to SI and enables the constraint.
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


def build_sensitivity_parameters() -> dict[str, dict]:
    """Return the full catalogue of sensitivity-analyzable parameters.

    Structure: ``{name: {"label": str, "kind": "int"|"float"|"choice"|"bool",
    ...extra}}``.

    - Global params are always present.
    - Algorithm params come from each registered algorithm's META,
      namespaced as ``<ALGO>.<key>`` (e.g. ``"GA.ga_pc"``).
    - Plugin algorithms' params are included automatically.
    """
    catalog: dict[str, dict] = {}
    for name, info in _GLOBAL_PARAMS.items():
        catalog[name] = dict(info)
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


# Primary name (v0.8.0).
SENSITIVITY_PARAMETERS = build_sensitivity_parameters()

# Aliases for backward compatibility with v0.6/v0.7.
SWEEP_PARAMETERS = SENSITIVITY_PARAMETERS
build_sweep_parameters = build_sensitivity_parameters


def parse_sensitivity_values(raw: str) -> list:
    """Parse a comma-separated values string.

    Numbers become int/float; other tokens become strings (for choice
    params).  A single layer of surrounding single or double quotes is
    stripped.  Blank entries are dropped.  Order is preserved.
    """
    if not raw.strip():
        raise ValueError("values list is empty")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise ValueError("values list is empty")
    out: list = []
    for t in tokens:
        if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
            t = t[1:-1]
        try:
            out.append(int(t))
            continue
        except ValueError:
            pass
        try:
            out.append(float(t))
            continue
        except ValueError:
            out.append(t)
    return out


# Alias for backward compatibility.
parse_sweep_values = parse_sensitivity_values