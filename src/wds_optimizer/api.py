"""Programmatic API for WDS Optimizer — the non-GUI entry point.

This module is the single source of truth for "given input files, build
the optimization problem and run it".  It is imported by:

- ``app/streamlit_app.py`` (GUI)
- ``cli.py`` (command-line interface, added in a later commit)
- any user code that wants to embed WDS Optimizer in its own pipeline

No Streamlit, Plotly, or GUI-related imports appear anywhere here.
This is deliberate: the module must be usable on a headless server
(SaaS, CI, embedded).

v0.1.0 public surface
---------------------
- ``load_network_isolated(inp_path)``
    Parse a WNTR network and return a fresh mutable copy — safe to
    mutate without affecting subsequent calls.

- ``build_problem(inp_path=..., d_path=..., cost_path=..., diam_unit=...,
                  constraints=..., fixed_pipes=...)``
    Load all three input files, validate, and return
    ``(problem, units, wn)`` — the tuple that ``run_experiment`` consumes.

Everything else in this module is private (leading underscore).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import numpy as np
import wntr

from wds_optimizer.data.loaders import load_optimization_data
from wds_optimizer.hydraulics.units import detect_network_units
from wds_optimizer.optimization.problem import build_optimization_problem


# =====================================================================
#  Network loading
# =====================================================================

def _parse_network(inp_path: str | Path) -> wntr.network.WaterNetworkModel:
    """Raw parse — no copy, no cache.

    Private.  Streamlit uses this under ``@st.cache_resource`` so the
    expensive parse runs once per file, while ``load_network_isolated``
    (below) hands out fresh copies per call.
    """
    return wntr.network.WaterNetworkModel(str(inp_path))


def load_network_isolated(
    inp_path: str | Path,
) -> wntr.network.WaterNetworkModel:
    """Parse a network and return a *fresh, mutable* copy.

    During optimisation ``run_hydraulic_simulation`` mutates the model
    (sets pipe diameters on it for each candidate design).  Returning a
    fresh copy per call guarantees those mutations do not leak between
    experiments — see the Hanoi→Duhala→Hanoi regression test.

    Callers that need the parse-only path (for caching) should use
    ``_parse_network`` and deepcopy the result themselves.
    """
    return copy.deepcopy(_parse_network(inp_path))


# =====================================================================
#  Problem construction
# =====================================================================

def _parse_fixed_pipes(
    fixed_pipes: Sequence[int] | str, n_pipes: int,
) -> list[int]:
    """Normalise ``fixed_pipes`` to a sorted list of 0-based ints.

    Accepts either:

    - a sequence of 0-based indices (``[0, 2]``), as API callers
      and stored snapshots use;
    - a comma-separated 1-based string (``"1, 3"``), as typed in the
      GUI's "Fixed pipe IDs" field.

    Raises ``ValueError`` with the same messages the GUI has always
    shown, so users who saw those errors continue to recognise them.
    """
    if isinstance(fixed_pipes, str):
        s = fixed_pipes.strip()
        if not s:
            return []
        try:
            one_based = [
                int(x.strip()) for x in s.split(",") if x.strip()
            ]
        except ValueError:
            raise ValueError(
                "Fixed pipe IDs must be integers separated by commas."
            )
        for f in one_based:
            if f < 1 or f > n_pipes:
                raise ValueError(
                    f"Fixed pipe IDs must be between 1 and {n_pipes}."
                )
        if len(set(one_based)) != len(one_based):
            raise ValueError("Fixed pipe IDs must be unique.")
        return sorted(f - 1 for f in one_based)

    indices = list(fixed_pipes or [])
    for f in indices:
        if not isinstance(f, int) or f < 0 or f >= n_pipes:
            raise ValueError(
                f"Fixed pipe index {f} out of range 0..{n_pipes - 1}."
            )
    if len(set(indices)) != len(indices):
        raise ValueError("Fixed pipe indices must be unique.")
    return sorted(indices)


def _validate_constraints(constraints: dict) -> None:
    """Check the four-constraint dict for structural and logical errors.

    Same conditions the GUI's ``_render_sidebar`` checks live; duplicated
    here so the API is self-sufficient.  Error messages are in SI because
    the API never sees display-unit values — the GUI validates in
    display units first for a friendlier message (see
    ``app/streamlit_app._prepare_problem``).
    """
    if not isinstance(constraints, dict):
        raise ValueError(
            "constraints must be a dict: "
            "{'Pmin'|'Pmax'|'Vmin'|'Vmax': {'enabled': bool, 'value': float}}"
        )

    required = ("Pmin", "Pmax", "Vmin", "Vmax")
    for key in required:
        if key not in constraints:
            raise ValueError(f"constraints missing key: {key!r}")
        entry = constraints[key]
        if (not isinstance(entry, dict)
                or "enabled" not in entry or "value" not in entry):
            raise ValueError(
                f"constraints[{key!r}] must be "
                f"{{'enabled': bool, 'value': float}}"
            )

    if not any(constraints[k]["enabled"] for k in required):
        raise ValueError("Enable at least one constraint before running.")

    if (constraints["Pmin"]["enabled"]
            and constraints["Pmax"]["enabled"]
            and float(constraints["Pmin"]["value"])
                >= float(constraints["Pmax"]["value"])):
        raise ValueError(
            f"Pmin ({constraints['Pmin']['value']}) must be smaller than "
            f"Pmax ({constraints['Pmax']['value']})."
        )
    if (constraints["Vmin"]["enabled"]
            and constraints["Vmax"]["enabled"]
            and float(constraints["Vmin"]["value"])
                >= float(constraints["Vmax"]["value"])):
        raise ValueError(
            f"Vmin ({constraints['Vmin']['value']}) must be smaller than "
            f"Vmax ({constraints['Vmax']['value']})."
        )


def build_problem(
    *,
    inp_path: str | Path,
    d_path: str | Path,
    cost_path: str | Path,
    diam_unit: str,
    constraints: dict,
    fixed_pipes: Sequence[int] | str = (),
) -> tuple[dict, dict, wntr.network.WaterNetworkModel]:
    """Build the optimisation problem dict from raw input files.

    Parameters
    ----------
    inp_path, d_path, cost_path : path-like
        Network, diameter-options and unit-cost files.
    diam_unit : {"mm", "in"}
        Unit of the values in ``d_path``.
    constraints : dict
        SI values and enabled flags::

            {
                "Pmin": {"enabled": bool, "value": float},   # m
                "Pmax": {"enabled": bool, "value": float},   # m
                "Vmin": {"enabled": bool, "value": float},   # m/s
                "Vmax": {"enabled": bool, "value": float},   # m/s
            }

    fixed_pipes : sequence[int] | str, default ()
        0-based indices (``[0, 2]``) or comma-separated 1-based string
        (``"1, 3"``, as the GUI accepts).

    Returns
    -------
    (problem, units, wn)
        ``problem`` is the dict consumed by ``run_experiment``;
        ``units`` describes the network's native unit system (for
        display); ``wn`` is the loaded, isolated network model.

    Raises
    ------
    ValueError
        If the constraints are invalid, ``fixed_pipes`` references an
        out-of-range index, or a file does not exist.
    """
    _validate_constraints(constraints)

    wn = load_network_isolated(inp_path)
    units = detect_network_units(wn)

    Din, Cost, D_m = load_optimization_data(
        str(d_path), str(cost_path),
        diam_file_unit=diam_unit,
    )

    pipe_names = list(wn.pipe_name_list)
    junction_names = list(wn.junction_name_list)
    lengths = np.array([wn.get_link(n).length for n in pipe_names])
    initial_D = np.array([wn.get_link(n).diameter for n in pipe_names])

    fixed_0based = _parse_fixed_pipes(fixed_pipes, len(pipe_names))
    variable_0based = [
        i for i in range(len(pipe_names)) if i not in fixed_0based
    ]

    problem = build_optimization_problem(
        wn=wn, Din=Din, Cost=Cost, D_m=D_m,
        full_lengths=lengths, initial_D=initial_D,
        fixed_pipes=fixed_0based,
        variable_pipes=variable_0based,
        constraints=constraints,
        pipe_names=pipe_names,
        junction_names=junction_names,
        units=units,
    )
    return problem, units, wn