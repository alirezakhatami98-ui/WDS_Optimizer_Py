"""Bundle all static data the algorithms need into one dict."""

import numpy as np


def build_optimization_problem(
    *, wn, Din, Cost, D_m, full_lengths, initial_D,
    fixed_pipes, variable_pipes, constraints,
    pipe_names, junction_names, units,
):
    """Return the Problem dict consumed by GA/PSO/SADE.

    ``constraints`` is ``{"Pmin"|"Pmax"|"Vmin"|"Vmax": {"enabled": bool,
    "value": float}}`` with pressure values in metres and velocity
    values in m/s.
    """
    return {
        "wn": wn,
        "pipe_names": list(pipe_names),
        "junction_names": list(junction_names),
        "Din": np.asarray(Din, dtype=float),
        "Cost": np.asarray(Cost, dtype=float),
        "D": np.asarray(D_m, dtype=float),
        "NP": len(pipe_names),
        "L": np.asarray(full_lengths, dtype=float),
        "InitialD": np.asarray(initial_D, dtype=float),
        "FixedPipes": np.asarray(fixed_pipes, dtype=int),
        "VariablePipes": np.asarray(variable_pipes, dtype=int),
        "Constraints": constraints,
        "JunctionIndices": np.asarray(junction_names),
        "Units": units,
    }