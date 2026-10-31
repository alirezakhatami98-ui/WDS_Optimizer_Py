"""Bundle all static data the algorithms need into one dict."""

import numpy as np


def build_optimization_problem(
    *, wn, Din, Cost, D_m, full_lengths, initial_D,
    fixed_pipes, variable_pipes, Pmin, Vmax,
    pipe_names, junction_names, units,
):
    """Return the Problem dict consumed by GA/PSO.

    Notes
    -----
    * ``VariablePipes`` is a 0-based array of indices into
      ``pipe_names`` / ``full_lengths`` / ``initial_D``.
    * ``InitialD`` is the initial diameter of *all* pipes, in meters.
    * WNTR internally uses meters everywhere, so ``D_m`` must be in
      meters; the original values are kept as ``Din`` for display.
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
        "Pmin": float(Pmin),
        "Vmax": float(Vmax),
        "JunctionIndices": np.asarray(junction_names),  # names, not indices
        "Units": units,
    }