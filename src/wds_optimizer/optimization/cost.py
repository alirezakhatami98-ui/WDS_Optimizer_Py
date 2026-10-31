"""Cost of the variable pipes only."""

import numpy as np


def calculate_cost_from_indices(indices, variable_pipes, unit_costs, lengths):
    """Mirror of the MATLAB function.

    indices[j] is the diameter-option index (0-based) of pipe
    ``variable_pipes[j]``. ``unit_costs`` and ``lengths`` are in the
    Problem dict.
    """
    indices = np.asarray(indices, dtype=int)
    variable_pipes = np.asarray(variable_pipes, dtype=int)
    if indices.size != variable_pipes.size:
        raise ValueError(
            f"indices has {indices.size} entries but variable_pipes has "
            f"{variable_pipes.size}."
        )
    Lv = np.asarray(lengths, dtype=float)[variable_pipes]
    Cv = np.asarray(unit_costs, dtype=float)[indices]
    return float(np.sum(Lv * Cv))