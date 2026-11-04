"""Evaluate a population matrix with caching."""

import numpy as np

from ..hydraulics.fast_solver import get_solver
from ..hydraulics.simulation import run_hydraulic_simulation
from .cost import calculate_cost_from_indices
from .constraints import check_constraints


def evaluate_population(Pop, problem, cache):
    """Evaluate ``Pop`` (shape ``[NVar, NS]``) and return cost/viol/feas.

    ``cache`` is a plain ``dict`` (equivalent to MATLAB ``containers.Map``).
    """
    NVar, NS = Pop.shape
    cost = np.zeros(NS)
    viol = np.zeros(NS)
    feas = np.zeros(NS, dtype=bool)

    D_m = problem["D"]
    var_pipes = problem["VariablePipes"]
    unit_costs = problem["Cost"]
    lengths = problem["L"]
    Pmin = problem["Pmin"]
    Vmax = problem["Vmax"]
    pipe_names = problem["pipe_names"]
    junction_names = problem["junction_names"]
    wn = problem["wn"]
    initial_D = problem["InitialD"]
    solver = get_solver(problem)          # None -> slow reference path

    for i in range(NS):
        key = tuple(int(x) for x in Pop[:, i])
        if key in cache:
            cost[i], viol[i], feas[i] = cache[key]
            continue

        full_D = initial_D.copy()
        full_D[var_pipes] = D_m[Pop[:, i]]

        if solver is not None:
            Pj, Vp, ok = solver.solve(full_D)
        else:
            Pj, Vp, ok = run_hydraulic_simulation(
                wn, full_D, pipe_names, junction_names
            )
        c = calculate_cost_from_indices(
            Pop[:, i], var_pipes, unit_costs, lengths
        )
        v, f = check_constraints(Pj, Vp, Pmin, Vmax, ok)

        cost[i], viol[i], feas[i] = c, v, f
        cache[key] = (c, v, f)

    return cost, viol, feas