"""Differential Evolution — paper-faithful reproduction of Suribabu (2010).

Reference
---------
Suribabu, C. R. (2010). "Differential evolution algorithm for optimal
design of water distribution networks." *Journal of Hydroinformatics*,
12(1), 66-82.

What makes this different from ``de.py``
----------------------------------------
The standard ``de.py`` in this project uses Deb's constraint tournament
(feasible-first, cost-second) and supports three mutation strategies.
The paper specifies something different:

1. **Fixed penalty** (Eq. 4 of the Application section):
   infeasible designs get ``effective_cost = cost of all variable
   pipes at max diameter`` — a constant.  Any feasible design beats
   any infeasible one.  This is NOT Deb's tournament.

2. **No jrand** (Eq. 8): crossover does not force at least one gene
   from the mutant.  With probability ``(1 - CR)^NVar`` a trial can
   be an exact copy of its target.

3. **Simple greedy selection** (Eq. 9):
   ``trial replaces target iff effective_cost(trial) <=
   effective_cost(target)``.  Ties go to the trial.

4. **Clip bound handling** (not re-draw):
   out-of-range continuous values are clipped to ``[0, ND-1]`` and
   rounded to the nearest diameter index for evaluation.

5. **DE/rand/1/bin only.**  No strategy choice; the ``DEConfig.Strategy``
   field is ignored.

6. **No restart-on-stagnation.**  Not part of the paper.

The paper's termination is a fixed number of generations; that is what
``config.MaxGen`` provides here.

Implementation notes
--------------------
- Positions are kept as continuous floats in ``[0, ND-1]`` and rounded
  to the nearest integer (MATLAB-style round-half-away-from-zero) only
  at evaluation time.  This matches the paper's "discrete sizes ...
  converted to continuous diameter when it undergoes the mutation
  process" description.
- The effective cost of an infeasible design is the cost of all
  variable pipes at the largest diameter (paper's Eq. 4).
- Best feasible is tracked throughout; if none is ever found, the
  return score is ``max_cost`` (the penalty ceiling) with
  ``feasible=False`` so downstream code cannot mistake it for a real
  solution.

Run-to-run variance is expected to be higher than ``de.py`` because
(a) no jrand reduces diversity, (b) fixed penalty collapses all
infeasible designs to a single effective cost, and (c) small
populations (the paper uses N=20) amplify variance.  The paper reports
40-70% success rates depending on settings.
"""

from __future__ import annotations

import numpy as np

from .compare import is_better_solution
from .de import _discretize, DECancelled
from ..optimization.evaluate import evaluate_population


def run_de_suribabu_2010(
    problem, config, *,
    progress_callback=None, cancel_callback=None, meta=None,
):
    """Run paper-faithful DE (Suribabu 2010) and return
    ``(score, full_position_m, convergence, feasible)``.

    The return contract matches ``run_de``.  When no feasible design
    is found, ``score`` is the penalty ceiling (cost of all variable
    pipes at max diameter) and ``feasible=False``.
    """
    rng = (
        np.random.default_rng(config.Seed)
        if config.Seed is not None
        else np.random.default_rng()
    )

    NP = config.NS
    MaxGen = config.MaxGen
    NVar = len(problem["VariablePipes"])
    ND = len(problem["D"])
    F = float(config.DE.F)
    CR = float(config.DE.CR)

    if NP < 4:
        raise ValueError("DE needs a population of at least 4 members.")
    if ND < 2:
        raise ValueError("DE needs at least 2 diameter options.")
    if NVar < 1:
        raise ValueError("DE needs at least 1 variable pipe.")

    # Paper's penalty ceiling: cost of all variable pipes at max diameter.
    lengths = problem["L"]
    unit_costs = problem["Cost"]
    var_pipes = problem["VariablePipes"]
    max_cost = float(
        sum(lengths[int(vp)] for vp in var_pipes) * unit_costs[ND - 1]
    )

    cache = config.Cache if config.Cache is not None else {}
    cache0 = len(cache)
    n_evals = 0
    nfe_best = 0

    def _record_meta():
        if meta is not None:
            meta["nfe"] = n_evals
            meta["n_sim"] = len(cache) - cache0
            meta["nfe_to_best"] = nfe_best

    if meta is not None:
        meta["restart_gens"] = []
        meta["epoch_bests"] = []

    Conv = np.zeros(MaxGen)
    ProgressInterval = max(1, MaxGen // 50)
    ProgressFailed = False

    # ---- Initial population (Eq. 7): continuous uniform in [0, ND-1] ----
    X = rng.uniform(0.0, float(ND - 1), size=(NVar, NP))
    X_discrete = _discretize(X, ND)
    cost, viol, feas = evaluate_population(X_discrete, problem, cache)
    n_evals += NP

    # Paper's Eq. 4: infeasible → max_cost
    eff = np.where(feas, cost, max_cost)

    # ---- Track best feasible and best infeasible ----
    best_feasible_cost = float("inf")
    best_feasible_sol: np.ndarray | None = None
    best_infeasible_viol = float("inf")
    best_infeasible_sol: np.ndarray | None = None

    feas_idx = np.where(feas)[0]
    if feas_idx.size:
        ib = int(feas_idx[np.argmin(cost[feas_idx])])
        best_feasible_cost = float(cost[ib])
        best_feasible_sol = X[:, ib].copy()
        nfe_best = n_evals
    else:
        iv = int(np.argmin(viol))
        best_infeasible_viol = float(viol[iv])
        best_infeasible_sol = X[:, iv].copy()

    def _current_best_cost() -> float:
        if best_feasible_sol is not None:
            return best_feasible_cost
        return max_cost

    def _make_partial(G: int):
        if best_feasible_sol is not None:
            sol = _discretize(best_feasible_sol, ND)
            pos = problem["InitialD"].copy()
            pos[var_pipes] = problem["D"][sol]
            conv = np.full(MaxGen, np.nan)
            conv[:G] = Conv[:G]
            return (best_feasible_cost, pos, conv, True)
        if best_infeasible_sol is not None:
            sol = _discretize(best_infeasible_sol, ND)
            pos = problem["InitialD"].copy()
            pos[var_pipes] = problem["D"][sol]
            conv = np.full(MaxGen, np.nan)
            conv[:G] = Conv[:G]
            return (max_cost, pos, conv, False)
        return None

    idx_all = np.arange(NP)

    for G in range(MaxGen):
        # ---- cancellation ----
        if cancel_callback is not None and cancel_callback():
            _record_meta()
            raise DECancelled(
                "DE_suribabu_2010 execution was cancelled by the user.",
                partial=_make_partial(G),
            )

        # ---- mutation: DE/rand/1 only (Eq. 8) ----
        V = np.empty_like(X)
        for i in range(NP):
            others = idx_all[idx_all != i]
            r1, r2, r3 = rng.choice(others, size=3, replace=False)
            V[:, i] = X[:, r3] + F * (X[:, r1] - X[:, r2])

        # ---- crossover: binomial, NO jrand (Eq. 8) ----
        mask = rng.random((NVar, NP)) <= CR
        U = np.where(mask, V, X)

        # ---- bound handling: clip (NOT re-draw) ----
        U = np.clip(U, 0.0, float(ND - 1))

        # ---- evaluate ----
        U_discrete = _discretize(U, ND)
        cost_t, viol_t, feas_t = evaluate_population(
            U_discrete, problem, cache,
        )
        n_evals += NP
        eff_t = np.where(feas_t, cost_t, max_cost)

        # ---- selection: greedy on effective cost, ties to trial (Eq. 9) ----
        improve = eff_t <= eff
        X[:, improve] = U[:, improve]
        cost[improve] = cost_t[improve]
        viol[improve] = viol_t[improve]
        feas[improve] = feas_t[improve]
        eff[improve] = eff_t[improve]

        # ---- update bests ----
        feas_idx = np.where(feas)[0]
        if feas_idx.size:
            ib = int(feas_idx[np.argmin(cost[feas_idx])])
            if cost[ib] < best_feasible_cost:
                best_feasible_cost = float(cost[ib])
                best_feasible_sol = X[:, ib].copy()
                nfe_best = n_evals
        inv_idx = np.where(~feas)[0]
        if inv_idx.size:
            iv = int(inv_idx[np.argmin(viol[inv_idx])])
            if viol[iv] < best_infeasible_viol:
                best_infeasible_viol = float(viol[iv])
                best_infeasible_sol = X[:, iv].copy()

        Conv[G] = _current_best_cost()

        if progress_callback is not None and not ProgressFailed:
            if (G == 0) or ((G % ProgressInterval) == 0) \
                    or (G == MaxGen - 1):
                try:
                    progress_callback({
                        "Algorithm": "DE_suribabu_2010",
                        "Generation": G + 1,
                        "MaxGenerations": MaxGen,
                        "BestCost": float(Conv[G]),
                        "Feasible": best_feasible_sol is not None,
                        "Convergence": Conv[: G + 1].copy(),
                        "NFE": n_evals,
                        "Restarts": 0,
                        "RestartGens": [],
                    })
                except Exception as e:
                    print(
                        f"[run_de_suribabu_2010] progress_callback "
                        f"error: {e}"
                    )
                    ProgressFailed = True

    _record_meta()

    if best_feasible_sol is not None:
        sol = _discretize(best_feasible_sol, ND)
        pos = problem["InitialD"].copy()
        pos[var_pipes] = problem["D"][sol]
        return best_feasible_cost, pos, Conv, True

    if best_infeasible_sol is not None:
        sol = _discretize(best_infeasible_sol, ND)
        pos = problem["InitialD"].copy()
        pos[var_pipes] = problem["D"][sol]
        return max_cost, pos, Conv, False

    # No designs evaluated (defensive — should not happen).
    pos = problem["InitialD"].copy()
    return max_cost, pos, Conv, False


# =====================================================================
#  Algorithm metadata
# =====================================================================

ALGORITHM_META = {
    "short_id": "DE_suribabu_2010",
    "display_name": "DE_suribabu_2010",
    "category": "Differential Evolution",
    "paper_faithful": True,
    "description": (
        "Paper-faithful DE from Suribabu (2010): DE/rand/1/bin, "
        "binomial crossover without jrand, fixed penalty for "
        "infeasible designs (cost = all-max network), simple greedy "
        "selection, clip bound handling.  No restart-on-stagnation."
    ),
    "references": [
        "Suribabu, C. R. (2010). \"Differential evolution algorithm "
        "for optimal design of water distribution networks.\" "
        "Journal of Hydroinformatics, 12(1), 66-82.",
        "Storn, R., and Price, K. (1995). \"Differential Evolution - "
        "A Simple and Efficient Adaptive Scheme for Global "
        "Optimization over Continuous Space.\" Technical Report "
        "TR-95-012, ICSI, Berkeley, CA.",
    ],
    "notes": (
        "Strategy is fixed at rand/1/bin; DEConfig.Strategy is "
        "ignored.  Infeasible designs collapse to a single effective "
        "cost (paper's fixed-penalty rule), which can reduce diversity "
        "in the early generations.  Reported success rates in the "
        "paper are 40-70% per trial depending on settings."
    ),
    "params": [
        {
            "key": "de_f",
            "name": "F",
            "display_name": "Differential weight (F)",
            "type": "float",
            "default": 0.8,
            "min": 0.0, "max": 2.0, "step": 0.05,
            "description": (
                "Scale applied to the differential vector.  Paper's "
                "recommended value is 0.8 (Table 2 and Table 6)."
            ),
        },
        {
            "key": "de_cr",
            "name": "CR",
            "display_name": "Crossover rate (CR)",
            "type": "float",
            "default": 0.4,
            "min": 0.0, "max": 1.0, "step": 0.05,
            "description": (
                "Probability that each gene is taken from the mutant. "
                "Paper's recommended value is 0.4 (Table 2 and "
                "Table 6)."
            ),
        },
    ],
}