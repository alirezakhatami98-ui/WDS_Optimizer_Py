"""Discrete PSO – mirror of runPSO.m."""

import numpy as np

from .compare import is_better_solution
from ..optimization.evaluate import evaluate_population

class PSOCancelled(Exception):
    """Raised when PSO is cancelled."""


def run_pso(problem, config,
            progress_callback=None, cancel_callback=None):
    """Run PSO and return (score, full_position_m, convergence, feasible)."""
    rng = (
        np.random.default_rng(config.Seed)
        if config.Seed is not None
        else np.random.default_rng()
    )

    N = config.NS
    MaxGen = config.MaxGen
    ND = len(problem["D"])
    NVar = len(problem["VariablePipes"])

    w = config.PSO.w
    c1 = config.PSO.c1
    c2 = config.PSO.c2

    cache = config.Cache if config.Cache is not None else {}

    # Continuous positions rounded on use
    X = rng.integers(0, ND, size=(NVar, N)).astype(float)
    V = np.zeros((NVar, N))

    PBestX = X.copy()
    PBestCost = np.full(N, np.inf)
    PBestViol = np.full(N, np.inf)

    GBestX = None
    GBestCost = np.inf
    GBestViol = np.inf

    BestX = None
    BestCost = np.inf
    BestViol = np.inf

    Conv = np.zeros(MaxGen)

    ProgressInterval = max(1, MaxGen // 50)
    ProgressFailed = False

    for G in range(MaxGen):
        if cancel_callback is not None and cancel_callback():
            raise PSOCancelled("PSO execution was cancelled by the user.")
        
        # MATLAB round() is half-away-from-zero; np.floor(x + 0.5) matches.
        X_discrete = np.clip(
            np.floor(X + 0.5).astype(int), 0, ND - 1
        )

        cost, viol, feas = evaluate_population(X_discrete, problem, cache)

        for i in range(N):
            # ---- update PBest ----
            if feas[i]:
                if PBestViol[i] > 0 or cost[i] < PBestCost[i]:
                    PBestCost[i] = cost[i]
                    PBestViol[i] = viol[i]
                    PBestX[:, i] = X_discrete[:, i]
            else:
                if PBestViol[i] > 0 and viol[i] < PBestViol[i]:
                    PBestCost[i] = cost[i]
                    PBestViol[i] = viol[i]
                    PBestX[:, i] = X_discrete[:, i]

            # ---- update GBest ----
            if feas[i]:
                if GBestViol > 0 or cost[i] < GBestCost:
                    GBestCost = cost[i]
                    GBestViol = viol[i]
                    GBestX = X_discrete[:, i].copy()
            else:
                if GBestViol > 0 and viol[i] < GBestViol:
                    GBestCost = cost[i]
                    GBestViol = viol[i]
                    GBestX = X_discrete[:, i].copy()

        # ---- velocity & position update ----
        for i in range(N):
            r1 = rng.random(NVar)
            r2 = rng.random(NVar)
            V[:, i] = (
                w * V[:, i]
                + c1 * r1 * (PBestX[:, i] - X[:, i])
                + c2 * r2 * (GBestX - X[:, i])
            )
            X[:, i] = X[:, i] + V[:, i]

        # ---- best over restarts ----
        if is_better_solution(GBestCost, GBestViol, BestCost, BestViol):
            BestX = GBestX.copy() if GBestX is not None else None
            BestCost, BestViol = GBestCost, GBestViol

        Conv[G] = BestCost

        if progress_callback is not None and not ProgressFailed:
            if (G == 0) or ((G % ProgressInterval) == 0) or (G == MaxGen - 1):
                try:
                    progress_callback({
                        "Algorithm": "PSO",
                        "Generation": G + 1,
                        "MaxGenerations": MaxGen,
                        "BestCost": float(Conv[G]),
                        "Feasible": bool(BestViol == 0.0),
                        "Convergence": Conv[: G + 1].copy(),
                    })
                except Exception as e:
                    print(f"[run_pso] progress_callback error: {e}")
                    ProgressFailed = True

    full_position = problem["InitialD"].copy()
    full_position[problem["VariablePipes"]] = problem["D"][BestX]

    return BestCost, full_position, Conv, BestViol == 0.0