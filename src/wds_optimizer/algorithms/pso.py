"""Discrete PSO – mirror of runPSO.m."""

import numpy as np

from .compare import is_better_solution
from ..optimization.evaluate import evaluate_population

class PSOCancelled(Exception):
    """Raised when PSO is cancelled.

    ``partial`` is ``(score, full_position, convergence, feasible)`` with
    the best-so-far design, or ``None`` if cancelled before the first
    iteration finished.  ``convergence`` has length MaxGen, padded with
    NaN for the iterations that never ran.
    """
    def __init__(self, message="PSO execution was cancelled by the user.",
                 partial=None):
        super().__init__(message)
        self.partial = partial


def run_pso(problem, config,
            progress_callback=None, cancel_callback=None, meta=None):
    """Run PSO and return (score, full_position_m, convergence, feasible).

    ``meta`` (optional dict) receives ``meta["restart_gens"]``: the
    iteration numbers (1-based) after which a restart happened.
    """
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

    # ---- restart-on-stagnation (disabled => no extra RNG use) ----
    rs = getattr(config, "Restart", None)
    rs_on = bool(rs is not None and rs.Enabled)
    patience = int(rs.Patience) if rs_on else 0
    max_restarts = int(rs.MaxRestarts) if rs_on else 0
    keep_best = bool(rs_on and rs.KeepBest)
    perturb = bool(rs_on and rs.Mode == "perturb")
    perturb_rate = float(rs.PerturbRate) if rs_on else 0.0
    restart_gens = []
    epoch_bests = []          # best feasible cost of each epoch (inf = none)
    stall = 0
    if meta is not None:
        meta["restart_gens"] = restart_gens
        meta["epoch_bests"] = epoch_bests

    for G in range(MaxGen):
        if cancel_callback is not None and cancel_callback():
            partial = None
            epoch_bests.append(
                float(GBestCost) if GBestViol == 0.0 else float("inf"))
            if BestX is not None:            # at least one iteration done
                pos = problem["InitialD"].copy()
                pos[problem["VariablePipes"]] = problem["D"][BestX]
                conv = np.full(MaxGen, np.nan)
                conv[:G] = Conv[:G]
                partial = (float(BestCost), pos, conv,
                           bool(BestViol == 0.0))
            raise PSOCancelled(
                "PSO execution was cancelled by the user.", partial=partial
            )

        # MATLAB round() is half-away-from-zero; np.floor(x + 0.5) matches.
        X_discrete = np.clip(
            np.floor(X + 0.5).astype(int), 0, ND - 1
        )

        cost, viol, feas = evaluate_population(X_discrete, problem, cache)
        prev_gbest = (GBestCost, GBestViol)      # swarm best of this epoch

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

        epoch_improved = is_better_solution(
            GBestCost, GBestViol, prev_gbest[0], prev_gbest[1]
        )

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

        # ---- stagnation detection / restart ----
        stall = 0 if epoch_improved else stall + 1
        if (rs_on and stall >= patience
                and len(restart_gens) < max_restarts
                and (MaxGen - (G + 1)) > patience):
            restart_gens.append(G + 1)
            epoch_bests.append(
                float(GBestCost) if GBestViol == 0.0 else float("inf"))
            if perturb and BestX is not None:
                X = np.tile(BestX[:, None].astype(float), (1, N))
                mask = rng.random((NVar, N)) < perturb_rate
                X[mask] = rng.integers(0, ND, size=int(mask.sum()))
                X[:, 0] = BestX
            else:
                X = rng.integers(0, ND, size=(NVar, N)).astype(float)
                if keep_best and BestX is not None:
                    X[:, 0] = BestX
            V = np.zeros((NVar, N))
            # Fresh personal and swarm memory (otherwise particles are
            # pulled straight back to the old basin).  The best over all
            # epochs stays in BestX / BestCost / BestViol.
            PBestX = X.copy()
            PBestCost = np.full(N, np.inf)
            PBestViol = np.full(N, np.inf)
            GBestX, GBestCost, GBestViol = None, np.inf, np.inf
            stall = 0

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
                        "Restarts": len(restart_gens),
                        "RestartGens": list(restart_gens),
                    })
                except Exception as e:
                    print(f"[run_pso] progress_callback error: {e}")
                    ProgressFailed = True

    epoch_bests.append(
        float(GBestCost) if GBestViol == 0.0 else float("inf"))

    full_position = problem["InitialD"].copy()
    full_position[problem["VariablePipes"]] = problem["D"][BestX]

    return BestCost, full_position, Conv, BestViol == 0.0