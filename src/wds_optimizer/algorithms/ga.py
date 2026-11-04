"""Genetic algorithm – mirror of runGA.m (cache, callbacks omitted)."""

import numpy as np

from .compare import is_better_solution
from .selection import selection_roulette, selection_tournament
from ..optimization.evaluate import evaluate_population
from ..optimization.fitness import calculate_fitness

class GACancelled(Exception):
    """Raised when the GA is cancelled via cancel_callback.

    ``partial`` is ``(score, full_position, convergence, feasible)`` with
    the best-so-far design, or ``None`` if cancelled before the first
    generation finished.  ``convergence`` has length MaxGen, padded with
    NaN for the generations that never ran.
    """
    def __init__(self, message="GA execution was cancelled by the user.",
                 partial=None):
        super().__init__(message)
        self.partial = partial

def run_ga(problem, config,
           progress_callback=None, cancel_callback=None):
    """Run GA.

    progress_callback : callable(info_dict) or None
        Called at selected generations with keys:
        Algorithm, Generation, MaxGenerations, BestCost, Feasible,
        Convergence (np.ndarray of length G+1)
    cancel_callback : callable() -> bool or None
        If it returns True, raises ``GACancelled`` (carrying the
        best-so-far result in ``.partial`` when one exists).

    Returns
    -------
    score : float
    full_position : np.ndarray
    convergence : np.ndarray (length MaxGen)
    feasible : bool
    """
    rng = (
        np.random.default_rng(config.Seed)
        if config.Seed is not None
        else np.random.default_rng()
    )

    NS = config.NS
    MaxGen = config.MaxGen
    NVar = len(problem["VariablePipes"])
    ND = len(problem["D"])
    Pc = config.GA.Pc
    Pm = config.GA.Pm

    cache = config.Cache if config.Cache is not None else {}

    Pop = rng.integers(0, ND, size=(NVar, NS))
    cost, viol, feas = evaluate_population(Pop, problem, cache)

    Conv = np.zeros(MaxGen)
    BestSolEver, BestCostEver, BestViolEver = None, np.inf, np.inf
    BestUnfSol, BestUnfCost = None, np.inf

    GlobalSol, GlobalCost, GlobalViol = None, np.inf, np.inf

    ProgressInterval = max(1, MaxGen // 50)
    ProgressFailed = False

    for G in range(MaxGen):
        # ---- cancellation check ----
        if cancel_callback is not None and cancel_callback():
            partial = None
            if GlobalSol is not None:        # at least one generation done
                pos = problem["InitialD"].copy()
                pos[problem["VariablePipes"]] = problem["D"][GlobalSol]
                conv = np.full(MaxGen, np.nan)
                conv[:G] = Conv[:G]
                partial = (float(GlobalCost), pos, conv,
                           bool(GlobalViol == 0.0))
            raise GACancelled(
                "GA execution was cancelled by the user.", partial=partial
            )

        # ---- selection ----
        if config.GA.Selection == "tournament":
            sel = selection_tournament(
                cost, viol, NS, config.GA.TournamentSize, rng
            )
        else:
            sel = selection_roulette(
                calculate_fitness(cost, viol), NS, rng
            )
        MatingPool = Pop[:, sel].copy()

        # ---- single-point crossover on pairs ----
        NewPop = MatingPool.copy()
        for i in range(0, NS - 1, 2):
            if rng.random() < Pc and NVar > 1:
                cp = int(rng.integers(1, NVar))
                NewPop[:cp, i], NewPop[:cp, i + 1] = (
                    MatingPool[:cp, i].copy(),
                    MatingPool[:cp, i + 1].copy(),
                )
                NewPop[cp:, i], NewPop[cp:, i + 1] = (
                    MatingPool[cp:, i + 1].copy(),
                    MatingPool[cp:, i].copy(),
                )

        # ---- uniform mutation ----
        for i in range(NS):
            for j in range(NVar):
                if rng.random() < Pm:
                    NewPop[j, i] = int(rng.integers(0, ND))

        # ---- elitism ----
        if BestSolEver is not None:
            NewPop[:, 0] = BestSolEver
        elif BestUnfSol is not None:
            NewPop[:, 0] = BestUnfSol

        Pop = NewPop
        cost, viol, feas = evaluate_population(Pop, problem, cache)

        # ---- update epoch bests ----
        f_idx = np.where(feas)[0]
        if f_idx.size:
            k = int(f_idx[np.argmin(cost[f_idx])])
            if cost[k] < BestCostEver:
                BestCostEver, BestSolEver = cost[k], Pop[:, k].copy()

        iv = int(np.argmin(viol))
        if viol[iv] < BestViolEver:
            BestViolEver = viol[iv]
            BestUnfSol = Pop[:, iv].copy()
            BestUnfCost = cost[iv]

        if BestSolEver is not None:
            eCost, eViol = BestCostEver, 0.0
        else:
            eCost, eViol = BestUnfCost, BestViolEver

        if is_better_solution(eCost, eViol, GlobalCost, GlobalViol):
            GlobalSol = (
                BestSolEver if BestSolEver is not None else BestUnfSol
            ).copy()
            GlobalCost, GlobalViol = eCost, eViol

        Conv[G] = GlobalCost

        if progress_callback is not None and not ProgressFailed:
            if (G == 0) or ((G % ProgressInterval) == 0) or (G == MaxGen - 1):
                try:
                    progress_callback({
                        "Algorithm": "GA",
                        "Generation": G + 1,
                        "MaxGenerations": MaxGen,
                        "BestCost": float(Conv[G]),
                        "Feasible": bool(GlobalViol == 0.0),
                        "Convergence": Conv[: G + 1].copy(),
                    })
                except Exception as e:
                    print(f"[run_ga] progress_callback error: {e}")
                    ProgressFailed = True

    full_position = problem["InitialD"].copy()
    full_position[problem["VariablePipes"]] = problem["D"][GlobalSol]

    return GlobalCost, full_position, Conv, GlobalViol == 0.0