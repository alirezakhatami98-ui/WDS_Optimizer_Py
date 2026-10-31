"""Genetic algorithm – mirror of runGA.m (cache, callbacks omitted)."""

import numpy as np

from .compare import is_better_solution
from .selection import selection_roulette, selection_tournament
from ..optimization.evaluate import evaluate_population
from ..optimization.fitness import calculate_fitness


def run_ga(problem, config):
    """Run GA and return (score, full_position_m, convergence, feasible)."""
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

    for G in range(MaxGen):
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

    full_position = problem["InitialD"].copy()
    full_position[problem["VariablePipes"]] = problem["D"][GlobalSol]

    return GlobalCost, full_position, Conv, GlobalViol == 0.0