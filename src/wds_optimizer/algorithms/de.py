"""Discrete Differential Evolution (DE) for pipe-diameter selection.

Classic DE works on real vectors.  As in the PSO of this project, the
population is kept as continuous positions in [0, ND-1] and rounded to the
nearest diameter index only when a design is evaluated.  Selection uses
Deb's feasibility rules (``is_better_solution``): a trial replaces its
target if it is not worse (ties are accepted, which lets the population
drift across plateaus of equal-cost designs).

Same interface as ``run_ga`` / ``run_pso``: progress/cancel callbacks,
``meta`` (restart generations, epoch bests, NFE counters) and the shared
restart-on-stagnation mechanism (``config.Restart``).
"""

import numpy as np

from .compare import is_better_solution
from ..optimization.evaluate import evaluate_population

DE_STRATEGIES = ("rand/1/bin", "best/1/bin", "current-to-best/1/bin")


class DECancelled(Exception):
    """Raised when DE is cancelled via cancel_callback.

    ``partial`` is ``(score, full_position, convergence, feasible)`` with
    the best-so-far design, or ``None`` if cancelled before the initial
    population was evaluated.  ``convergence`` has length MaxGen, padded
    with NaN for the generations that never ran.
    """
    def __init__(self, message="DE execution was cancelled by the user.",
                 partial=None):
        super().__init__(message)
        self.partial = partial


def _discretize(X, ND):
    return np.clip(np.floor(X + 0.5).astype(int), 0, ND - 1)


def _best_index(cost, viol):
    """Index of the best member under Deb's rules."""
    feasible = np.where(viol == 0.0)[0]
    if feasible.size:
        return int(feasible[np.argmin(cost[feasible])])
    return int(np.argmin(viol))


def run_de(problem, config,
           progress_callback=None, cancel_callback=None, meta=None):
    """Run DE and return (score, full_position_m, convergence, feasible).

    ``meta`` (optional dict) receives restart generations, per-epoch best
    costs and the NFE counters, exactly like ``run_ga`` / ``run_pso``.
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
    strategy = config.DE.Strategy
    if strategy not in DE_STRATEGIES:
        raise ValueError(f"Unknown DE strategy: {strategy!r}")
    if NP < 4:
        raise ValueError("DE needs a population of at least 4 members.")

    cache = config.Cache if config.Cache is not None else {}

    # ---- evaluation accounting (NFE) ----
    cache0 = len(cache)
    n_evals = 0
    nfe_best = 0

    # ---- restart-on-stagnation (disabled => no extra RNG use) ----
    rs = getattr(config, "Restart", None)
    rs_on = bool(rs is not None and rs.Enabled)
    patience = int(rs.Patience) if rs_on else 0
    max_restarts = int(rs.MaxRestarts) if rs_on else 0
    keep_best = bool(rs_on and rs.KeepBest)
    perturb = bool(rs_on and rs.Mode == "perturb")
    perturb_rate = float(rs.PerturbRate) if rs_on else 0.0
    restart_gens = []
    epoch_bests = []
    stall = 0

    def _record_meta():
        if meta is not None:
            meta["nfe"] = n_evals
            meta["n_sim"] = len(cache) - cache0
            meta["nfe_to_best"] = nfe_best

    if meta is not None:
        meta["restart_gens"] = restart_gens
        meta["epoch_bests"] = epoch_bests

    Conv = np.zeros(MaxGen)
    ProgressInterval = max(1, MaxGen // 50)
    ProgressFailed = False

    # ---- initial population (registered: it is real information) ----
    X = rng.integers(0, ND, size=(NVar, NP)).astype(float)
    cost, viol, feas = evaluate_population(_discretize(X, ND), problem, cache)
    n_evals += NP

    ib = _best_index(cost, viol)
    EpochCost, EpochViol = float(cost[ib]), float(viol[ib])
    GlobalSol = _discretize(X[:, ib], ND)
    GlobalCost, GlobalViol = EpochCost, EpochViol
    nfe_best = n_evals

    def _partial(G):
        pos = problem["InitialD"].copy()
        pos[problem["VariablePipes"]] = problem["D"][GlobalSol]
        conv = np.full(MaxGen, np.nan)
        conv[:G] = Conv[:G]
        return (float(GlobalCost), pos, conv, bool(GlobalViol == 0.0))

    idx_all = np.arange(NP)

    for G in range(MaxGen):
        # ---- cancellation check ----
        if cancel_callback is not None and cancel_callback():
            epoch_bests.append(
                float(EpochCost) if EpochViol == 0.0 else float("inf"))
            _record_meta()
            raise DECancelled(
                "DE execution was cancelled by the user.", partial=_partial(G)
            )

        ibest = _best_index(cost, viol)
        Xbest = X[:, ibest]

        # ---- mutation ----
        V = np.empty_like(X)
        for i in range(NP):
            others = idx_all[idx_all != i]
            if strategy == "rand/1/bin":
                r1, r2, r3 = rng.choice(others, size=3, replace=False)
                V[:, i] = X[:, r1] + F * (X[:, r2] - X[:, r3])
            elif strategy == "best/1/bin":
                r1, r2 = rng.choice(others, size=2, replace=False)
                V[:, i] = Xbest + F * (X[:, r1] - X[:, r2])
            else:                                  # current-to-best/1/bin
                r1, r2 = rng.choice(others, size=2, replace=False)
                V[:, i] = (X[:, i] + F * (Xbest - X[:, i])
                           + F * (X[:, r1] - X[:, r2]))

        # ---- binomial crossover (>= 1 gene from the mutant) ----
        mask = rng.random((NVar, NP)) < CR
        mask[rng.integers(0, NVar, size=NP), idx_all] = True
        U = np.where(mask, V, X)

        # ---- bound handling: re-draw out-of-range genes uniformly ----
        bad = (U < 0.0) | (U > ND - 1)
        if bad.any():
            U[bad] = rng.uniform(0.0, ND - 1, size=int(bad.sum()))

        # ---- evaluate trials and select (Deb's rules, ties accepted) ----
        cost_t, viol_t, _ = evaluate_population(
            _discretize(U, ND), problem, cache
        )
        n_evals += NP
        for i in range(NP):
            if not is_better_solution(cost[i], viol[i], cost_t[i], viol_t[i]):
                X[:, i] = U[:, i]
                cost[i], viol[i] = cost_t[i], viol_t[i]

        # ---- update epoch / global bests ----
        epoch_improved = False
        ib = _best_index(cost, viol)
        if is_better_solution(cost[ib], viol[ib], EpochCost, EpochViol):
            EpochCost, EpochViol = float(cost[ib]), float(viol[ib])
            epoch_improved = True
            if is_better_solution(EpochCost, EpochViol,
                                  GlobalCost, GlobalViol):
                GlobalCost, GlobalViol = EpochCost, EpochViol
                GlobalSol = _discretize(X[:, ib], ND)
                nfe_best = n_evals

        Conv[G] = GlobalCost

        # ---- stagnation detection / restart ----
        stall = 0 if epoch_improved else stall + 1
        if (rs_on and stall >= patience
                and len(restart_gens) < max_restarts
                and (MaxGen - (G + 1)) > patience):
            restart_gens.append(G + 1)
            epoch_bests.append(
                float(EpochCost) if EpochViol == 0.0 else float("inf"))
            if perturb:
                X = np.tile(GlobalSol[:, None].astype(float), (1, NP))
                m = rng.random((NVar, NP)) < perturb_rate
                X[m] = rng.integers(0, ND, size=int(m.sum()))
                X[:, 0] = GlobalSol
            else:
                X = rng.integers(0, ND, size=(NVar, NP)).astype(float)
                if keep_best:
                    X[:, 0] = GlobalSol
            cost, viol, feas = evaluate_population(
                _discretize(X, ND), problem, cache
            )
            n_evals += NP
            ib = _best_index(cost, viol)
            EpochCost, EpochViol = float(cost[ib]), float(viol[ib])
            if is_better_solution(EpochCost, EpochViol,
                                  GlobalCost, GlobalViol):
                GlobalCost, GlobalViol = EpochCost, EpochViol
                GlobalSol = _discretize(X[:, ib], ND)
                nfe_best = n_evals
            stall = 0

        if progress_callback is not None and not ProgressFailed:
            if (G == 0) or ((G % ProgressInterval) == 0) or (G == MaxGen - 1):
                try:
                    progress_callback({
                        "Algorithm": "DE",
                        "Generation": G + 1,
                        "MaxGenerations": MaxGen,
                        "BestCost": float(Conv[G]),
                        "Feasible": bool(GlobalViol == 0.0),
                        "Convergence": Conv[: G + 1].copy(),
                        "NFE": n_evals,
                        "Restarts": len(restart_gens),
                        "RestartGens": list(restart_gens),
                    })
                except Exception as e:
                    print(f"[run_de] progress_callback error: {e}")
                    ProgressFailed = True

    epoch_bests.append(
        float(EpochCost) if EpochViol == 0.0 else float("inf"))
    _record_meta()

    full_position = problem["InitialD"].copy()
    full_position[problem["VariablePipes"]] = problem["D"][GlobalSol]

    return GlobalCost, full_position, Conv, GlobalViol == 0.0
