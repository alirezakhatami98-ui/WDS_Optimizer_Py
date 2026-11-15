"""Genetic algorithm – mirror of runGA.m (cache, callbacks omitted)."""

import numpy as np

from ..optimization.evaluate import evaluate_population
from ..optimization.policies import get_policy

# =====================================================================
#  GA-internal helpers (previously in separate modules)
# =====================================================================

def _selection_tournament(cost, viol, n, tournament_size, rng, policy):
    chosen = []
    for _ in range(n):
        contestants = rng.integers(0, len(cost), size=tournament_size)
        best = contestants[0]
        for c in contestants[1:]:
            if policy(cost[c], viol[c], cost[best], viol[best]):
                best = c
        chosen.append(int(best))
    return np.asarray(chosen, dtype=int)


def selection_roulette(fitness, n_select, rng):
    """Standard roulette-wheel selection with replacement."""
    prob = fitness / fitness.sum()
    cum = np.cumsum(prob)
    r = rng.random(n_select)
    return np.clip(np.searchsorted(cum, r, side="left"), 0, len(cum) - 1)


def _calculate_fitness(cost, viol):
    """Violation-weighted fitness for roulette selection.

    Feasible designs get 1/cost; infeasible get 1/((1+viol)*cost).
    This is Deb's-style fitness shaping and is preserved verbatim for
    backward compatibility.

    **Known limitation:** this function is not parameterized by the
    configured constraint policy.  If you select ``fixed_penalty_prefers``
    or ``simple_cost_only_prefers`` **and** use roulette selection, the
    fitness shaping here still applies a Deb's-style violation penalty.
    For full policy sensitivity, use tournament selection (``ga_sel =
    "tournament"``).  This is documented in ``docs/algorithms.md``.
    """
    scale = np.where(viol == 0, 1.0, 1.0 / (1.0 + viol))
    return scale / np.maximum(cost, 1e-9)

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
           progress_callback=None, cancel_callback=None, meta=None):
    """Run GA.

    progress_callback : callable(info_dict) or None
        Called at selected generations with keys:
        Algorithm, Generation, MaxGenerations, BestCost, Feasible,
        Convergence (np.ndarray of length G+1)
    meta : dict or None
        If given, ``meta["restart_gens"]`` is filled with the generation
        numbers (1-based) after which a restart happened.
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
    policy = get_policy(config.constraint_policy)
    MaxGen = config.MaxGen
    NVar = len(problem["VariablePipes"])
    ND = len(problem["D"])
    Pc = config.GA.Pc
    Pm = config.GA.Pm

    cache = config.Cache if config.Cache is not None else {}

    # ---- evaluation accounting (NFE) ----
    cache0 = len(cache)       # unique simulations = cache growth in this run
    n_evals = 0               # candidate evaluations requested
    nfe_best = 0              # NFE at which the final best was first found

    def _record_meta():
        if meta is not None:
            meta["nfe"] = n_evals
            meta["n_sim"] = len(cache) - cache0
            meta["nfe_to_best"] = nfe_best

    Pop = rng.integers(0, ND, size=(NVar, NS))
    cost, viol, feas = evaluate_population(Pop, problem, cache)
    n_evals += NS

    Conv = np.zeros(MaxGen)
    BestSolEver, BestCostEver, BestViolEver = None, np.inf, np.inf
    BestUnfSol, BestUnfCost = None, np.inf

    GlobalSol, GlobalCost, GlobalViol = None, np.inf, np.inf

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
        # ---- cancellation check ----
        if cancel_callback is not None and cancel_callback():
            partial = None
            epoch_bests.append(float(BestCostEver))
            _record_meta()
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
            sel = _selection_tournament(
                cost, viol, NS, config.GA.TournamentSize, rng, policy
            )
        else:
            sel = selection_roulette(
                _calculate_fitness(cost, viol), NS, rng
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
        n_evals += NS

        # ---- update epoch bests ----
        epoch_improved = False
        f_idx = np.where(feas)[0]
        if f_idx.size:
            k = int(f_idx[np.argmin(cost[f_idx])])
            if cost[k] < BestCostEver:
                BestCostEver, BestSolEver = cost[k], Pop[:, k].copy()
                epoch_improved = True

        iv = int(np.argmin(viol))
        if viol[iv] < BestViolEver:
            BestViolEver = viol[iv]
            BestUnfSol = Pop[:, iv].copy()
            BestUnfCost = cost[iv]
            epoch_improved = True

        if BestSolEver is not None:
            eCost, eViol = BestCostEver, 0.0
        else:
            eCost, eViol = BestUnfCost, BestViolEver

        if policy(eCost, eViol, GlobalCost, GlobalViol):
            GlobalSol = (
                BestSolEver if BestSolEver is not None else BestUnfSol
            ).copy()
            GlobalCost, GlobalViol = eCost, eViol
            nfe_best = n_evals

        Conv[G] = GlobalCost

        # ---- stagnation detection / restart ----
        stall = 0 if epoch_improved else stall + 1
        if (rs_on and stall >= patience
                and len(restart_gens) < max_restarts
                and (MaxGen - (G + 1)) > patience):
            restart_gens.append(G + 1)
            epoch_bests.append(float(BestCostEver))
            if perturb:
                # copies of the best design, each gene re-drawn with
                # probability `perturb_rate`; column 0 keeps the best intact
                Pop = np.tile(GlobalSol[:, None], (1, NS))
                mask = rng.random((NVar, NS)) < perturb_rate
                Pop[mask] = rng.integers(0, ND, size=int(mask.sum()))
                Pop[:, 0] = GlobalSol
            else:
                Pop = rng.integers(0, ND, size=(NVar, NS))
            BestSolEver, BestCostEver, BestViolEver = None, np.inf, np.inf
            BestUnfSol, BestUnfCost = None, np.inf
            if keep_best or perturb:
                Pop[:, 0] = GlobalSol
                if GlobalViol == 0.0:
                    BestSolEver = GlobalSol.copy()
                    BestCostEver, BestViolEver = GlobalCost, 0.0
                else:
                    BestUnfSol = GlobalSol.copy()
                    BestUnfCost, BestViolEver = GlobalCost, GlobalViol
            cost, viol, feas = evaluate_population(Pop, problem, cache)
            n_evals += NS
            stall = 0

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
                        "NFE": n_evals,
                        "Restarts": len(restart_gens),
                        "RestartGens": list(restart_gens),
                    })
                except Exception as e:
                    print(f"[run_ga] progress_callback error: {e}")
                    ProgressFailed = True

    epoch_bests.append(float(BestCostEver))
    _record_meta()

    full_position = problem["InitialD"].copy()
    full_position[problem["VariablePipes"]] = problem["D"][GlobalSol]

    return GlobalCost, full_position, Conv, GlobalViol == 0.0

# =====================================================================
#  Algorithm metadata (consumed by the GUI, CLI, and plugin system)
# =====================================================================

ALGORITHM_META = {
    "short_id": "GA",
    "display_name": "GA",
    "category": "Genetic Algorithm",
    "paper_faithful": False,
    "description": (
        "Standard integer-coded GA with tournament or roulette "
        "selection, one-point crossover, uniform-reset mutation, "
        "and one elite."
    ),
    "references": [
        "Holland, J. H. (1975). Adaptation in Natural and Artificial "
        "Systems. University of Michigan Press.",
        "Goldberg, D. E. (1989). Genetic Algorithms in Search, "
        "Optimization and Machine Learning. Addison-Wesley.",
        "Simpson, Dandy & Murphy (1994) is the closest published "
        "Simple-GA benchmark reference; see docs/algorithms.md.",
    ],
    "notes": (
        "NOT a faithful reproduction of Simpson et al. (1994) "
        "(binary coding + bitwise mutation) or Dandy et al. (1996) "
        "(adds Gray codes + adjacency mutation + variable power "
        "scaling).  Uses integer coding and uniform-reset mutation. "
        "See docs/algorithms.md."
    ),
    "constraint_policy": "deb_prefers",
    "params": [
        {
            "key": "ga_pc",
            "name": "Pc",
            "display_name": "Crossover probability (Pc)",
            "type": "float",
            "default": 0.8,
            "min": 0.0, "max": 1.0, "step": 0.05,
            "maps_to": ("GA", "Pc"),
            "description": (
                "Probability that a mating pair undergoes crossover."
            ),
        },
        {
            "key": "ga_pm",
            "name": "Pm",
            "display_name": "Mutation probability (Pm)",
            "type": "float",
            "default": 0.03,
            "min": 0.0, "max": 1.0, "step": 0.01,
            "maps_to": ("GA", "Pm"),
            "description": (
                "Per-gene probability of resetting to a random "
                "diameter index."
            ),
        },
        {
            "key": "ga_sel",
            "name": "Selection",
            "display_name": "Selection scheme",
            "type": "choice",
            "default": "tournament",
            "choices": ["tournament", "roulette"],
            "maps_to": ("GA", "Selection"),
            "description": "Parent selection operator.",
        },
    ],
}