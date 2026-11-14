"""Self-adaptive Differential Evolution (SADE) for pipe-diameter selection.

Implements Zheng, Zecchin & Simpson (2013), "A self-adaptive differential
evolution algorithm applied to water distribution system optimization",
J. Comput. Civ. Eng., 27(2), 148-158.

Paper features implemented here
-------------------------------
* F and CR are encoded in every individual (individual-level control):
  initial values uniform in [Fmin, Fmax] / [CRmin, CRmax]            (Eq. 1)
* DE/rand/1 mutation  V = Xa + F_i (Xb - Xc)                          (Eq. 2)
* Crossover: gene taken from the mutant if Rand <= CR_i               (Eq. 3)
* Selection with Deb's constraint tournament - NO penalty function:
  feasible beats infeasible; two feasible -> lower cost; two infeasible
  -> lower violation. The trial survives if it is not worse (<=)      (Eq. 4)
* F, CR survive with a successful individual; after a failed trial the
  individual receives freshly drawn F, CR                             (Eq. 5)
* Termination by the coefficient of variation of the population's
  objective values, Cv = s / |mean| < Tol (paper: Tol = 1e-6)         (Eq. 6)
* Decision variables are diameter indices; real values from mutation are
  rounded to the nearest index before the hydraulic analysis.

Choices where the paper is silent (documented, same as ``run_de``)
-----------------------------------------------------------------
* Xa, Xb, Xc are distinct and also different from the target Xi.
* Genes outside [0, ND-1] are re-drawn uniformly inside the bounds.
* Initial genes are drawn uniformly over the diameter indices.
* ``MaxGen`` acts as an upper cap on the generations (the paper has none);
  disable ``UseConvergence`` to always run the full MaxGen like GA/PSO/DE.

Same interface as the other algorithms: progress/cancel callbacks, ``meta``
(restart generations, epoch bests, NFE counters, adaptation diagnostics)
and the shared restart-on-stagnation mechanism. When the population
converges (Cv < Tol) and restarts are enabled and still available, the
convergence triggers a restart instead of ending the run.
"""

import numpy as np

from .compare import is_better_solution
from .de import _best_index, _discretize
from ..optimization.evaluate import evaluate_population


class SADECancelled(Exception):
    """Raised when SADE is cancelled via cancel_callback.

    ``partial`` is ``(score, full_position, convergence, feasible)`` with
    the best-so-far design, or ``None`` if cancelled before the initial
    population was evaluated.  ``convergence`` has length MaxGen, padded
    with NaN for the generations that never ran.
    """
    def __init__(self, message="SADE execution was cancelled by the user.",
                 partial=None):
        super().__init__(message)
        self.partial = partial


def run_sade(problem, config,
             progress_callback=None, cancel_callback=None, meta=None):
    """Run SADE and return (score, full_position_m, convergence, feasible).

    The returned convergence array is shorter than MaxGen when the run
    stopped early because the population converged.
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
    sc = config.SADE
    Fmin, Fmax = float(sc.Fmin), float(sc.Fmax)
    CRmin, CRmax = float(sc.CRmin), float(sc.CRmax)
    use_conv = bool(sc.UseConvergence)
    tol = float(sc.Tol)
    stag_max = int(getattr(sc, "StagnationMax", 0))
    if NP < 4:
        raise ValueError("SADE needs a population of at least 4 members.")
    if not (0.0 <= Fmin <= Fmax):
        raise ValueError("SADE needs 0 <= Fmin <= Fmax.")
    if not (0.0 <= CRmin <= CRmax <= 1.0):
        raise ValueError("SADE needs 0 <= CRmin <= CRmax <= 1.")

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

    # ---- adaptation diagnostics (paper Fig. 3-style Cv curve etc.) ----
    cv_hist, f_hist, cr_hist = [], [], []
    state = {"converged": False}

    def _record_meta():
        if meta is not None:
            meta["nfe"] = n_evals
            meta["n_sim"] = len(cache) - cache0
            meta["nfe_to_best"] = nfe_best
            meta["extra"] = {
                "cv": list(cv_hist), "f_mean": list(f_hist),
                "cr_mean": list(cr_hist),
                "converged": bool(state["converged"]),
                "stopped_by_stagnation": bool(
                    state.get("stopped_by_stagnation", False)),
                "generations": len(cv_hist),
            }

    if meta is not None:
        meta["restart_gens"] = restart_gens
        meta["epoch_bests"] = epoch_bests

    Conv = np.zeros(MaxGen)
    ProgressInterval = max(1, MaxGen // 50)
    ProgressFailed = False

    def _draw_params(n):
        return (rng.uniform(Fmin, Fmax, size=n),
                rng.uniform(CRmin, CRmax, size=n))

    # ---- initial population with individual F and CR (Eq. 1) ----
    X = rng.integers(0, ND, size=(NVar, NP)).astype(float)
    Fvec, CRvec = _draw_params(NP)
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
    last_G = -1

    for G in range(MaxGen):
        # ---- cancellation check ----
        if cancel_callback is not None and cancel_callback():
            epoch_bests.append(
                float(EpochCost) if EpochViol == 0.0 else float("inf"))
            _record_meta()
            raise SADECancelled(
                "SADE execution was cancelled by the user.",
                partial=_partial(G),
            )

        # ---- mutation, individual F_i (Eq. 2: DE/rand/1) ----
        V = np.empty_like(X)
        for i in range(NP):
            r1, r2, r3 = rng.choice(idx_all[idx_all != i], size=3,
                                    replace=False)
            V[:, i] = X[:, r1] + Fvec[i] * (X[:, r2] - X[:, r3])

        # ---- crossover, individual CR_i (Eq. 3) ----
        # Binomial DE guarantees that at least one gene comes from the
        # mutant (jrand).  Without this, a trial can be an exact copy of
        # its target - probability (1-CR)^NVar, which is ~11% for NYTP
        # with CR~0.1.  An identical trial is scored "not worse" and the
        # parent's F/CR are never refreshed, stalling self-adaptation.
        mask = rng.random((NVar, NP)) <= CRvec[None, :]
        jrand = rng.integers(0, NVar, size=NP)
        mask[jrand, idx_all] = True
        U = np.where(mask, V, X)

        # ---- bound handling: clip out-of-range genes ----
        # Matches the MATLAB implementation (min(max(V,1),ND)) and keeps
        # the direction that the mutation produced; re-drawing discards
        # that information.  The paper truncates the mutant to the nearest
        # diameter, so clipping to the boundary is the natural choice.
        U = np.clip(U, 0.0, float(ND - 1))

        # ---- evaluate trials; Deb's constraint tournament (Eq. 4) ----
        cost_t, viol_t, _ = evaluate_population(
            _discretize(U, ND), problem, cache
        )
        n_evals += NP
        success = np.zeros(NP, dtype=bool)
        for i in range(NP):
            success[i] = not is_better_solution(
                cost[i], viol[i], cost_t[i], viol_t[i])
        X[:, success] = U[:, success]
        cost[success] = cost_t[success]
        viol[success] = viol_t[success]

        # ---- F / CR self-adaptation (Eq. 5) ----
        n_fail = int((~success).sum())
        if n_fail:
            newF, newCR = _draw_params(n_fail)
            Fvec[~success] = newF
            CRvec[~success] = newCR

        # ---- convergence measure (Eq. 6) ----
        # As in the reference MATLAB implementation, Cv is taken over
        # feasible members only.  Infeasible designs are typically cheaper
        # (small pipes) and pull the mean down, inflating Cv and delaying
        # convergence.  With fewer than two feasible members Cv cannot be
        # meaningfully defined; report +inf so the run does not stop.
        feas_mask = (viol == 0.0)
        if int(feas_mask.sum()) >= 2:
            c_feas = cost[feas_mask]
            m = float(np.mean(c_feas))
            cv = (float(np.std(c_feas, ddof=1)) / abs(m)
                  if m != 0.0 else 0.0)
        else:
            cv = float("inf")
        cv_hist.append(cv)
        f_hist.append(float(np.mean(Fvec)))
        cr_hist.append(float(np.mean(CRvec)))

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
        last_G = G

        # ---- restart: stagnation (as GA/PSO/DE) or convergence ----
        stall = 0 if epoch_improved else stall + 1
        converged = use_conv and cv < tol
        stagnated = (stag_max > 0 and stall >= stag_max)
        can_restart = (rs_on and len(restart_gens) < max_restarts
                       and (MaxGen - (G + 1)) > patience)
        if can_restart and (stall >= patience or converged):
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
            Fvec, CRvec = _draw_params(NP)
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
            converged = False          # a restart consumed the convergence
            stagnated = False          # restart resets the stall counter

        if progress_callback is not None and not ProgressFailed:
            if (G == 0) or ((G % ProgressInterval) == 0) \
                    or (G == MaxGen - 1) or converged:
                try:
                    progress_callback({
                        "Algorithm": "SADE",
                        "Generation": G + 1,
                        "MaxGenerations": MaxGen,
                        "BestCost": float(Conv[G]),
                        "Feasible": bool(GlobalViol == 0.0),
                        "Convergence": Conv[: G + 1].copy(),
                        "NFE": n_evals,
                        "Cv": cv,
                        "Restarts": len(restart_gens),
                        "RestartGens": list(restart_gens),
                    })
                except Exception as e:
                    print(f"[run_sade] progress_callback error: {e}")
                    ProgressFailed = True

        # ---- termination ----
        if converged:
            state["converged"] = True
            break
        if stagnated:
            state["stopped_by_stagnation"] = True
            break

    epoch_bests.append(
        float(EpochCost) if EpochViol == 0.0 else float("inf"))
    _record_meta()

    full_position = problem["InitialD"].copy()
    full_position[problem["VariablePipes"]] = problem["D"][GlobalSol]

    return GlobalCost, full_position, Conv[: last_G + 1], GlobalViol == 0.0

ALGORITHM_META = {
    "short_id": "SADE",
    "display_name": "SADE",
    "category": "Differential Evolution",
    "paper_faithful": True,
    "description": (
        "Self-adaptive DE (Zheng, Zecchin & Simpson 2013): F and CR "
        "encoded in every individual and self-adapted by evolution, "
        "DE/rand/1/bin mutation, binomial crossover with jrand, "
        "Deb's constraint tournament, and coefficient-of-variation "
        "termination on feasible members."
    ),
    "references": [
        "Zheng, F., Zecchin, A. C., and Simpson, A. R. (2013). "
        "\"A self-adaptive differential evolution algorithm applied "
        "to water distribution system optimization.\" Journal of "
        "Computing in Civil Engineering, 27(2), 148-158.",
    ],
    "notes": (
        "Faithful reproduction of Zheng et al. (2013), with one "
        "addition: a StagnationMax safeguard.  The paper relies "
        "only on Cv < Tol to terminate."
    ),
    "params": [
        {
            "key": "sade_fmin",
            "name": "Fmin",
            "display_name": "F minimum",
            "type": "float",
            "default": 0.1,
            "min": 0.0, "max": 2.0, "step": 0.05,
            "description": (
                "Lower bound of the mutation weighting factor range."
            ),
        },
        {
            "key": "sade_fmax",
            "name": "Fmax",
            "display_name": "F maximum",
            "type": "float",
            "default": 0.9,
            "min": 0.0, "max": 2.0, "step": 0.05,
            "description": (
                "Upper bound of the mutation weighting factor range."
            ),
        },
        {
            "key": "sade_crmin",
            "name": "CRmin",
            "display_name": "CR minimum",
            "type": "float",
            "default": 0.1,
            "min": 0.0, "max": 1.0, "step": 0.05,
            "description": (
                "Lower bound of the crossover rate range."
            ),
        },
        {
            "key": "sade_crmax",
            "name": "CRmax",
            "display_name": "CR maximum",
            "type": "float",
            "default": 0.9,
            "min": 0.0, "max": 1.0, "step": 0.05,
            "description": (
                "Upper bound of the crossover rate range."
            ),
        },
        {
            "key": "sade_conv",
            "name": "UseConvergence",
            "display_name": "Convergence-based stop",
            "type": "bool",
            "default": True,
            "description": (
                "Stop the run when Cv falls below the tolerance."
            ),
        },
        {
            "key": "sade_tol",
            "name": "Tol",
            "display_name": "Cv tolerance",
            "type": "float",
            "default": 1e-6,
            "min": 1e-9, "max": 1e-3, "step": 1e-7,
            "description": (
                "Coefficient-of-variation threshold for the "
                "convergence stop."
            ),
        },
        {
            "key": "sade_stag",
            "name": "StagnationMax",
            "display_name": "Stagnation safeguard",
            "type": "int",
            "default": 300,
            "min": 0, "max": 10000, "step": 10,
            "description": (
                "Stop after this many generations without "
                "improvement (0 = off)."
            ),
        },
    ],
}
