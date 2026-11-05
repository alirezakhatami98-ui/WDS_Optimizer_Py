"""Multi-run experiment orchestration (mirror of runExperiment.m)."""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from ..algorithms.config import AlgorithmConfig, build_algorithm_config
from ..algorithms.ga import GACancelled, run_ga
from ..algorithms.pso import PSOCancelled, run_pso
from ..algorithms.de import DECancelled, run_de


@dataclass
class RunRecord:
    algorithm: str
    run_index: int
    seed: int
    score: float = float("nan")
    position: Optional[np.ndarray] = None
    feasible: bool = False
    execution_time: float = float("nan")
    convergence: Optional[np.ndarray] = None
    # Success | Partial (stopped, best-so-far kept) | Failed | Cancelled | NotStarted
    status: str = "NotStarted"
    error_message: str = ""
    restart_gens: list = field(default_factory=list)   # 1-based generations
    epoch_bests: list = field(default_factory=list)    # best cost per epoch
    nfe: int = 0            # candidate evaluations requested (pop x gens ...)
    n_sim: int = 0          # hydraulic simulations actually run (cache misses)
    nfe_to_best: int = 0    # NFE when the final best was first found


@dataclass
class ExperimentResult:
    experiment_id: str
    timestamp: datetime
    number_of_runs: int
    algorithms: list
    experiment_seed: int
    NS: int
    MaxGen: int
    Pmin: float
    Vmax: float
    NP: int
    variable_pipes: np.ndarray
    fixed_pipes: np.ndarray
    runs: dict = field(default_factory=dict)   # {algo: [RunRecord, ...]}
    cancelled: bool = False                    # True if stopped by the user
    restart: dict = field(default_factory=dict)  # restart settings used


def run_experiment(
    problem,
    base_config: AlgorithmConfig,
    *,
    number_of_runs: int,
    algorithms: list,
    experiment_seed: int,
    progress_callback: Callable = None,
    cancel_callback: Callable = None,
) -> ExperimentResult:
    """Execute each requested algorithm ``number_of_runs`` times.

    Parameters
    ----------
    problem : dict
    base_config : AlgorithmConfig
        NS, MaxGen, GA/PSO parameters.
    number_of_runs : int
    algorithms : list[str]
        Subset of ``["GA", "PSO", "DE"]``.
    experiment_seed : int
    progress_callback : callable(info), optional
        Forwarded to each algorithm. ``info`` gains ``RunIndex``,
        ``NumberOfRuns``, ``RunLabel``, ``Algorithm``.
    cancel_callback : callable() -> bool, optional
    """
    # Normalize and order: GA, PSO, DE
    selected = [a for a in ("GA", "PSO", "DE") if a in algorithms]
    if not selected:
        raise ValueError("At least one of GA/PSO/DE must be selected.")

    # Generate per-run seeds (unique)
    rng = np.random.default_rng(experiment_seed)
    run_seeds = rng.integers(0, 2**32 - 1, size=number_of_runs, dtype=np.int64)
    run_seeds = np.unique(run_seeds)
    while len(run_seeds) < number_of_runs:
        extra = rng.integers(0, 2**32 - 1, size=number_of_runs - len(run_seeds))
        run_seeds = np.unique(np.concatenate([run_seeds, extra]))
    run_seeds = run_seeds[:number_of_runs]

    result = ExperimentResult(
        experiment_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        timestamp=datetime.now(),
        number_of_runs=number_of_runs,
        algorithms=selected,
        experiment_seed=experiment_seed,
        NS=base_config.NS,
        MaxGen=base_config.MaxGen,
        Pmin=problem["Pmin"],
        Vmax=problem["Vmax"],
        NP=problem["NP"],
        variable_pipes=np.asarray(problem["VariablePipes"]),
        fixed_pipes=np.asarray(problem["FixedPipes"]),
        restart=_restart_dict(base_config),
    )

    for algo in selected:
        runs = []

        for r in range(number_of_runs):
            # Cancellation between runs
            if cancel_callback is not None and cancel_callback():
                runs.append(RunRecord(
                    algorithm=algo, run_index=r + 1, seed=int(run_seeds[r]),
                    status="Cancelled",
                    error_message="Stopped before this run started.",
                ))
                continue

            cfg = build_algorithm_config(base_config.NS, base_config.MaxGen)
            cfg.GA.Pc = base_config.GA.Pc
            cfg.GA.Pm = base_config.GA.Pm
            cfg.GA.Selection = base_config.GA.Selection
            cfg.GA.TournamentSize = base_config.GA.TournamentSize
            cfg.PSO.w = base_config.PSO.w
            cfg.PSO.c1 = base_config.PSO.c1
            cfg.PSO.c2 = base_config.PSO.c2
            if getattr(base_config, "DE", None) is not None:
                cfg.DE = replace(base_config.DE)
            if getattr(base_config, "Restart", None) is not None:
                cfg.Restart = replace(base_config.Restart)
            cfg.Seed = int(run_seeds[r])

            # Wrap the per-gen callback to add run-level context
            per_run_cb = None
            if progress_callback is not None:
                def per_run_cb(info, _r=r, _a=algo):
                    info = dict(info)
                    info["RunIndex"] = _r + 1
                    info["NumberOfRuns"] = number_of_runs
                    info["RunLabel"] = f"{_a} run {_r+1}/{number_of_runs}"
                    progress_callback(info)

            import time
            meta = {}                      # filled by the algorithm
            t0 = time.perf_counter()
            try:
                algo_fn = {"GA": run_ga, "PSO": run_pso, "DE": run_de}[algo]
                score, pos, conv, feas = algo_fn(
                    problem, cfg,
                    progress_callback=per_run_cb,
                    cancel_callback=cancel_callback,
                    meta=meta,
                )
                dt = time.perf_counter() - t0
                runs.append(RunRecord(
                    algorithm=algo, run_index=r + 1, seed=int(run_seeds[r]),
                    score=float(score), position=pos, feasible=bool(feas),
                    execution_time=dt, convergence=conv, status="Success",
                    restart_gens=list(meta.get("restart_gens", [])),
                    epoch_bests=list(meta.get("epoch_bests", [])),
                    nfe=int(meta.get("nfe", 0)),
                    n_sim=int(meta.get("n_sim", 0)),
                    nfe_to_best=int(meta.get("nfe_to_best", 0)),
                ))
            except (GACancelled, PSOCancelled, DECancelled) as e:
                # If the algorithm attached its best-so-far
                # (score, position, convergence, feasible) keep it.
                partial = getattr(e, "partial", None)
                if partial is not None:
                    p_score, p_pos, p_conv, p_feas = partial
                    runs.append(RunRecord(
                        algorithm=algo, run_index=r + 1,
                        seed=int(run_seeds[r]),
                        score=float(p_score), position=p_pos,
                        feasible=bool(p_feas),
                        execution_time=time.perf_counter() - t0,
                        convergence=p_conv, status="Partial",
                        error_message="Stopped by user; best-so-far kept.",
                        restart_gens=list(meta.get("restart_gens", [])),
                        epoch_bests=list(meta.get("epoch_bests", [])),
                        nfe=int(meta.get("nfe", 0)),
                        n_sim=int(meta.get("n_sim", 0)),
                        nfe_to_best=int(meta.get("nfe_to_best", 0)),
                    ))
                else:
                    runs.append(RunRecord(
                        algorithm=algo, run_index=r + 1,
                        seed=int(run_seeds[r]),
                        status="Cancelled", error_message=str(e),
                    ))
                break
            except Exception as e:
                runs.append(RunRecord(
                    algorithm=algo, run_index=r + 1, seed=int(run_seeds[r]),
                    status="Failed", error_message=f"{type(e).__name__}: {e}",
                ))
                print(f"[run_experiment] {algo} run {r+1} failed: {e}")

        # Complete NotStarted runs after a cancellation
        while len(runs) < number_of_runs:
            idx = len(runs)
            runs.append(RunRecord(
                algorithm=algo, run_index=idx + 1, seed=int(run_seeds[idx]),
                status="NotStarted", error_message="Experiment stopped.",
            ))

        result.runs[algo] = runs

    result.cancelled = any(
        r.status in ("Cancelled", "NotStarted", "Partial")
        for runs in result.runs.values() for r in runs
    )
    return result


def _restart_dict(base_config) -> dict:
    rs = getattr(base_config, "Restart", None)
    if rs is None:
        return {"enabled": False}
    return {
        "enabled": bool(rs.Enabled),
        "patience": int(rs.Patience),
        "max_restarts": int(rs.MaxRestarts),
        "mode": str(rs.Mode),
        "keep_best": bool(rs.KeepBest),
        "perturb_rate": float(rs.PerturbRate),
    }


def summarize_experiment(result: ExperimentResult) -> dict:
    """Return {algo: {'n_ok': int, 'n_feasible': int,
                     'mean': float, 'std': float, 'best': float}}."""
    summary = {}
    for algo, runs in result.runs.items():
        ok = [r for r in runs if r.status == "Success"]
        feas = [r for r in ok if r.feasible]
        costs = np.array([r.score for r in feas]) if feas else np.array([])
        costs_all = np.array([r.score for r in ok]) if ok else np.array([])
        summary[algo] = {
            "n_success": len(ok),
            "n_feasible": len(feas),
            "mean": float(costs.mean()) if costs.size else float("nan"),
            "std":  float(costs.std(ddof=1)) if costs.size > 1 else 0.0,
            "best": float(costs.min()) if costs.size else float("nan"),
            "best_overall": float(costs_all.min()) if costs_all.size else float("nan"),
            "nfe_mean": (float(np.mean([r.nfe for r in ok if r.nfe > 0]))
                         if any(r.nfe > 0 for r in ok) else float("nan")),
        }
    return summary