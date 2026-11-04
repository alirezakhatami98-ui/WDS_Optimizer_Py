"""Multi-run experiment orchestration (mirror of runExperiment.m)."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from ..algorithms.config import AlgorithmConfig, build_algorithm_config
from ..algorithms.ga import GACancelled, run_ga
from ..algorithms.pso import PSOCancelled, run_pso


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
        Subset of ``["GA", "PSO"]``.
    experiment_seed : int
    progress_callback : callable(info), optional
        Forwarded to each algorithm. ``info`` gains ``RunIndex``,
        ``NumberOfRuns``, ``RunLabel``, ``Algorithm``.
    cancel_callback : callable() -> bool, optional
    """
    # Normalize and order: GA first, then PSO
    selected = [a for a in ("GA", "PSO") if a in algorithms]
    if not selected:
        raise ValueError("At least one of GA/PSO must be selected.")

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
            t0 = time.perf_counter()
            try:
                if algo == "GA":
                    score, pos, conv, feas = run_ga(
                        problem, cfg,
                        progress_callback=per_run_cb,
                        cancel_callback=cancel_callback,
                    )
                else:
                    score, pos, conv, feas = run_pso(
                        problem, cfg,
                        progress_callback=per_run_cb,
                        cancel_callback=cancel_callback,
                    )
                dt = time.perf_counter() - t0
                runs.append(RunRecord(
                    algorithm=algo, run_index=r + 1, seed=int(run_seeds[r]),
                    score=float(score), position=pos, feasible=bool(feas),
                    execution_time=dt, convergence=conv, status="Success",
                ))
            except (GACancelled, PSOCancelled) as e:
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
        }
    return summary