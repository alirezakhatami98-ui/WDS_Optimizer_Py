"""Parameter sensitivity sweep."""

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

from ..experiments.runner import (
    ExperimentResult, run_experiment,
)
from .config import apply_sweep_parameter


@dataclass
class SweepPoint:
    value: float
    status: str
    best_score: float
    mean_score: float
    std_score: float
    feasible_pct: float
    mean_time: float
    experiment: Optional[ExperimentResult] = None


@dataclass
class SweepResult:
    parameter: str
    values: list = field(default_factory=list)
    points: list = field(default_factory=list)
    algorithms: list = field(default_factory=list)
    n_runs: int = 1
    experiment_seed: int = 1
    cancelled: bool = False          # stopped by the user before the end


def run_parameter_sweep(
    problem, base_cfg,
    *,
    parameter_name: str,
    values: list,
    algorithms: list,
    n_runs: int,
    experiment_seed: int,
    progress_callback: Callable = None,
    cancel_callback: Callable = None,
) -> SweepResult:
    points = []
    n_total = len(values)
    cancelled = False

    for i, v in enumerate(values):
        if cancel_callback and cancel_callback():
            cancelled = True
            break

        cfg, prob, n = apply_sweep_parameter(
            base_cfg, problem, parameter_name, v, n_runs
        )
        # Make sure the restart settings from the sidebar survive
        # apply_sweep_parameter (unless the swept parameter is a restart one).
        if (getattr(base_cfg, "Restart", None) is not None
                and not parameter_name.startswith("Restart")):
            cfg.Restart = replace(base_cfg.Restart)
        seed = int(v) if parameter_name == "ExperimentSeed" else experiment_seed

        if progress_callback:
            progress_callback({
                "Phase": "StartingPoint",
                "PointIndex": i + 1,
                "TotalPoints": n_total,
                "ParameterName": parameter_name,
                "ParameterValue": v,
            })

        def _inner(info, _i=i, _v=v):
            if progress_callback:
                progress_callback({
                    "Phase": "RunProgress",
                    "PointIndex": _i + 1,
                    "TotalPoints": n_total,
                    "ParameterValue": _v,
                    "ProgressInfo": info,
                })

        try:
            exp = run_experiment(
                prob, cfg,
                number_of_runs=n,
                algorithms=algorithms,
                experiment_seed=seed,
                progress_callback=_inner,
                cancel_callback=cancel_callback,
            )
        except Exception as e:
            print(f"[run_parameter_sweep] point {v} failed: {e}")
            points.append(SweepPoint(
                value=float(v), status="Failed",
                best_score=float("inf"), mean_score=float("inf"),
                std_score=0.0, feasible_pct=0.0, mean_time=0.0,
            ))
            continue

        costs, times, n_feas, n_total_runs = [], [], 0, 0
        for runs in exp.runs.values():
            for r in runs:
                if r.status != "Success":
                    continue
                n_total_runs += 1
                times.append(r.execution_time)
                if r.feasible:
                    costs.append(r.score)
                    n_feas += 1

        if costs:
            best = float(min(costs))
            mean = float(np.mean(costs))
            std = float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0
        else:
            best = float("inf")
            mean = float("inf")
            std = 0.0

        feas_pct = 100.0 * n_feas / max(n_total_runs, 1)
        mean_t = float(np.mean(times)) if times else 0.0

        # A point interrupted by Cancel has fewer runs than requested:
        # keep it, but flag it so it is not mistaken for a complete point.
        point_incomplete = bool(exp.cancelled) or (
            n_total_runs < n * len(exp.algorithms)
            and cancel_callback is not None and cancel_callback()
        )
        if point_incomplete:
            cancelled = True
            status = "Partial" if n_total_runs > 0 else "Cancelled"
        else:
            status = "Success" if costs else "AllInfeasible"

        points.append(SweepPoint(
            value=float(v),
            status=status,
            best_score=best, mean_score=mean, std_score=std,
            feasible_pct=feas_pct, mean_time=mean_t,
            experiment=exp,
        ))

        if progress_callback:
            progress_callback({
                "Phase": "PointDone",
                "PointIndex": i + 1,
                "TotalPoints": n_total,
                "BestScore": best,
                "BestFeasible": bool(costs),
            })

        if cancelled:
            break

    return SweepResult(
        parameter=parameter_name,
        values=list(values),
        points=points,
        algorithms=list(algorithms),
        n_runs=n_runs,
        experiment_seed=experiment_seed,
        cancelled=cancelled,
    )