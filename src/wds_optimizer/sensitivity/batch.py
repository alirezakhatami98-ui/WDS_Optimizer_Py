"""Multi-parameter sensitivity batch runner.

Executes one experiment per (parameter, value) pair over a plan of
rows.  Each row specifies a parameter, the algorithm that uses it,
and the values to sweep.  Rows are processed sequentially; each
(parameter, value) becomes one BatchPoint.

The batch is a thin composition of the single-sweep machinery in
runner.py: _clone_config, _apply_sensitivity_param, and
run_experiment.  No new algorithm-side behaviour; the batch only
orchestrates.

Recommendations
---------------
After all points of a row finish, a recommended value is computed:
- "lowest_mean" — value with lowest mean score among successful points
- "flat_line" — all successful means within 1% of each other (the
  parameter has no measurable effect in this regime; no recommendation)
- "high_variance" — best value has std > 50% of mean (worth noting)
- "no_successful_points" — nothing to recommend
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..algorithms.config import AlgorithmConfig
from ..experiments.runner import ExperimentResult, run_experiment
from .config import build_sensitivity_parameters, normalize_param_name
from .runner import (
    _apply_sensitivity_param, _clone_config, _wrap_progress,
)


_FLAT_LINE_TOLERANCE = 0.01  # means within 1% are considered identical
_HIGH_VARIANCE = 0.5          # std > 50% of mean = "high variance"


@dataclass
class PlanRow:
    """One (parameter, algorithm, values) triple in a batch plan."""
    param: str
    algorithm: str
    values: list
    enabled: bool = True


@dataclass
class BatchPoint:
    """Result of one (parameter, algorithm, value) in a batch."""
    param: str
    algorithm: str
    value: Any
    experiment: Optional[ExperimentResult] = None
    status: str = "NotStarted"
    best_score: float = float("inf")
    mean_score: float = float("nan")
    std_score: float = float("nan")
    feasible_pct: float = 0.0
    mean_time: float = 0.0
    error: str = ""


@dataclass
class BatchParamResult:
    """One row's outcome: all its points + a recommendation."""
    param: str
    algorithm: str
    values: list
    points: list[BatchPoint] = field(default_factory=list)
    recommended_value: Optional[Any] = None
    recommended_mean: Optional[float] = None
    reason: str = ""
    benchmark_name: str = ""   # set by load_batch_from_jsons (may be empty)


@dataclass
class BatchSweepResult:
    n_runs: int
    experiment_seed: int
    params: list[BatchParamResult] = field(default_factory=list)
    cancelled: bool = False
    total_points: int = 0
    points_done: int = 0
    total_time_seconds: float = 0.0


# ---------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------

def validate_plan(rows: list[PlanRow]) -> None:
    """Raise ValueError on any structural problem in the plan.

    Called before execution so that a typo in a parameter name or a
    mismatch between a row's algorithm and the parameter's owning
    algorithm fails fast (same lesson as sensitivity_analysis.py's
    pre-validation).
    """
    catalog = build_sensitivity_parameters()
    enabled = [r for r in rows if r.enabled]
    if not enabled:
        raise ValueError("Plan has no enabled rows.")

    from ..algorithms.registry import list_algorithms
    valid_algos = set(list_algorithms())

    for row in enabled:
        if row.algorithm not in valid_algos:
            raise ValueError(
                f"Unknown algorithm {row.algorithm!r} in row for "
                f"{row.param!r}.  Available: {sorted(valid_algos)}"
            )
        norm = normalize_param_name(row.param)
        if norm not in catalog:
            available = sorted(catalog.keys())
            preview = (
                available[:8] + (["..."] if len(available) > 8 else [])
            )
            raise ValueError(
                f"Unknown parameter {row.param!r} "
                f"(normalized: {norm!r}).  Available: {preview}"
            )
        owning = catalog[norm].get("algorithm")
        if owning is not None and owning != row.algorithm:
            raise ValueError(
                f"Parameter {row.param!r} belongs to algorithm "
                f"{owning!r}, but row specifies {row.algorithm!r}."
            )
        if not row.values:
            raise ValueError(
                f"Row for {row.param!r} (algorithm {row.algorithm!r}) "
                f"has no values."
            )


# ---------------------------------------------------------------------
#  Recommendation logic
# ---------------------------------------------------------------------

def _compute_recommendation(pr: BatchParamResult) -> None:
    """Fill pr.recommended_value / recommended_mean / reason."""
    successes = [
        p for p in pr.points
        if p.status == "Success" and np.isfinite(p.mean_score)
    ]
    if not successes:
        pr.reason = "no_successful_points"
        return

    means = [p.mean_score for p in successes]
    lo = min(means)
    hi = max(means)
    if hi > 0 and (hi - lo) / lo < _FLAT_LINE_TOLERANCE:
        pr.reason = (
            f"flat_line (means ${lo:,.0f}..${hi:,.0f}, "
            f"<{_FLAT_LINE_TOLERANCE * 100:.0f}% spread; "
            f"parameter has no measurable effect)"
        )
        pr.recommended_value = None
        pr.recommended_mean = lo
        return

    best = min(successes, key=lambda p: p.mean_score)
    pr.recommended_value = best.value
    pr.recommended_mean = best.mean_score

    if best.std_score > _HIGH_VARIANCE * abs(best.mean_score):
        pr.reason = (
            f"lowest_mean (high variance: "
            f"std=${best.std_score:,.0f} vs mean ${best.mean_score:,.0f})"
        )
    else:
        pr.reason = "lowest_mean"

# ---------------------------------------------------------------------
#  JSON reconstruction and annotation helpers
# ---------------------------------------------------------------------

def _batch_param_from_json(data: dict) -> BatchParamResult:
    """Rebuild a BatchParamResult from a per-parameter JSON dict.

    Format matches what _save_batch_param_json writes (and what
    sensitivity_analysis.py has written since v0.7.1).
    """
    pr = BatchParamResult(
        param=data["parameter"],
        algorithm=data["algorithm"],
        values=list(data["values"]),
        benchmark_name=data.get("benchmark", ""),
    )
    for pd in data.get("points", []):
        best = pd.get("best_score")
        mean = pd.get("mean_score")
        std = pd.get("std_score")
        pr.points.append(BatchPoint(
            param=data["parameter"],
            algorithm=data["algorithm"],
            value=pd["value"],
            status=pd.get("status", "Failed"),
            best_score=float(best) if best is not None else float("inf"),
            mean_score=float(mean) if mean is not None else float("nan"),
            std_score=float(std) if std is not None else float("nan"),
            feasible_pct=float(pd.get("feasible_pct", 0.0)),
            mean_time=float(pd.get("mean_time_s", 0.0)),
            error=pd.get("error", ""),
        ))
    # Use saved recommendation if present; else recompute.
    if "recommended_value" in data:
        pr.recommended_value = data.get("recommended_value")
        pr.recommended_mean = data.get("recommended_mean")
        pr.reason = data.get("reason", "")
    else:
        _compute_recommendation(pr)
    return pr


_TIMING_ANOMALY_RATIO = 1.5   # 1.5x slower than median
_HIGH_CV = 0.03               # CV > 3% = "high variance" (per point)


def _annotations_for_param(
    pr: BatchParamResult, n_runs: int,
) -> tuple[list[str], bool]:
    """Human-readable notes for a BatchParamResult.

    Returns ``(notes, is_flat)`` where:
    - notes is a list of strings suitable for rendering in markdown
    - is_flat is True when means are within 1% (no effect detected)

    Checks performed (each contributes at most one line to notes):
      1. Failed points (any status not Success / AllInfeasible).
      2. Flat line: all successful means within 1%.
      3. Significance: best mean vs runner-up, using 2·SEM.
      4. High variance: any successful point with CV > 3%.
      5. Timing anomaly: any successful point 1.5x slower than median.
    """
    notes: list[str] = []

    successes = [
        p for p in pr.points
        if p.status == "Success" and np.isfinite(p.mean_score)
    ]
    failures = [
        p for p in pr.points
        if p.status not in ("Success", "AllInfeasible")
    ]

    # ---- 1. Failures ----
    if failures:
        first = failures[0]
        msg = f"{len(failures)} point(s) failed"
        if first.error:
            msg += f": {first.error[:100]}"
        notes.append(msg)

    if not successes:
        return notes, False

    # ---- 2. Flat line ----
    means = [p.mean_score for p in successes]
    lo, hi = min(means), max(means)
    is_flat = False
    if hi > 0 and (hi - lo) / lo < _FLAT_LINE_TOLERANCE:
        is_flat = True
        notes.append(
            f"Flat line: means ${lo:,.0f}–${hi:,.0f} "
            f"(<{_FLAT_LINE_TOLERANCE * 100:.0f}% spread; parameter "
            f"has no measurable effect)"
        )

    # ---- 3. Significance (only when not flat and 2+ successes) ----
    if not is_flat and len(successes) >= 2:
        ranked = sorted(successes, key=lambda p: p.mean_score)
        best, runner = ranked[0], ranked[1]
        gap = runner.mean_score - best.mean_score
        sqrt_n = float(np.sqrt(max(1, n_runs)))
        sem_best = best.std_score / sqrt_n
        sem_runner = runner.std_score / sqrt_n
        combined_sem = float(np.sqrt(sem_best ** 2 + sem_runner ** 2))
        two_sem = 2 * combined_sem
        if combined_sem > 0 and gap > two_sem:
            notes.append(
                f"Significant: best ${best.mean_score:,.0f} at "
                f"value={best.value} vs runner-up "
                f"${runner.mean_score:,.0f} at value={runner.value} "
                f"(gap ${gap:,.0f} > 2·SEM ${two_sem:,.0f})"
            )
        elif combined_sem > 0:
            notes.append(
                f"Not significant: gap ${gap:,.0f} between best and "
                f"runner-up vs 2·SEM ${two_sem:,.0f}"
            )

    # ---- 4. High variance (per point, CV > 3%) ----
    for p in successes:
        if p.mean_score > 0:
            cv = p.std_score / p.mean_score
            if cv > _HIGH_CV:
                notes.append(
                    f"High variance at value={p.value}: "
                    f"CV = {cv * 100:.1f}%"
                )

    # ---- 5. Timing anomaly ----
    times = [p.mean_time for p in successes if p.mean_time > 0]
    if times:
        median_t = float(np.median(times))
        for p in successes:
            if p.mean_time > _TIMING_ANOMALY_RATIO * median_t:
                notes.append(
                    f"Timing anomaly at value={p.value}: "
                    f"{p.mean_time:.1f}s vs median {median_t:.1f}s"
                )

    return notes, is_flat

# ---------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------

def run_batch_sweep(
    problem: dict,
    base_config: AlgorithmConfig,
    *,
    rows: list[PlanRow],
    n_runs: int,
    experiment_seed: int,
    progress_callback: Optional[Callable] = None,
    cancel_callback: Optional[Callable] = None,
) -> BatchSweepResult:
    """Execute all enabled rows, one experiment per (row, value).

    Progress payloads sent to ``progress_callback``:
      {"Phase": "PointDone", "PointIndex", "TotalPoints",
       "ParameterName", "Algorithm", "ParameterValue"}
      {"Phase": "ParamDone", "ParamIndex", "ParamTotal",
       "ParameterName", "Algorithm"}
    """
    validate_plan(rows)
    enabled = [r for r in rows if r.enabled]
    total_points = sum(len(r.values) for r in enabled)

    result = BatchSweepResult(
        n_runs=int(n_runs),
        experiment_seed=int(experiment_seed),
        total_points=total_points,
        points_done=0,
    )

    t_start = time.perf_counter()
    pidx = 0

    for row in enabled:
        pr = BatchParamResult(
            param=row.param,
            algorithm=row.algorithm,
            values=list(row.values),
        )
        # If the caller passed a benchmark name on `problem`, keep it.
        pr.benchmark_name = str(problem.get("_benchmark_name", ""))

        for value in row.values:
            if cancel_callback is not None and cancel_callback():
                result.cancelled = True
                break

            point = BatchPoint(
                param=row.param,
                algorithm=row.algorithm,
                value=value,
            )
            try:
                cfg = _clone_config(base_config)
                problem_iter = dict(problem)
                problem_iter["Constraints"] = copy.deepcopy(
                    problem.get("Constraints", {})
                )
                _apply_sensitivity_param(
                    cfg, problem_iter, row.param, value,
                )

                eff_n_runs = n_runs
                eff_seed = experiment_seed
                nk = normalize_param_name(row.param)
                if nk == "n_runs":
                    eff_n_runs = int(value)
                elif nk == "seed":
                    eff_seed = int(value)

                t0 = time.perf_counter()
                exp = run_experiment(
                    problem_iter, cfg,
                    number_of_runs=eff_n_runs,
                    algorithms=[row.algorithm],
                    experiment_seed=eff_seed,
                    progress_callback=_wrap_progress(
                        progress_callback, pidx, total_points, value,
                    ),
                    cancel_callback=cancel_callback,
                )
                dt = time.perf_counter() - t0

                point.experiment = exp
                point.mean_time = dt

                runs = [r for runs in exp.runs.values() for r in runs]
                feasible_runs = [
                    r for r in runs
                    if r.status in ("Success", "Partial") and r.feasible
                ]
                if feasible_runs:
                    costs = np.array([r.score for r in feasible_runs])
                    point.best_score = float(costs.min())
                    point.mean_score = float(costs.mean())
                    point.std_score = (
                        float(costs.std(ddof=1)) if len(costs) > 1 else 0.0
                    )
                    point.feasible_pct = (
                        100.0 * len(feasible_runs) / max(1, len(runs))
                    )
                    point.status = "Success"
                elif runs:
                    point.status = "AllInfeasible"
                else:
                    point.status = "Failed"
            except Exception as e:
                point.status = "Failed"
                point.error = f"{type(e).__name__}: {e}"

            pr.points.append(point)
            pidx += 1
            result.points_done = pidx

            if progress_callback is not None:
                try:
                    progress_callback({
                        "Phase": "PointDone",
                        "PointIndex": pidx,
                        "TotalPoints": total_points,
                        "ParameterName": row.param,
                        "Algorithm": row.algorithm,
                        "ParameterValue": value,
                    })
                except Exception:
                    pass

        _compute_recommendation(pr)
        result.params.append(pr)

        if progress_callback is not None:
            try:
                progress_callback({
                    "Phase": "ParamDone",
                    "ParamIndex": len(result.params),
                    "ParamTotal": len(enabled),
                    "ParameterName": row.param,
                    "Algorithm": row.algorithm,
                })
            except Exception:
                pass

        if result.cancelled:
            break

    result.total_time_seconds = time.perf_counter() - t_start
    return result