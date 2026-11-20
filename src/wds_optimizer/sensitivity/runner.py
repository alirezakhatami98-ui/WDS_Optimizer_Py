"""Sensitivity single-sweep runner.

Runs one experiment per value of a single parameter, keeping
everything else the same.  Each iteration gets a fresh copy of the
base AlgorithmConfig and a fresh copy of problem["Constraints"] so
constraint-value sweeps do not leak across iterations.

The multi-parameter ("batch") API lives in
wds_optimizer/sensitivity/batch.py (added in v0.8.1).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..algorithms.config import AlgorithmConfig, build_algorithm_config
from ..experiments.runner import ExperimentResult, run_experiment
from ..hydraulics.units import pressure_to_si, velocity_to_si
from .config import build_sensitivity_parameters, normalize_param_name


@dataclass
class SingleSweepPoint:
    """Result of one value in a single-parameter sweep."""
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
class SingleSweepResult:
    """Result of one single-parameter sensitivity sweep."""
    parameter: str
    values: list
    algorithms: list[str]
    n_runs: int
    experiment_seed: int
    points: list[SingleSweepPoint] = field(default_factory=list)
    cancelled: bool = False


# ---------------------------------------------------------------------
#  Applying one parameter value
# ---------------------------------------------------------------------

def _apply_sensitivity_param(
    cfg: AlgorithmConfig,
    problem: dict,
    param_name: str,
    value: Any,
) -> None:
    """Apply a sensitivity value to a config and/or problem.

    Mutates ``cfg`` in place.  Mutates ``problem["Constraints"]`` in
    place (caller must pass a fresh copy of Constraints).

    Raises ValueError if the parameter is unknown.
    """
    key = normalize_param_name(param_name)
    catalog = build_sensitivity_parameters()

    if key not in catalog:
        raise ValueError(
            f"unknown sensitivity parameter {param_name!r} "
            f"(normalized: {key!r}); available: "
            f"{sorted(catalog.keys())[:10]}..."
        )

    info = catalog[key]

    # ---- Global scalar config fields ----
    if key == "NS":
        cfg.NS = int(value); return
    if key == "MaxGen":
        cfg.MaxGen = int(value); return
    if key == "n_runs":
        # handled by run_single_sweep
        return
    if key == "seed":
        # handled by run_single_sweep
        return
    if key == "constraint_policy":
        cfg.constraint_policy = str(value); return

    # ---- Global: constraint values (display units → SI) ----
    if key in ("Pmin", "Pmax", "Vmin", "Vmax"):
        units = problem.get("Units") or {
            "System": "SI", "PressureUnit": "m",
            "VelocityUnit": "m/s", "LengthUnit": "m",
            "DiameterUnit": "mm",
        }
        if key in ("Pmin", "Pmax"):
            si = pressure_to_si(float(value), units)
        else:
            si = velocity_to_si(float(value), units)
        problem["Constraints"][key]["value"] = float(si)
        problem["Constraints"][key]["enabled"] = True
        return

    # ---- Algorithm-specific: namespaced `ALGO.param` ----
    if "." in key:
        algo_id, param_key = key.split(".", 1)
        entry = catalog[key]
        family = entry.get("algorithm")
        if family is None:
            raise ValueError(
                f"cannot apply sensitivity value for {key!r}: "
                f"no algorithm associated with this parameter"
            )
        # Plugin algorithms are not fields on AlgorithmConfig; their
        # params live in cfg.extra[family].
        if not hasattr(cfg, family):
            cfg.extra.setdefault(family, {})[param_key] = value
            return
        from ..algorithms.registry import ALGORITHM_REGISTRY
        meta = ALGORITHM_REGISTRY[family]["meta"]
        maps_to = None
        for p in meta["params"]:
            if p["key"] == param_key:
                maps_to = p.get("maps_to")
                break
        if maps_to is not None:
            fam_field, field_name = maps_to
            fam_cfg = getattr(cfg, fam_field, None)
            if fam_cfg is None or not hasattr(fam_cfg, field_name):
                raise ValueError(
                    f"cannot set {fam_field}.{field_name} for "
                    f"sensitivity {key}: config field missing"
                )
            setattr(fam_cfg, field_name, value)
        else:
            cfg.extra.setdefault(family, {})[param_key] = value
        return

    raise ValueError(f"cannot apply sensitivity value for {key!r}")


# ---------------------------------------------------------------------
#  Single-sweep runner
# ---------------------------------------------------------------------

def run_single_sweep(
    problem: dict,
    base_config: AlgorithmConfig,
    *,
    parameter_name: str,
    values: list,
    algorithms: list[str],
    n_runs: int,
    experiment_seed: int,
    progress_callback: Optional[Callable] = None,
    cancel_callback: Optional[Callable] = None,
) -> SingleSweepResult:
    """Execute one experiment per value of one parameter."""
    result = SingleSweepResult(
        parameter=parameter_name,
        values=list(values),
        algorithms=list(algorithms),
        n_runs=int(n_runs),
        experiment_seed=int(experiment_seed),
    )

    for idx, value in enumerate(values):
        if cancel_callback is not None and cancel_callback():
            result.cancelled = True
            break

        point = SingleSweepPoint(value=value)
        try:
            cfg = _clone_config(base_config)
            problem_iter = dict(problem)
            problem_iter["Constraints"] = copy.deepcopy(
                problem.get("Constraints", {})
            )
            _apply_sensitivity_param(
                cfg, problem_iter, parameter_name, value,
            )

            eff_n_runs = n_runs
            eff_seed = experiment_seed
            nk = normalize_param_name(parameter_name)
            if nk == "n_runs":
                eff_n_runs = int(value)
            elif nk == "seed":
                eff_seed = int(value)

            import time
            t0 = time.perf_counter()
            exp = run_experiment(
                problem_iter, cfg,
                number_of_runs=eff_n_runs,
                algorithms=algorithms,
                experiment_seed=eff_seed,
                progress_callback=_wrap_progress(
                    progress_callback, idx, len(values), value,
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

        result.points.append(point)

        if progress_callback is not None:
            try:
                progress_callback({
                    "Phase": "PointDone",
                    "PointIndex": idx + 1,
                    "TotalPoints": len(values),
                    "ParameterValue": value,
                })
            except Exception:
                pass

    return result


def _clone_config(base: AlgorithmConfig) -> AlgorithmConfig:
    from ..algorithms.config import (
        DEConfig, GAConfig, PSOConfig, RestartConfig, SADEConfig,
    )
    return AlgorithmConfig(
        NS=base.NS, MaxGen=base.MaxGen, Seed=base.Seed,
        Cache=base.Cache,
        GA=copy.deepcopy(base.GA),
        PSO=copy.deepcopy(base.PSO),
        DE=copy.deepcopy(base.DE),
        SADE=copy.deepcopy(base.SADE),
        Restart=copy.deepcopy(base.Restart),
        Parallel=copy.deepcopy(base.Parallel),
        extra=copy.deepcopy(base.extra),
        constraint_policy=base.constraint_policy,
    )


def _wrap_progress(cb, idx: int, total: int, value):
    if cb is None:
        return None
    def wrapper(info: dict) -> None:
        info = dict(info)
        info["PointIndex"] = idx + 1
        info["TotalPoints"] = total
        info["ParameterValue"] = value
        try:
            cb(info)
        except Exception:
            pass
    return wrapper


# Aliases for backward compatibility with the old name.
run_parameter_sweep = run_single_sweep
SweepResult = SingleSweepResult
SweepPointResult = SingleSweepPoint