"""Save / load experiment snapshots as JSON."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ..experiments.runner import ExperimentResult, RunRecord


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _run_to_dict(r: RunRecord) -> dict:
    return {
        "algorithm": r.algorithm,
        "run_index": r.run_index,
        "seed": r.seed,
        "score": None if not np.isfinite(r.score) else float(r.score),
        "position": None if r.position is None else _to_list(r.position),
        "feasible": bool(r.feasible),
        "execution_time": (
            None if not np.isfinite(r.execution_time) else float(r.execution_time)
        ),
        "convergence": (
            None if r.convergence is None else _to_list(r.convergence)
        ),
        "status": r.status,
        "error_message": r.error_message,
    }


def _dict_to_run(d: dict) -> RunRecord:
    return RunRecord(
        algorithm=d["algorithm"],
        run_index=d["run_index"],
        seed=d["seed"],
        score=d["score"] if d["score"] is not None else float("nan"),
        position=np.array(d["position"]) if d["position"] is not None else None,
        feasible=d["feasible"],
        execution_time=(
            d["execution_time"] if d["execution_time"] is not None
            else float("nan")
        ),
        convergence=(
            np.array(d["convergence"]) if d["convergence"] is not None else None
        ),
        status=d["status"],
        error_message=d["error_message"],
    )


def build_snapshot(
    *,
    problem_snapshot: dict,
    experiment: ExperimentResult,
    best_algo: str,
    best_run_index: int,
    best_position_m: np.ndarray,
    node_pressures: np.ndarray,
    pipe_velocities: np.ndarray,
    hydraulics_converged: bool,
    config: dict,
    inp_content: str | None = None,       # NEW
    d_content: str | None = None,         # NEW
    cost_content: str | None = None,      # NEW
) -> dict:
    snapshot = {
        "version": "0.4.0",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "diam_unit": problem_snapshot.get("Units", {}).get(
                "DiameterUnit", "mm"
            ),
            "pmin": problem_snapshot["Pmin"],
            "vmax": problem_snapshot["Vmax"],
        },
        "config": {k: config.get(k) for k in (
            "ns", "max_gen", "ga_pc", "ga_pm", "ga_sel",
            "pso_w", "pso_c1", "pso_c2", "n_runs", "exp_seed",
        )},
        "problem": {
            "pipe_names":     list(problem_snapshot["pipe_names"]),
            "junction_names": list(problem_snapshot["junction_names"]),
            "L":              _to_list(problem_snapshot["L"]),
            "initial_D":      _to_list(problem_snapshot["initial_D"]),
            "D_m":            _to_list(problem_snapshot["D_m"]),
            "Din":            _to_list(problem_snapshot["Din"]),
            "Cost":           _to_list(problem_snapshot["Cost"]),
            "VariablePipes":  _to_list(problem_snapshot["VariablePipes"]),
            "FixedPipes":     _to_list(problem_snapshot["FixedPipes"]),
            "Pmin":           problem_snapshot["Pmin"],
            "Vmax":           problem_snapshot["Vmax"],
            "Units":          dict(problem_snapshot["Units"]),
        },
        "experiment": {
            "experiment_id":   experiment.experiment_id,
            "number_of_runs":  experiment.number_of_runs,
            "algorithms":      list(experiment.algorithms),
            "experiment_seed": experiment.experiment_seed,
            "NS":              experiment.NS,
            "MaxGen":          experiment.MaxGen,
            "runs": {
                algo: [_run_to_dict(r) for r in runs]
                for algo, runs in experiment.runs.items()
            },
        },
        "best": {
            "algorithm": best_algo,
            "run_index": best_run_index,
            "position":  _to_list(best_position_m),
        },
        "hydraulics": {
            "node_pressures":  _to_list(node_pressures),
            "pipe_velocities": _to_list(pipe_velocities),
            "converged":       bool(hydraulics_converged),
        },
    }

    # Embed input files so the experiment is fully self-contained.
    if inp_content is not None:
        snapshot["inputs"] = {
            "inp":      inp_content,
            "D_txt":    d_content,
            "cost_txt": cost_content,
        }

    return snapshot


def save_snapshot(snapshot: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)


def load_snapshot(path_or_file) -> dict:
    if hasattr(path_or_file, "read"):
        return json.load(path_or_file)
    return json.loads(Path(path_or_file).read_text(encoding="utf-8"))


def restore_experiment(data: dict) -> ExperimentResult:
    """Rebuild an ExperimentResult from a loaded snapshot dict."""
    exp = data["experiment"]
    result = ExperimentResult(
        experiment_id=exp["experiment_id"],
        timestamp=datetime.now(),
        number_of_runs=exp["number_of_runs"],
        algorithms=exp["algorithms"],
        experiment_seed=exp["experiment_seed"],
        NS=exp["NS"],
        MaxGen=exp["MaxGen"],
        Pmin=data["problem"]["Pmin"],
        Vmax=data["problem"]["Vmax"],
        NP=len(data["problem"]["pipe_names"]),
        variable_pipes=np.array(data["problem"]["VariablePipes"]),
        fixed_pipes=np.array(data["problem"]["FixedPipes"]),
    )
    for algo, runs in exp["runs"].items():
        result.runs[algo] = [_dict_to_run(r) for r in runs]
    return result