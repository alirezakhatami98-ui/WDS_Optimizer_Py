"""History entries — records of past experiments.

An entry captures everything needed to reproduce one run: the config
(network paths, algorithm selection, algorithm params, constraints)
and the result (an ExperimentResult, once the run has completed).

Design principles
-----------------
1. **Auto-populated.**  A new entry is created automatically when a
   run completes; the user never has to manually "save" a result.

2. **User-named.**  An entry starts with an auto-generated name (e.g.
   ``"GA+SADE / hanoi / ns=100 / seed=42"``) and can be renamed at
   any time.  Users can also enter a name *before* running.

3. **Self-contained.**  Serialization embeds the input files by
   default, so an entry JSON can be shared with a colleague.

4. **Read-only after creation (except name).**  Config and result are
   captured at run time; to change a config, load it into the sidebar,
   edit, and re-run — that creates a *new* entry, keeping the old one
   intact for comparison.

5. **No coupling to GUI or Streamlit.**  history.py is pure data + I/O.
"""
from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .experiments.runner import ExperimentResult
from .persistence.snapshot import _run_to_dict, _dict_to_run


_HISTORY_VERSION = "0.4.1a"


@dataclass
class HistoryEntry:
    """One record of a completed (or cancelled) experiment."""
    entry_id: str
    name: str
    created_at: str

    # Config
    inp_path: str
    d_path: str
    cost_path: str
    diam_unit: str
    algorithms: list[str]
    n_runs: int
    seed: int
    ns: int
    max_gen: int
    constraints: dict
    algorithm_params: dict
    constraint_policy: str = "deb_prefers"
    fixed_pipes: str = ""

    # Result
    experiment: Optional[ExperimentResult] = None
    status: str = "Complete"       # Complete | Failed | Cancelled
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None

    # Sweep tagging (v0.6.1).  Entries created as part of a sweep
    # share the same sweep_group_id; the History panel groups them.
    # sweep_meta records which parameter and value this point used.
    sweep_group_id: Optional[str] = None
    sweep_meta: Optional[dict] = None


def new_entry(
    name: str,
    *,
    inp_path: str,
    d_path: str,
    cost_path: str,
    diam_unit: str,
    algorithms: list[str],
    n_runs: int,
    seed: int,
    ns: int,
    max_gen: int,
    constraints: dict,
    algorithm_params: dict,
    constraint_policy: str = "deb_prefers",
    fixed_pipes: str = "",
    sweep_group_id: Optional[str] = None,
    sweep_meta: Optional[dict] = None,
) -> HistoryEntry:
    """Create a fresh HistoryEntry."""
    return HistoryEntry(
        entry_id=uuid.uuid4().hex,
        name=name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        inp_path=inp_path, d_path=d_path, cost_path=cost_path,
        diam_unit=diam_unit,
        algorithms=list(algorithms),
        n_runs=int(n_runs),
        seed=int(seed),
        ns=int(ns),
        max_gen=int(max_gen),
        constraints=copy.deepcopy(constraints),
        algorithm_params=copy.deepcopy(algorithm_params),
        constraint_policy=constraint_policy,
        fixed_pipes=fixed_pipes,
        sweep_group_id=sweep_group_id,
        sweep_meta=copy.deepcopy(sweep_meta) if sweep_meta else None,
    )


def auto_name(
    *, algorithms: list[str], inp_path: str, ns: int, seed: int,
) -> str:
    """Auto-generated label for a config."""
    net = Path(inp_path).stem if inp_path else "network"
    algos = "+".join(algorithms) if algorithms else "?"
    return f"{algos} / {net} / ns={ns} / seed={seed}"


def _read_text_safe(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def to_dict(entry: HistoryEntry, *, embed_inputs: bool = True) -> dict:
    """Serialize one entry to a JSON-friendly dict."""
    data = {
        "version": _HISTORY_VERSION,
        "entry": {
            "entry_id": entry.entry_id,
            "name": entry.name,
            "created_at": entry.created_at,
            "inp_path": entry.inp_path,
            "d_path": entry.d_path,
            "cost_path": entry.cost_path,
            "diam_unit": entry.diam_unit,
            "algorithms": list(entry.algorithms),
            "n_runs": entry.n_runs,
            "seed": entry.seed,
            "ns": entry.ns,
            "max_gen": entry.max_gen,
            "constraints": copy.deepcopy(entry.constraints),
            "algorithm_params": copy.deepcopy(entry.algorithm_params),
            "constraint_policy": entry.constraint_policy,
            "fixed_pipes": entry.fixed_pipes,
            "status": entry.status,
            "error": entry.error,
            "elapsed_seconds": entry.elapsed_seconds,
            "sweep_group_id": entry.sweep_group_id,
            "sweep_meta": copy.deepcopy(entry.sweep_meta),
        },
    }
    if entry.experiment is not None:
        data["experiment"] = {
            "experiment_id": entry.experiment.experiment_id,
            "number_of_runs": entry.experiment.number_of_runs,
            "algorithms": list(entry.experiment.algorithms),
            "experiment_seed": entry.experiment.experiment_seed,
            "NS": entry.experiment.NS,
            "MaxGen": entry.experiment.MaxGen,
            "cancelled": bool(getattr(entry.experiment, "cancelled", False)),
            "restart": dict(getattr(entry.experiment, "restart", {}) or {}),
            "constraint_policy": getattr(
                entry.experiment, "constraint_policy", "deb_prefers"),
            "runs": {
                algo: [_run_to_dict(r) for r in runs]
                for algo, runs in entry.experiment.runs.items()
            },
        }
    if embed_inputs:
        data["inputs"] = {
            "inp": _read_text_safe(entry.inp_path),
            "D_txt": _read_text_safe(entry.d_path),
            "cost_txt": _read_text_safe(entry.cost_path),
        }
    return data


def _experiment_from_entry_dict(exp_d: dict) -> ExperimentResult:
    exp = ExperimentResult(
        experiment_id=exp_d["experiment_id"],
        timestamp=datetime.now(),
        number_of_runs=exp_d["number_of_runs"],
        algorithms=list(exp_d["algorithms"]),
        experiment_seed=exp_d["experiment_seed"],
        NS=exp_d["NS"],
        MaxGen=exp_d["MaxGen"],
        Pmin=0.0, Vmax=0.0, NP=0,
        variable_pipes=[], fixed_pipes=[],
        cancelled=bool(exp_d.get("cancelled", False)),
        restart=dict(exp_d.get("restart", {})),
        constraint_policy=exp_d.get("constraint_policy", "deb_prefers"),
    )
    for algo, runs in exp_d.get("runs", {}).items():
        exp.runs[algo] = [_dict_to_run(r) for r in runs]
    return exp


def from_dict(data: dict) -> HistoryEntry:
    """Deserialize a HistoryEntry from a dict."""
    e = data["entry"]
    experiment = None
    if "experiment" in data:
        experiment = _experiment_from_entry_dict(data["experiment"])
    return HistoryEntry(
        entry_id=e["entry_id"],
        name=e["name"],
        created_at=e["created_at"],
        inp_path=e["inp_path"],
        d_path=e["d_path"],
        cost_path=e["cost_path"],
        diam_unit=e["diam_unit"],
        algorithms=list(e["algorithms"]),
        n_runs=int(e["n_runs"]),
        seed=int(e["seed"]),
        ns=int(e["ns"]),
        max_gen=int(e["max_gen"]),
        constraints=copy.deepcopy(e["constraints"]),
        algorithm_params=copy.deepcopy(e.get("algorithm_params", {})),
        constraint_policy=e.get("constraint_policy", "deb_prefers"),
        fixed_pipes=e.get("fixed_pipes", ""),
        experiment=experiment,
        status=e.get("status", "Complete"),
        error=e.get("error"),
        elapsed_seconds=e.get("elapsed_seconds"),
        sweep_group_id=e.get("sweep_group_id"),
        sweep_meta=copy.deepcopy(e.get("sweep_meta")),
    )


def save_entry(entry: HistoryEntry, path: str | Path, *,
               embed_inputs: bool = True) -> None:
    """Save one entry to a JSON file."""
    Path(path).write_text(
        json.dumps(to_dict(entry, embed_inputs=embed_inputs),
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_entry(path_or_file) -> HistoryEntry:
    """Load one entry from a JSON file or file-like object."""
    if hasattr(path_or_file, "read"):
        data = json.load(path_or_file)
    else:
        data = json.loads(Path(path_or_file).read_text(encoding="utf-8"))
    return from_dict(data)


def history_to_dict(entries: list[HistoryEntry]) -> dict:
    """Serialize a whole history to a JSON-friendly dict."""
    return {
        "version": _HISTORY_VERSION,
        "entries": [to_dict(e) for e in entries],
    }


def save_history(entries: list[HistoryEntry], path: str | Path) -> None:
    """Save the whole history to one JSON file."""
    Path(path).write_text(
        json.dumps(history_to_dict(entries), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_history(path_or_file) -> list[HistoryEntry]:
    """Load a history from a JSON file or file-like object.

    Accepts both:
    - a full history (``{"entries": [...]}``) — as written by
      ``save_history`` / the "💾 Save all" button;
    - a single entry (``{"entry": {...}, "experiment": {...}}``) — as
      written by ``save_entry`` / the "💾" button on one row.

    Loading a single entry returns a one-element list.  This lets the
    user upload either kind of file from the same "Load" widget
    without first needing to distinguish them.
    """
    if hasattr(path_or_file, "read"):
        data = json.load(path_or_file)
    else:
        data = json.loads(Path(path_or_file).read_text(encoding="utf-8"))
    if "entries" in data:
        return [from_dict(e) for e in data["entries"]]
    if "entry" in data:
        return [from_dict(data)]
    return []