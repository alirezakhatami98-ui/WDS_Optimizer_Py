"""Scenario — a saved, replayable configuration for one experiment.

A ``Scenario`` captures every input needed to reproduce one run:
network files, constraints, algorithm selection, and algorithm
parameters.  After the run completes, the scenario also carries the
``ExperimentResult``.

Design principles
-----------------
1. **Separate plan from execution.** A scenario starts as ``Draft``
   (no result).  After ``run_optimization`` it becomes ``Complete``
   (with result) or ``Failed``.  This lets users define scenarios
   *before* running them — useful for batch planning.

2. **Carry enough to be shared.** ``save_scenario`` embeds input files
   by default so a scenario can be shared with a colleague who has
   the same WDS Optimizer version.

3. **Compatible with existing snapshots.** A completed scenario, when
   serialized, uses the *same* JSON schema as an experiment snapshot
   (v0.4.0 of the project), so tools that read snapshots also read
   completed scenarios.

4. **No coupling to GUI/Streamlit.** ``scenario.py`` has no Streamlit
   imports; it can be used from CLI, API, tests, and a future SaaS.

Public surface
--------------
- ``Scenario`` dataclass
- ``new_scenario(...)`` helper (UUID + timestamp)
- ``save_scenario(sc, path, *, embed_inputs=True)``
- ``load_scenario(path)``
- ``Workspace`` (dict of scenarios + active id)
- ``save_workspace(ws, path)``
- ``load_workspace(path)``
"""
from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .experiments.runner import ExperimentResult
from .persistence.snapshot import (
    _run_to_dict, _dict_to_run, restore_experiment,
)


# =====================================================================
#  Scenario
# =====================================================================

@dataclass
class Scenario:
    """A single, reproducible experiment definition."""

    scenario_id: str
    name: str
    created_at: str

    # Inputs
    inp_path: str
    d_path: str
    cost_path: str
    diam_unit: str

    # Configuration
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
    status: str = "Draft"           # Draft | Running | Complete | Failed
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None

    def is_complete(self) -> bool:
        return self.status == "Complete" and self.experiment is not None


def new_scenario(
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
) -> Scenario:
    """Create a fresh Draft scenario with a new UUID and timestamp."""
    return Scenario(
        scenario_id=uuid.uuid4().hex,
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
    )


def clone_scenario(sc: Scenario, new_name: str) -> Scenario:
    """Deep-copy a scenario with a new id, name, and timestamp.

    The experiment result is *not* copied: the new scenario starts as
    Draft.  This matches user expectation: duplicating a scenario is
    "give me another config to run", not "give me another copy of the
    result".
    """
    return Scenario(
        scenario_id=uuid.uuid4().hex,
        name=new_name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        inp_path=sc.inp_path, d_path=sc.d_path, cost_path=sc.cost_path,
        diam_unit=sc.diam_unit,
        algorithms=list(sc.algorithms),
        n_runs=sc.n_runs,
        seed=sc.seed,
        ns=sc.ns,
        max_gen=sc.max_gen,
        constraints=copy.deepcopy(sc.constraints),
        algorithm_params=copy.deepcopy(sc.algorithm_params),
        constraint_policy=sc.constraint_policy,
        fixed_pipes=sc.fixed_pipes,
    )


# =====================================================================
#  Serialization
# =====================================================================

_SCENARIO_VERSION = "0.4.0"


def _read_text_safe(path: str | Path) -> str:
    """Read a text file, returning "" on any error (missing file,
    permission, etc.).  Inputs are optional in the serialized form."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def to_dict(sc: Scenario, *, embed_inputs: bool = True) -> dict:
    """Serialize a Scenario to a JSON-friendly dict.

    If ``embed_inputs`` is True, the current contents of ``inp_path``,
    ``d_path``, ``cost_path`` are embedded under ``inputs`` so the
    scenario is portable.  The paths themselves are still recorded.
    """
    data = {
        "version": _SCENARIO_VERSION,
        "scenario": {
            "scenario_id": sc.scenario_id,
            "name": sc.name,
            "created_at": sc.created_at,
            "inp_path": sc.inp_path,
            "d_path": sc.d_path,
            "cost_path": sc.cost_path,
            "diam_unit": sc.diam_unit,
            "algorithms": list(sc.algorithms),
            "n_runs": sc.n_runs,
            "seed": sc.seed,
            "ns": sc.ns,
            "max_gen": sc.max_gen,
            "constraints": copy.deepcopy(sc.constraints),
            "algorithm_params": copy.deepcopy(sc.algorithm_params),
            "constraint_policy": sc.constraint_policy,
            "fixed_pipes": sc.fixed_pipes,
            "status": sc.status,
            "error": sc.error,
            "elapsed_seconds": sc.elapsed_seconds,
        },
    }
    if sc.experiment is not None:
        data["experiment"] = {
            "experiment_id": sc.experiment.experiment_id,
            "number_of_runs": sc.experiment.number_of_runs,
            "algorithms": list(sc.experiment.algorithms),
            "experiment_seed": sc.experiment.experiment_seed,
            "NS": sc.experiment.NS,
            "MaxGen": sc.experiment.MaxGen,
            "cancelled": bool(getattr(sc.experiment, "cancelled", False)),
            "restart": dict(getattr(sc.experiment, "restart", {}) or {}),
            "constraint_policy": getattr(
                sc.experiment, "constraint_policy", "deb_prefers",
            ),
            "runs": {
                algo: [_run_to_dict(r) for r in runs]
                for algo, runs in sc.experiment.runs.items()
            },
        }
    if embed_inputs:
        data["inputs"] = {
            "inp":      _read_text_safe(sc.inp_path),
            "D_txt":    _read_text_safe(sc.d_path),
            "cost_txt": _read_text_safe(sc.cost_path),
        }
    return data


def from_dict(data: dict) -> Scenario:
    """Deserialize a scenario from a dict (see ``to_dict``)."""
    s = data["scenario"]
    experiment = None
    if "experiment" in data:
        # Reuse snapshot.restore_experiment for compatibility: it needs
        # a few extra keys (problem, best, hydraulics) that a scenario
        # does not have.  Build a minimal stub for those, then replace
        # the runs with what the scenario actually recorded.
        experiment = _experiment_from_scenario_dict(data["experiment"])

    return Scenario(
        scenario_id=s["scenario_id"],
        name=s["name"],
        created_at=s["created_at"],
        inp_path=s["inp_path"],
        d_path=s["d_path"],
        cost_path=s["cost_path"],
        diam_unit=s["diam_unit"],
        algorithms=list(s["algorithms"]),
        n_runs=int(s["n_runs"]),
        seed=int(s["seed"]),
        ns=int(s["ns"]),
        max_gen=int(s["max_gen"]),
        constraints=dict(s["constraints"]),
        algorithm_params={k: dict(v) for k, v in
                          s.get("algorithm_params", {}).items()},
        constraint_policy=s.get("constraint_policy", "deb_prefers"),
        fixed_pipes=s.get("fixed_pipes", ""),
        experiment=experiment,
        status=s.get("status", "Draft"),
        error=s.get("error"),
        elapsed_seconds=s.get("elapsed_seconds"),
    )


def _experiment_from_scenario_dict(exp_d: dict) -> ExperimentResult:
    """Rebuild an ExperimentResult from a scenario's experiment dict.

    Scenario serialization stores only the fields an ExperimentResult
    needs (id, runs, algorithms, seed, NS, MaxGen, cancelled, restart,
    constraint_policy).  The other snapshot fields (problem, best,
    hydraulics) are not applicable; we fill them with placeholders
    and substitute the runs from the scenario.
    """
    exp = ExperimentResult(
        experiment_id=exp_d["experiment_id"],
        timestamp=datetime.now(),
        number_of_runs=exp_d["number_of_runs"],
        algorithms=list(exp_d["algorithms"]),
        experiment_seed=exp_d["experiment_seed"],
        NS=exp_d["NS"],
        MaxGen=exp_d["MaxGen"],
        Pmin=0.0, Vmax=0.0,             # placeholders
        NP=0,                            # placeholder
        variable_pipes=[],
        fixed_pipes=[],
        cancelled=bool(exp_d.get("cancelled", False)),
        restart=dict(exp_d.get("restart", {})),
        constraint_policy=exp_d.get("constraint_policy", "deb_prefers"),
    )
    for algo, runs in exp_d.get("runs", {}).items():
        exp.runs[algo] = [_dict_to_run(r) for r in runs]
    return exp


# =====================================================================
#  Save / load
# =====================================================================

def save_scenario(
    sc: Scenario, path: str | Path, *,
    embed_inputs: bool = True,
) -> None:
    """Save a scenario to a JSON file.

    If ``embed_inputs=True``, the current contents of the input files
    are embedded so the scenario is portable.
    """
    data = to_dict(sc, embed_inputs=embed_inputs)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_scenario(path_or_file) -> Scenario:
    """Load a scenario from a JSON file or file-like object."""
    if hasattr(path_or_file, "read"):
        data = json.load(path_or_file)
    else:
        data = json.loads(
            Path(path_or_file).read_text(encoding="utf-8"),
        )
    return from_dict(data)


# =====================================================================
#  Workspace — a collection of scenarios + active id
# =====================================================================

@dataclass
class Workspace:
    """A collection of scenarios with one active at a time.

    Not persisted as part of this commit; v0.4.1 will add save/load
    of the whole workspace.  This dataclass exists so that v0.4.1 can
    use ``st.session_state.workspace`` immediately, without changing
    the data layout.
    """
    scenarios: dict[str, Scenario] = field(default_factory=dict)
    active_scenario_id: Optional[str] = None

    def add(self, sc: Scenario, *, activate: bool = True) -> None:
        if sc.scenario_id in self.scenarios:
            raise ValueError(
                f"Scenario id {sc.scenario_id!r} already in workspace"
            )
        self.scenarios[sc.scenario_id] = sc
        if activate or self.active_scenario_id is None:
            self.active_scenario_id = sc.scenario_id

    def remove(self, scenario_id: str) -> None:
        self.scenarios.pop(scenario_id, None)
        if self.active_scenario_id == scenario_id:
            self.active_scenario_id = next(
                iter(self.scenarios), None,
            )

    def active(self) -> Optional[Scenario]:
        if self.active_scenario_id is None:
            return None
        return self.scenarios.get(self.active_scenario_id)

    def list(self) -> list[Scenario]:
        """Scenarios in insertion order."""
        return list(self.scenarios.values())


def save_workspace(ws: Workspace, path: str | Path) -> None:
    """Save an entire workspace (all scenarios) to one JSON file."""
    data = {
        "version": _SCENARIO_VERSION,
        "active_scenario_id": ws.active_scenario_id,
        "scenarios": [to_dict(sc) for sc in ws.list()],
    }
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_workspace(path_or_file) -> Workspace:
    """Load a workspace from a JSON file or file-like object."""
    if hasattr(path_or_file, "read"):
        data = json.load(path_or_file)
    else:
        data = json.loads(
            Path(path_or_file).read_text(encoding="utf-8"),
        )
    ws = Workspace(active_scenario_id=data.get("active_scenario_id"))
    for entry in data.get("scenarios", []):
        sc = from_dict(entry)
        ws.scenarios[sc.scenario_id] = sc
    return ws