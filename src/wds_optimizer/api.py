"""Programmatic API for WDS Optimizer — the non-GUI entry point.

This module is the single source of truth for "given input files, build
the optimization problem and run it".  It is imported by:

- ``app/streamlit_app.py`` (GUI)
- ``cli.py`` (command-line interface, added in a later commit)
- any user code that wants to embed WDS Optimizer in its own pipeline

No Streamlit, Plotly, or GUI-related imports appear anywhere here.
This is deliberate: the module must be usable on a headless server
(SaaS, CI, embedded).

v0.1.0 public surface
---------------------
- ``load_network_isolated(inp_path)``
- ``build_problem(inp_path=..., d_path=..., cost_path=..., ...)``
- ``OptimizationRequest`` — dataclass describing one run
- ``OptimizationResult`` — dataclass holding results + export methods
- ``run_optimization(request, *, progress_callback, cancel_callback)``

Everything else in this module is private (leading underscore).
"""

from __future__ import annotations

import copy
import io
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import wntr

from wds_optimizer.algorithms.config import (
    AlgorithmConfig,
    DEConfig,
    GAConfig,
    PSOConfig,
    RestartConfig,
    SADEConfig,
    build_algorithm_config,
)
from wds_optimizer.data.loaders import load_optimization_data
from wds_optimizer.experiments.runner import (
    ExperimentResult, run_experiment, summarize_experiment,
)
from wds_optimizer.hydraulics.simulation import run_hydraulic_simulation
from wds_optimizer.hydraulics.units import detect_network_units
from wds_optimizer.optimization.problem import build_optimization_problem
from wds_optimizer.persistence.snapshot import build_snapshot
from wds_optimizer.reporting.report import build_report_zip


# =====================================================================
#  Network loading (from commit 1)
# =====================================================================

def _parse_network(inp_path: str | Path) -> wntr.network.WaterNetworkModel:
    """Raw parse — no copy, no cache.  Private (see commit 1 docstring)."""
    return wntr.network.WaterNetworkModel(str(inp_path))


def load_network_isolated(
    inp_path: str | Path,
) -> wntr.network.WaterNetworkModel:
    """Parse a network and return a *fresh, mutable* copy (see commit 1)."""
    return copy.deepcopy(_parse_network(inp_path))


# =====================================================================
#  Problem construction (from commit 1)
# =====================================================================

def _parse_fixed_pipes(
    fixed_pipes: Sequence[int] | str, n_pipes: int,
) -> list[int]:
    """Normalise ``fixed_pipes`` to a sorted list of 0-based ints.

    Accepts either 0-based indices (API form, ``[0, 2]``) or a
    comma-separated 1-based string (GUI form, ``"1, 3"``).
    """
    if isinstance(fixed_pipes, str):
        s = fixed_pipes.strip()
        if not s:
            return []
        try:
            one_based = [
                int(x.strip()) for x in s.split(",") if x.strip()
            ]
        except ValueError:
            raise ValueError(
                "Fixed pipe IDs must be integers separated by commas."
            )
        for f in one_based:
            if f < 1 or f > n_pipes:
                raise ValueError(
                    f"Fixed pipe IDs must be between 1 and {n_pipes}."
                )
        if len(set(one_based)) != len(one_based):
            raise ValueError("Fixed pipe IDs must be unique.")
        return sorted(f - 1 for f in one_based)

    indices = list(fixed_pipes or [])
    for f in indices:
        if not isinstance(f, int) or f < 0 or f >= n_pipes:
            raise ValueError(
                f"Fixed pipe index {f} out of range 0..{n_pipes - 1}."
            )
    if len(set(indices)) != len(indices):
        raise ValueError("Fixed pipe indices must be unique.")
    return sorted(indices)


def _validate_constraints(constraints: dict) -> None:
    """Check the four-constraint dict (structural + logical)."""
    if not isinstance(constraints, dict):
        raise ValueError(
            "constraints must be a dict: "
            "{'Pmin'|'Pmax'|'Vmin'|'Vmax': {'enabled': bool, 'value': float}}"
        )

    required = ("Pmin", "Pmax", "Vmin", "Vmax")
    for key in required:
        if key not in constraints:
            raise ValueError(f"constraints missing key: {key!r}")
        entry = constraints[key]
        if (not isinstance(entry, dict)
                or "enabled" not in entry or "value" not in entry):
            raise ValueError(
                f"constraints[{key!r}] must be "
                f"{{'enabled': bool, 'value': float}}"
            )

    if not any(constraints[k]["enabled"] for k in required):
        raise ValueError("Enable at least one constraint before running.")

    if (constraints["Pmin"]["enabled"]
            and constraints["Pmax"]["enabled"]
            and float(constraints["Pmin"]["value"])
                >= float(constraints["Pmax"]["value"])):
        raise ValueError(
            f"Pmin ({constraints['Pmin']['value']}) must be smaller than "
            f"Pmax ({constraints['Pmax']['value']})."
        )
    if (constraints["Vmin"]["enabled"]
            and constraints["Vmax"]["enabled"]
            and float(constraints["Vmin"]["value"])
                >= float(constraints["Vmax"]["value"])):
        raise ValueError(
            f"Vmin ({constraints['Vmin']['value']}) must be smaller than "
            f"Vmax ({constraints['Vmax']['value']})."
        )


def build_problem(
    *,
    inp_path: str | Path,
    d_path: str | Path,
    cost_path: str | Path,
    diam_unit: str,
    constraints: dict,
    fixed_pipes: Sequence[int] | str = (),
) -> tuple[dict, dict, wntr.network.WaterNetworkModel]:
    """Build the optimisation problem dict from raw input files.

    See commit 1 docstring for full details.  Constraints are in SI
    (pressure in m, velocity in m/s).
    """
    _validate_constraints(constraints)

    wn = load_network_isolated(inp_path)
    units = detect_network_units(wn)

    Din, Cost, D_m = load_optimization_data(
        str(d_path), str(cost_path),
        diam_file_unit=diam_unit,
    )

    pipe_names = list(wn.pipe_name_list)
    junction_names = list(wn.junction_name_list)
    lengths = np.array([wn.get_link(n).length for n in pipe_names])
    initial_D = np.array([wn.get_link(n).diameter for n in pipe_names])

    fixed_0based = _parse_fixed_pipes(fixed_pipes, len(pipe_names))
    variable_0based = [
        i for i in range(len(pipe_names)) if i not in fixed_0based
    ]

    problem = build_optimization_problem(
        wn=wn, Din=Din, Cost=Cost, D_m=D_m,
        full_lengths=lengths, initial_D=initial_D,
        fixed_pipes=fixed_0based,
        variable_pipes=variable_0based,
        constraints=constraints,
        pipe_names=pipe_names,
        junction_names=junction_names,
        units=units,
    )
    return problem, units, wn


# =====================================================================
#  OptimizationRequest
# =====================================================================

def _default_constraints() -> dict:
    """Defaults matching the GUI (Pmin on, Vmax on, others off)."""
    return {
        "Pmin": {"enabled": True,  "value": 30.0},
        "Pmax": {"enabled": False, "value": 100.0},
        "Vmin": {"enabled": False, "value": 0.0},
        "Vmax": {"enabled": True,  "value": 2.5},
    }


@dataclass
class OptimizationRequest:
    """Complete specification of one optimisation run.

    All constraint values are in **SI units** (pressure in metres,
    velocity in m/s).  The GUI converts user-entered display units to
    SI before constructing this object; the CLI parses --pmin etc. and
    converts to SI.  This keeps the API single-unit and unambiguous.

    Example
    -------
    >>> req = OptimizationRequest(
    ...     inp_path="hanoi.inp", d_path="D.txt", cost_path="Cost.txt",
    ...     diam_unit="mm",
    ...     algorithms=["SADE"], n_runs=10, seed=42,
    ...     ns=200, max_gen=500,
    ...     constraints={
    ...         "Pmin": {"enabled": True,  "value": 30.0},
    ...         "Pmax": {"enabled": False, "value": 100.0},
    ...         "Vmin": {"enabled": False, "value": 0.0},
    ...         "Vmax": {"enabled": True,  "value": 2.5},
    ...     },
    ... )
    >>> req.validate()                     # raises if anything is wrong
    >>> result = run_optimization(req)
    >>> result.to_json()                   # bytes, JSON snapshot
    """

    # ---- Required inputs ----
    inp_path:  str | Path
    d_path:    str | Path
    cost_path: str | Path

    # ---- Optional inputs ----
    diam_unit: str = "mm"

    # ---- Algorithms ----
    algorithms: Sequence[str] = ("GA",)
    n_runs:     int = 1
    seed:       int = 1

    # ---- Population / generations ----
    ns:      int = 100
    max_gen: int = 500

    # ---- Constraints (SI) ----
    constraints: dict | None = None

    # ---- Per-algorithm parameters (None = library default) ----
    ga:      GAConfig      | None = None
    pso:     PSOConfig     | None = None
    de:      DEConfig      | None = None
    sade:    SADEConfig    | None = None
    restart: RestartConfig | None = None

    # ---- Fixed pipes ----
    # Sequence[int] = 0-based indices (API form)
    # str          = comma-separated 1-based IDs (GUI form)
    fixed_pipes: Sequence[int] | str = ()

    # ---- Advanced ----
    verbose: bool = False

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = _default_constraints()
        # Coerce paths to str for consistent hashing / caching
        self.inp_path  = str(self.inp_path)
        self.d_path    = str(self.d_path)
        self.cost_path = str(self.cost_path)
        self.algorithms = tuple(self.algorithms)

    def validate(self) -> None:
        """Raise ``ValueError`` with a clear message on any invalid field.

        Checks, in order:

        - input files exist
        - ``diam_unit`` is one of ``"mm"`` / ``"in"``
        - at least one algorithm; each is GA/PSO/DE/SADE
        - ``n_runs >= 1``, ``seed >= 0``
        - ``ns >= 4`` (DE/SADE requirement; enforced always so a
          switch between algorithms cannot break at runtime)
        - ``max_gen >= 5``
        - ``constraints`` (all four families, SI values)
        """
        for name, p in (("inp_path", self.inp_path),
                        ("d_path", self.d_path),
                        ("cost_path", self.cost_path)):
            if not Path(p).exists():
                raise ValueError(f"input file not found: {name}={p!r}")

        if self.diam_unit not in ("mm", "in"):
            raise ValueError(
                f"diam_unit must be 'mm' or 'in', got {self.diam_unit!r}"
            )

        if not self.algorithms:
            raise ValueError("specify at least one algorithm")

        valid_algos = {"GA", "PSO", "DE", "SADE"}
        for a in self.algorithms:
            if a not in valid_algos:
                raise ValueError(
                    f"unknown algorithm {a!r}; expected one of "
                    f"{sorted(valid_algos)}"
                )

        if self.n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {self.n_runs}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.ns < 4:
            raise ValueError(
                f"population (ns) must be >= 4 (DE/SADE requirement), "
                f"got {self.ns}"
            )
        if self.max_gen < 5:
            raise ValueError(
                f"max_gen must be >= 5, got {self.max_gen}"
            )

        _validate_constraints(self.constraints)

        # SADE-specific consistency
        if self.sade is not None:
            if self.sade.Fmin > self.sade.Fmax:
                raise ValueError(
                    f"SADE Fmin ({self.sade.Fmin}) must be <= Fmax "
                    f"({self.sade.Fmax})"
                )
            if self.sade.CRmin > self.sade.CRmax:
                raise ValueError(
                    f"SADE CRmin ({self.sade.CRmin}) must be <= CRmax "
                    f"({self.sade.CRmax})"
                )


# =====================================================================
#  OptimizationResult
# =====================================================================

@dataclass
class OptimizationResult:
    """Everything produced by one optimisation run.

    Export methods return ``bytes`` (or ``str`` for CSV) so the caller
    decides where to write.  The API never touches the filesystem
    beyond reading the input files.
    """

    # ---- Core ----
    experiment:      ExperimentResult
    problem_snapshot: dict              # dict schema used by snapshots
    units:           dict               # detected network unit system
    best_algo:       str
    best_run_index:  int
    best_position_m: np.ndarray         # full diameter vector, metres
    best_score:      float
    best_feasible:   bool

    # ---- Hydraulics of the best design ----
    node_pressures_m:    np.ndarray | None
    pipe_velocities_ms:  np.ndarray | None
    hydraulics_converged: bool

    # ---- Metadata ----
    request:         OptimizationRequest
    elapsed_seconds: float

    # ---- Export methods ----

    def to_json(self, *, include_inputs: bool = True) -> bytes:
        """Snapshot compatible with the existing GUI loader.

        If ``include_inputs``, embeds the three input files in the JSON
        so the experiment is fully self-contained (same behaviour as
        the GUI's Save button).
        """
        inp_content = d_content = cost_content = None
        if include_inputs:
            inp_content = Path(self.request.inp_path).read_text(
                encoding="utf-8", errors="replace")
            d_content = Path(self.request.d_path).read_text(
                encoding="utf-8", errors="replace")
            cost_content = Path(self.request.cost_path).read_text(
                encoding="utf-8", errors="replace")

        snap = build_snapshot(
            problem_snapshot=self.problem_snapshot,
            experiment=self.experiment,
            best_algo=self.best_algo,
            best_run_index=self.best_run_index,
            best_position_m=self.best_position_m,
            node_pressures=self.node_pressures_m,
            pipe_velocities=self.pipe_velocities_ms,
            hydraulics_converged=self.hydraulics_converged,
            config=_request_to_config_dict(self.request),
            inp_content=inp_content,
            d_content=d_content,
            cost_content=cost_content,
        )
        return json.dumps(snap, indent=2, ensure_ascii=False).encode("utf-8")

    def to_csv(self) -> bytes:
        """Per-run table (matches the GUI's All Runs export)."""
        import csv as _csv
        rows = []
        for algo, runs in self.experiment.runs.items():
            for r in runs:
                rows.append({
                    "Algorithm": algo,
                    "Run": r.run_index,
                    "Seed": r.seed,
                    "Status": r.status,
                    "Feasible": "Yes" if r.feasible else "No",
                    "Cost": ("" if not np.isfinite(r.score)
                             else f"{r.score:.2f}"),
                    "NFE": r.nfe,
                    "Simulations": r.n_sim,
                    "NFE_to_best": r.nfe_to_best,
                    "Time_s": (f"{r.execution_time:.3f}"
                               if np.isfinite(r.execution_time) else ""),
                    "Restarts": len(getattr(r, "restart_gens", [])),
                    "Error": r.error_message,
                })
        buf = io.StringIO()
        if rows:
            w = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        # utf-8-sig: BOM so Excel opens the file with correct encoding
        return buf.getvalue().encode("utf-8-sig")

    def to_excel(self) -> bytes:
        """Multi-sheet workbook (Summary + one sheet per algorithm)."""
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            summary = summarize_experiment(self.experiment)
            s_rows = [{"Algorithm": a, **s} for a, s in summary.items()]
            pd.DataFrame(s_rows).to_excel(
                writer, sheet_name="Summary", index=False
            )
            for algo, runs in self.experiment.runs.items():
                rows = [{
                    "Run": r.run_index, "Seed": r.seed,
                    "Status": r.status, "Cost": r.score,
                    "Feasible": r.feasible,
                    "Time (s)": r.execution_time,
                    "NFE": r.nfe,
                    "Simulations": r.n_sim,
                    "NFE to best": r.nfe_to_best,
                    "Error": r.error_message,
                } for r in runs]
                pd.DataFrame(rows).to_excel(
                    writer, sheet_name=f"{algo}_runs", index=False
                )
        return buf.getvalue()

    def to_report_zip(
        self, *,
        reference_cost: float | None = None,
        success_tol_pct: float = 1.0,
        include_inputs: bool = True,
    ) -> bytes:
        """Full reproducibility report (REPORT.md + figures + data)."""
        return build_report_zip(
            inp_path=self.request.inp_path,
            d_path=self.request.d_path,
            cost_path=self.request.cost_path,
            problem_snapshot=self.problem_snapshot,
            experiment=self.experiment,
            best_algo=self.best_algo,
            best_run_index=self.best_run_index,
            best_position_m=self.best_position_m,
            node_pressures=self.node_pressures_m,
            pipe_velocities=self.pipe_velocities_ms,
            hydraulics_converged=self.hydraulics_converged,
            config=_request_to_config_dict(self.request),
            include_inputs=include_inputs,
            reference_cost=reference_cost,
            success_tol_pct=success_tol_pct,
        )


# =====================================================================
#  Internal helpers
# =====================================================================

def _build_algorithm_config(request: OptimizationRequest) -> AlgorithmConfig:
    """Convert OptimizationRequest → AlgorithmConfig (with defaults)."""
    cfg = build_algorithm_config(
        NS=int(request.ns), MaxGen=int(request.max_gen)
    )
    if request.ga is not None:
        cfg.GA = replace(request.ga)
    if request.pso is not None:
        cfg.PSO = replace(request.pso)
    if request.de is not None:
        cfg.DE = replace(request.de)
    if request.sade is not None:
        cfg.SADE = replace(request.sade)
    if request.restart is not None:
        cfg.Restart = replace(request.restart)
    return cfg


def _request_to_config_dict(request: OptimizationRequest) -> dict:
    """Flat config dict, expected by ``build_snapshot`` and
    ``build_report_zip`` (keys must match their hard-coded lists)."""
    cfg = _build_algorithm_config(request)
    return {
        "ns": request.ns,
        "max_gen": request.max_gen,
        "n_runs": request.n_runs,
        "exp_seed": request.seed,
        "ga_pc": cfg.GA.Pc,
        "ga_pm": cfg.GA.Pm,
        "ga_sel": cfg.GA.Selection,
        "pso_w": cfg.PSO.w,
        "pso_c1": cfg.PSO.c1,
        "pso_c2": cfg.PSO.c2,
        "de_f": cfg.DE.F,
        "de_cr": cfg.DE.CR,
        "de_strategy": cfg.DE.Strategy,
        "sade_fmin": cfg.SADE.Fmin,
        "sade_fmax": cfg.SADE.Fmax,
        "sade_crmin": cfg.SADE.CRmin,
        "sade_crmax": cfg.SADE.CRmax,
        "sade_conv": cfg.SADE.UseConvergence,
        "sade_tol": cfg.SADE.Tol,
        "sade_stag": cfg.SADE.StagnationMax,
        "rs_enabled": cfg.Restart.Enabled,
        "rs_patience": cfg.Restart.Patience,
        "rs_max": cfg.Restart.MaxRestarts,
        "rs_keep_best": cfg.Restart.KeepBest,
        "rs_mode": cfg.Restart.Mode,
        "rs_perturb": cfg.Restart.PerturbRate,
    }


def _pick_best_run(
    experiment: ExperimentResult,
) -> tuple:
    """Return (best_record, best_algo); (None, None) if no usable run.

    Uses the same rule as the GUI:
    1. Prefer feasible over infeasible.
    2. Among runs with the same feasibility, prefer lower cost.
    Both "Success" and "Partial" (cancelled mid-run with a best-so-far)
    runs are eligible.
    """
    best_record = None
    best_algo = None
    for algo, runs in experiment.runs.items():
        for r in runs:
            if r.status not in ("Success", "Partial"):
                continue
            if best_record is None:
                best_record, best_algo = r, algo
                continue
            if r.feasible and not best_record.feasible:
                best_record, best_algo = r, algo
            elif (r.feasible == best_record.feasible
                    and r.score < best_record.score):
                best_record, best_algo = r, algo
    return best_record, best_algo


# =====================================================================
#  Public entry point
# =====================================================================

def run_optimization(
    request: OptimizationRequest,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> OptimizationResult:
    """Execute one optimisation run end-to-end.

    Steps
    -----
    1. Validate the request.
    2. Load the network and build the problem (via ``build_problem``).
    3. Run the experiment (``run_experiment`` from runner.py).
    4. Select the best run (feasible first, then lower cost).
    5. Compute hydraulics of that best design.
    6. Package into ``OptimizationResult``.

    Parameters
    ----------
    request : OptimizationRequest
    progress_callback : callable(info: dict) -> None, optional
        Called at each generation with the same payload the GUI's live
        progress panel uses.
    cancel_callback : callable() -> bool, optional
        Called frequently; returning True triggers a graceful stop.
        If a "Partial" run exists after the stop, its best design is
        returned.  If no run produced any result, this function raises
        ``RuntimeError``.

    Returns
    -------
    OptimizationResult

    Raises
    ------
    ValueError
        If the request is invalid (see ``OptimizationRequest.validate``).
    RuntimeError
        If no run produced any usable result.
    """
    request.validate()

    t0 = time.perf_counter()

    problem, units, wn = build_problem(
        inp_path=request.inp_path,
        d_path=request.d_path,
        cost_path=request.cost_path,
        diam_unit=request.diam_unit,
        constraints=request.constraints,
        fixed_pipes=request.fixed_pipes,
    )

    base_cfg = _build_algorithm_config(request)

    experiment = run_experiment(
        problem, base_cfg,
        number_of_runs=int(request.n_runs),
        algorithms=list(request.algorithms),
        experiment_seed=int(request.seed),
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )

    best_record, best_algo = _pick_best_run(experiment)
    if best_record is None:
        errs = [
            r.error_message
            for rs in experiment.runs.values() for r in rs
            if r.status == "Failed" and r.error_message
        ]
        raise RuntimeError(
            "No run produced a usable result." +
            (f" First error: {errs[0]}" if errs else "")
        )

    # Hydraulics of the best design (position is already a full-
    # diameter vector in metres; fixed pipes retain their InitialD).
    Pj, Vp, conv_ok = run_hydraulic_simulation(
        wn, best_record.position,
        problem["pipe_names"], problem["junction_names"],
    )

    c = problem["Constraints"]
    problem_snapshot = {
        "pipe_names": list(problem["pipe_names"]),
        "junction_names": list(problem["junction_names"]),
        "initial_D": problem["InitialD"].copy(),
        "D_m":       problem["D"].copy(),
        "Din":       problem["Din"].copy(),
        "Cost":      problem["Cost"].copy(),
        "L":         problem["L"].copy(),
        "VariablePipes": problem["VariablePipes"].copy(),
        "FixedPipes":    problem["FixedPipes"].copy(),
        "Pmin": c["Pmin"]["value"],
        "Pmax": c["Pmax"]["value"],
        "Vmin": c["Vmin"]["value"],
        "Vmax": c["Vmax"]["value"],
        "use_pmin": c["Pmin"]["enabled"],
        "use_pmax": c["Pmax"]["enabled"],
        "use_vmin": c["Vmin"]["enabled"],
        "use_vmax": c["Vmax"]["enabled"],
        "Units": dict(units),
    }

    return OptimizationResult(
        experiment=experiment,
        problem_snapshot=problem_snapshot,
        units=units,
        best_algo=best_algo,
        best_run_index=best_record.run_index,
        best_position_m=best_record.position,
        best_score=best_record.score,
        best_feasible=best_record.feasible,
        node_pressures_m=Pj,
        pipe_velocities_ms=Vp,
        hydraulics_converged=conv_ok,
        request=request,
        elapsed_seconds=time.perf_counter() - t0,
    )