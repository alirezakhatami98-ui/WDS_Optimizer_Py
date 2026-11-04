"""WDS Optimizer — Streamlit application (Phase 2)."""

from __future__ import annotations

# ---------------------------------------------------------------------
# Suppress a benign Windows-only asyncio warning that fires after a
# download completes:
#     ConnectionResetError: [WinError 10054] ...
#     File ".../asyncio/proactor_events.py", line 165, in _call_connection_lost
# The file downloads successfully; the browser closes the TCP socket a
# moment before ProactorEventLoop tries to shut it down, and asyncio
# logs the resulting error.  Filtering it keeps the console clean.
# ---------------------------------------------------------------------
import logging

class _SuppressWinError10054(logging.Filter):
    def filter(self, record):
        if record.exc_info and record.exc_info[0] is ConnectionResetError:
            return False
        if "WinError 10054" in record.getMessage():
            return False
        return True

logging.getLogger("asyncio").addFilter(_SuppressWinError10054())


import io
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import wntr

from wds_optimizer.algorithms.config import build_algorithm_config
from wds_optimizer.data.loaders import load_optimization_data
from wds_optimizer.experiments.runner import (
    ExperimentResult,
    run_experiment,
    summarize_experiment,
)
from wds_optimizer.hydraulics.simulation import run_hydraulic_simulation
from wds_optimizer.hydraulics.units import detect_network_units
from wds_optimizer.hydraulics.units import (
    detect_network_units,
    pressure_from_si, pressure_to_si,
    velocity_from_si, velocity_to_si,
    diameter_to_display,
)
from wds_optimizer.optimization.problem import build_optimization_problem
from wds_optimizer.app.network_map import build_network_figure
from wds_optimizer.app.state import init_state, reset_experiment
from wds_optimizer.app.jobs import BackgroundJob
from wds_optimizer.persistence.snapshot import (
    build_snapshot, load_snapshot, restore_experiment,
)
from wds_optimizer.sweep.config import SWEEP_PARAMETERS, parse_sweep_values
from wds_optimizer.sweep.runner import run_parameter_sweep
from wds_optimizer.app.compare import (
    compute_comparison_stats, overall_stats,
)
from wds_optimizer.reporting.report import build_report_zip

# =====================================================================
#  Page config & cached resources
# =====================================================================

st.set_page_config(
    page_title="WDS Optimizer",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _load_network(inp_path: str):
    """Load (and cache) a WNTR network model."""
    wn = wntr.network.WaterNetworkModel(inp_path)
    return wn


@st.cache_data(show_spinner=False)
def _read_uploaded(uploaded_bytes: bytes, name: str) -> str:
    """Save uploaded bytes to a stable temp file and return the path."""
    tmpdir = Path(tempfile.gettempdir()) / "wds_optimizer_uploads"
    tmpdir.mkdir(parents=True, exist_ok=True)
    p = tmpdir / name
    p.write_bytes(uploaded_bytes)
    return str(p)

@st.cache_data(show_spinner=False)
def _get_network_units(inp_path: str):
    """Detect (and cache) the unit system of an .inp file."""
    import wntr
    wn = wntr.network.WaterNetworkModel(inp_path)
    return detect_network_units(wn)

# =====================================================================
#  UI helpers
# =====================================================================

_BUTTON_CSS = """
<style>
/* Start buttons: green */
.st-key-run_start button, .st-key-sweep_start button {
    background-color: #2f9e44 !important; border-color: #2b8a3e !important;
}
.st-key-run_start button:hover, .st-key-sweep_start button:hover {
    background-color: #2b8a3e !important;
}
/* Cancel buttons: red */
.st-key-run_stop button, .st-key-sweep_stop button {
    background-color: #e03131 !important; border-color: #c92a2a !important;
}
.st-key-run_stop button:hover, .st-key-sweep_stop button:hover {
    background-color: #c92a2a !important;
}
/* "Stopping..." state: amber */
.st-key-run_stopping button, .st-key-sweep_stopping button {
    background-color: #f08c00 !important; border-color: #e67700 !important;
}
.st-key-run_start button *, .st-key-sweep_start button *,
.st-key-run_stop button *,  .st-key-sweep_stop button *,
.st-key-run_stopping button *, .st-key-sweep_stopping button * {
    color: #ffffff !important;
}
.st-key-run_start button:disabled, .st-key-sweep_start button:disabled {
    background-color: #adb5bd !important; border-color: #adb5bd !important;
    opacity: 0.6;
}
</style>
"""


def _job_button(kind: str, start_label: str, stop_label: str, *,
                enabled: bool) -> bool:
    """Green start button that turns into a red cancel button while its
    job is running.  Returns True only when START was clicked."""
    job = st.session_state.get("job")
    if job is not None and job.kind == kind:
        if job.cancel_requested:
            st.button("⏳ Stopping… saving results", key=f"{kind}_stopping",
                      type="primary", width="stretch", disabled=True)
        elif st.button(stop_label, key=f"{kind}_stop",
                       type="primary", width="stretch"):
            job.cancel()
            st.rerun()          # redraw immediately in the "Stopping" state
        return False
    return st.button(
        start_label, key=f"{kind}_start", type="primary", width="stretch",
        disabled=(not enabled) or (job is not None),   # one job at a time
    )


def _header():
    st.markdown(_BUTTON_CSS, unsafe_allow_html=True)
    st.title("💧 WDS Optimizer")
    st.caption(
        "Water distribution network optimization using GA and PSO, "
        "powered by WNTR/EPANET. Python port of the MATLAB toolkit."
    )

def _init_widget_state():
    """Initialize widget keys + apply any pending restore.

    Streamlit forbids modifying a widget's session_state key after the
    widget was created in the same render pass. The Load button runs at
    the bottom of the sidebar (after all widgets are created), so it
    cannot set widget keys directly. Instead it queues changes in
    ``_pending_restore`` and calls ``st.rerun()``. This function applies
    them at the top of the next pass, *before* the widgets are created.
    """
    # Apply any queued restore first (before widgets are instantiated).
    pending = st.session_state.pop("_pending_restore", None)
    if pending:
        for k, v in pending.items():
            st.session_state[k] = v

    # Seed defaults for keys that still don't exist.
    defaults = {
        "ns_input":          int(st.session_state.get("ns", 50)),
        "max_gen_input":     int(st.session_state.get("max_gen", 100)),
        "pmin_input":        float(st.session_state.get("pmin", 30.0)),
        "vmax_input":        float(st.session_state.get("vmax", 2.5)),
        "fixed_pipes_input": st.session_state.get("fixed_pipes", ""),
        "ga_pc_input":       float(st.session_state.get("ga_pc", 0.8)),
        "ga_pm_input":       float(st.session_state.get("ga_pm", 0.03)),
        "ga_sel_input":      st.session_state.get("ga_sel", "tournament"),
        "pso_w_input":       float(st.session_state.get("pso_w", 0.7)),
        "pso_c1_input":      float(st.session_state.get("pso_c1", 1.5)),
        "pso_c2_input":      float(st.session_state.get("pso_c2", 1.5)),
        "n_runs_input":      int(st.session_state.get("n_runs", 3)),
        "exp_seed_input":    int(st.session_state.get("exp_seed", 1)),
        "algo_input":        "GA",
        "rs_enabled_input":  bool(st.session_state.get("rs_enabled", False)),
        "rs_patience_input": int(st.session_state.get("rs_patience", 30)),
        "rs_max_input":      int(st.session_state.get("rs_max", 3)),
        "rs_keep_best_input": bool(
            st.session_state.get("rs_keep_best", False)),
        "rs_mode_input":     st.session_state.get("rs_mode", "perturb"),
        "rs_perturb_input":  float(st.session_state.get("rs_perturb", 0.3)),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _render_sidebar():
    _init_widget_state()          # ← این خط اضافه شود
    with st.sidebar:
        st.header("1. Input Files")

        inp_upload = st.file_uploader(
            "EPANET .inp file", type=["inp"], key="inp_upload"
        )
        d_upload = st.file_uploader(
            "Diameters (D.txt)", type=["txt"], key="d_upload"
        )
        cost_upload = st.file_uploader(
            "Costs (Cost.txt)", type=["txt"], key="cost_upload"
        )
        diam_unit = st.selectbox(
            "D.txt unit", ["in", "mm"], index=0, key="diam_unit_input"
        )

        if inp_upload and d_upload and cost_upload:
            inp_path = _read_uploaded(inp_upload.getvalue(), inp_upload.name)
            d_path = _read_uploaded(d_upload.getvalue(), d_upload.name)
            cost_path = _read_uploaded(
                cost_upload.getvalue(), cost_upload.name
            )
            st.session_state.inp_path = inp_path
            st.session_state.d_path = d_path
            st.session_state.cost_path = cost_path
            st.session_state.diam_unit = diam_unit
            st.success("All files loaded.")
        elif st.session_state.inp_path:
            st.info("Using previously loaded files.")

        st.divider()
        st.header("2. Hydraulics & Constraints")

        units = None
        if st.session_state.inp_path:
            try:
                units = _get_network_units(st.session_state.inp_path)
            except Exception:
                units = None
        st.session_state.network_units = units

        # Auto-convert Pmin/Vmax when the unit system changes.
        # Both canonical keys and widget keys are updated; widgets below
        # are rendered with `key=` only (no `value=`), so this is safe.
        prev = st.session_state.get("network_units_prev")
        if units is not None and prev is not None:
            if prev.get("System") != units.get("System"):
                st.session_state.pmin = pressure_from_si(
                    pressure_to_si(st.session_state.pmin, prev), units
                )
                st.session_state.vmax = velocity_from_si(
                    velocity_to_si(st.session_state.vmax, prev), units
                )
                st.session_state.pmin_input = float(st.session_state.pmin)
                st.session_state.vmax_input = float(st.session_state.vmax)
                st.info(
                    f"Pressure and velocity values converted to "
                    f"{units['System']} units "
                    f"({units['PressureUnit']}, {units['VelocityUnit']})."
                )
        st.session_state.network_units_prev = units

        p_label = (
            f"Min pressure ({units['PressureUnit']})" if units
            else "Min pressure (m)"
        )
        v_label = (
            f"Max velocity ({units['VelocityUnit']})" if units
            else "Max velocity (m/s)"
        )
        if units and units["System"] == "US":
            st.caption(
                f"Network uses US customary units: "
                f"pressure in {units['PressureUnit']}, "
                f"velocity in {units['VelocityUnit']}."
            )

        st.number_input(
            "Population / Swarm size (NS)",
            min_value=4, max_value=500, step=1,
            key="ns_input",
        )
        st.number_input(
            "Max generations / iterations",
            min_value=5, max_value=5000, step=1,
            key="max_gen_input",
        )
        st.number_input(
            p_label, min_value=0.0, step=1.0,
            key="pmin_input",
        )
        st.number_input(
            v_label, min_value=0.1, step=0.1,
            key="vmax_input",
        )
        st.text_input(
            "Fixed pipe IDs (e.g. '1, 3')",
            key="fixed_pipes_input",
        )

        st.divider()
        st.header("3. Algorithm")
        # Algorithm
        algo = st.radio(
            "Algorithm", ["GA", "PSO", "Both"],
            key="algo_input", horizontal=True,
        )

        if algo in ("GA", "Both"):
            with st.expander("GA parameters", expanded=True):
                st.number_input("Pc (crossover)", 0.0, 1.0, step=0.05,
                                key="ga_pc_input")
                st.number_input("Pm (mutation)", 0.0, 1.0, step=0.01,
                                key="ga_pm_input")
                st.selectbox("Selection", ["tournament", "roulette"],
                             key="ga_sel_input")

        if algo in ("PSO", "Both"):
            with st.expander("PSO parameters", expanded=True):
                st.number_input("w (inertia)", 0.0, 2.0, step=0.05,
                                key="pso_w_input")
                st.number_input("c1 (cognitive)", 0.0, 3.0, step=0.1,
                                key="pso_c1_input")
                st.number_input("c2 (social)", 0.0, 3.0, step=0.1,
                                key="pso_c2_input")

        with st.expander("Restart on stagnation", expanded=False):
            rs_on = st.checkbox(
                "Enable restart", key="rs_enabled_input",
                help="Re-initialise the population when the search "
                     "stagnates. The best solution found so far is always "
                     "kept, and the total number of generations does not "
                     "change.",
            )
            st.number_input(
                "Patience (generations without improvement)",
                min_value=2, max_value=5000, step=1,
                key="rs_patience_input", disabled=not rs_on,
            )
            st.number_input(
                "Max restarts per run", min_value=1, max_value=50, step=1,
                key="rs_max_input", disabled=not rs_on,
            )
            rs_mode = st.radio(
                "Restart type",
                ["perturb", "random"],
                format_func=lambda m: {
                    "perturb": "Perturb best design",
                    "random": "Random population",
                }[m],
                key="rs_mode_input", disabled=not rs_on,
                help="Perturb: new population = copies of the best design "
                     "with a share of the pipes re-drawn (keeps what is "
                     "already good, explores nearby basins). Random: "
                     "discards everything and starts over; each new epoch "
                     "must climb again from scratch.",
            )
            st.slider(
                "Pipes re-drawn (perturb)", 0.05, 0.9, step=0.05,
                key="rs_perturb_input",
                disabled=(not rs_on) or rs_mode != "perturb",
            )
            st.checkbox(
                "Keep best solution in the new population",
                key="rs_keep_best_input",
                disabled=(not rs_on) or rs_mode != "random",
                help="Random type only. Off: fully independent restart "
                     "(more exploration). On: the best design seeds the "
                     "new population.",
            )
            if rs_on:
                _pat = int(st.session_state.get("rs_patience_input", 30))
                _mg = int(st.session_state.get("max_gen_input", 100))
                _fit = max(0, _mg // (_pat + 1) - 1)
                st.caption(
                    f"Budget is unchanged (MaxGen = {_mg}). At most ~{_fit} "
                    f"restart(s) can fit with this patience."
                )

        st.divider()
        st.header("4. Experiment")
        st.number_input("Runs per algorithm", min_value=1, max_value=50,
                        step=1, key="n_runs_input")
        st.number_input("Experiment seed", min_value=0,
                        max_value=2**31 - 1, step=1, key="exp_seed_input")

        st.divider()
        can_run = bool(st.session_state.inp_path)
        busy = st.session_state.get("job") is not None
        run_clicked = _job_button(
            "run", "🚀 Run Experiment", "⏹ Cancel experiment",
            enabled=can_run,
        )
        if not can_run:
            st.caption("Load all three input files to enable.")
        if st.session_state.experiment is not None:
            if st.button("🗑 Clear results", width="stretch", disabled=busy):
                reset_experiment()
                st.rerun()

        # ---- 5. Save / Load ----
        st.divider()
        st.header("5. Save / Load")

        if st.session_state.experiment is not None:
            try:
                inp_content = Path(st.session_state.inp_path).read_text(
                    encoding="utf-8", errors="replace"
                )
                d_content = Path(st.session_state.d_path).read_text(
                    encoding="utf-8", errors="replace"
                )
                cost_content = Path(st.session_state.cost_path).read_text(
                    encoding="utf-8", errors="replace"
                )
                snap = build_snapshot(
                    problem_snapshot=st.session_state.problem_snapshot,
                    experiment=st.session_state.experiment,
                    best_algo=st.session_state.best_algo,
                    best_run_index=st.session_state.best_run_index,
                    best_position_m=st.session_state.best_position_m,
                    node_pressures=st.session_state.node_pressures,
                    pipe_velocities=st.session_state.pipe_velocities,
                    hydraulics_converged=st.session_state.hydraulics_converged,
                    config={k: st.session_state[k] for k in (
                        "ns", "max_gen", "ga_pc", "ga_pm", "ga_sel",
                        "pso_w", "pso_c1", "pso_c2", "n_runs", "exp_seed",
                        "rs_enabled", "rs_patience", "rs_max",
                        "rs_keep_best", "rs_mode", "rs_perturb",
                    )},
                    inp_content=inp_content,
                    d_content=d_content,
                    cost_content=cost_content,
                )
                import json as _json
                st.download_button(
                    "💾 Save experiment (.json)",
                    data=_json.dumps(snap, indent=2, ensure_ascii=False)
                          .encode("utf-8"),
                    file_name=f"wds_experiment_{datetime.now():%Y%m%d_%H%M%S}.json",
                    mime="application/json",
                    width="stretch",
                )
            except Exception as e:
                st.caption(f"Cannot save: {e}")

        load_upload = st.file_uploader(
            "Load experiment (.json)", type=["json"], key="load_json"
        )
        if load_upload is not None:
            if st.button("Load selected file", width="stretch",
                         disabled=busy):
                try:
                    data = load_snapshot(load_upload)
                    _restore_from_snapshot(data)   # includes st.rerun()
                except Exception as e:
                    st.error(f"Load failed: {e}")

        # ---- 6. Sweep ----
        st.divider()
        st.header("6. Parameter Sweep")

        with st.expander("Configure sweep", expanded=False):
            _units_for_sweep = st.session_state.get("network_units")
            sweep_param = st.selectbox(
                "Parameter",
                list(SWEEP_PARAMETERS.keys()),
                format_func=lambda k: _sweep_param_label(k, _units_for_sweep),
                key="sweep_param_input",
            )
            sweep_vals_str = st.text_input(
                "Values (comma-separated)", value="",
                key="sweep_vals_input",
                placeholder="e.g. 0.6, 0.7, 0.8, 0.9",
            )
            sweep_algo = st.multiselect(
                "Algorithms", ["GA", "PSO"], default=["GA"],
                key="sweep_algo_input",
            )
            sweep_runs = st.number_input(
                "Runs per point", 1, 20, value=1, step=1,
                key="sweep_runs_input",
            )

        sweep_go = _job_button(
            "sweep", "🔍 Run sweep", "⏹ Cancel sweep",
            enabled=bool(st.session_state.inp_path),
        )

        sweep_config = None
        if sweep_go:
            try:
                vals = parse_sweep_values(sweep_vals_str)
                if not sweep_algo:
                    raise ValueError("Select at least one algorithm.")
                sweep_config = {
                    "param_name": sweep_param,
                    "values": vals,
                    "algorithms": sweep_algo,
                    "n_runs": int(sweep_runs),
                }
            except ValueError as e:
                st.error(f"Invalid sweep config: {e}")

        # ---- 7. Reproducibility Report ----
        st.divider()
        st.header("7. Reproducibility Report")
        if st.session_state.experiment is not None:
            if st.button("📄 Generate report (.zip)", width="stretch"):
                with st.spinner("Building report (figures, data, JSON)..."):
                    try:
                        zip_bytes = build_report_zip(
                            inp_path=st.session_state.inp_path,
                            d_path=st.session_state.d_path,
                            cost_path=st.session_state.cost_path,
                            problem_snapshot=st.session_state.problem_snapshot,
                            experiment=st.session_state.experiment,
                            best_algo=st.session_state.best_algo,
                            best_run_index=st.session_state.best_run_index,
                            best_position_m=st.session_state.best_position_m,
                            node_pressures=st.session_state.node_pressures,
                            pipe_velocities=st.session_state.pipe_velocities,
                            hydraulics_converged=st.session_state.hydraulics_converged,
                            config={k: st.session_state[k] for k in (
                                "ns", "max_gen", "ga_pc", "ga_pm", "ga_sel",
                                "pso_w", "pso_c1", "pso_c2", "n_runs", "exp_seed",
                        "rs_enabled", "rs_patience", "rs_max",
                        "rs_keep_best", "rs_mode", "rs_perturb",
                            )},
                        )
                        st.session_state.report_zip = zip_bytes
                        st.session_state.report_filename = (
                            f"wds_report_{datetime.now():%Y%m%d_%H%M%S}.zip"
                        )
                        st.success("Report ready — download below.")
                    except Exception as e:
                        st.error(f"Report failed: {type(e).__name__}: {e}")

            if st.session_state.get("report_zip"):
                st.download_button(
                    "⬇ Download report (.zip)",
                    data=st.session_state.report_zip,
                    file_name=st.session_state.report_filename,
                    mime="application/zip",
                    width="stretch",
                    key="report_download",
                )
        else:
            st.caption("Run an experiment first.")

    return {
        "run_clicked": run_clicked,
        "algo_label": algo,         # ← درست: algo همان مقدار st.radio است
        "sweep": sweep_config,
    }

def _sweep_param_label(key: str, units: dict | None = None) -> str:
    """Return a sweep-parameter label with the network's units."""
    if key == "Pmin":
        u = (units or {}).get("PressureUnit", "m")
        return f"Min pressure ({u})"
    if key == "Vmax":
        u = (units or {}).get("VelocityUnit", "m/s")
        return f"Max velocity ({u})"
    return SWEEP_PARAMETERS.get(key, key)


# =====================================================================
#  Preparation: build the Problem dict from loaded files
# =====================================================================

def _prepare_problem():
    wn = _load_network(st.session_state.inp_path)
    units = detect_network_units(wn)

    Din, Cost, D_m = load_optimization_data(
        st.session_state.d_path,
        st.session_state.cost_path,
        diam_file_unit=st.session_state.diam_unit,
    )

    pipe_names = list(wn.pipe_name_list)
    junction_names = list(wn.junction_name_list)
    lengths = np.array([wn.get_link(n).length for n in pipe_names])
    initial_D = np.array([wn.get_link(n).diameter for n in pipe_names])

    # Parse fixed pipes
    fixed_str = (st.session_state.get("fixed_pipes", "") or "").strip()
    if fixed_str:
        try:
            fixed_1based = [
                int(x.strip()) for x in fixed_str.split(",") if x.strip()
            ]
        except ValueError:
            raise ValueError(
                "Fixed pipe IDs must be integers separated by commas."
            )
        if any(f < 1 or f > len(pipe_names) for f in fixed_1based):
            raise ValueError(
                f"Fixed pipe IDs must be between 1 and {len(pipe_names)}."
            )
        if len(set(fixed_1based)) != len(fixed_1based):
            raise ValueError("Fixed pipe IDs must be unique.")
        fixed_0based = [f - 1 for f in sorted(fixed_1based)]
    else:
        fixed_0based = []

    variable_0based = [
        i for i in range(len(pipe_names)) if i not in fixed_0based
    ]

    # Convert Pmin/Vmax from user-facing (network) units to SI for WNTR
    Pmin_si = pressure_to_si(st.session_state.pmin, units)
    Vmax_si = velocity_to_si(st.session_state.vmax, units)

    problem = build_optimization_problem(
        wn=wn, Din=Din, Cost=Cost, D_m=D_m,
        full_lengths=lengths, initial_D=initial_D,
        fixed_pipes=fixed_0based,
        variable_pipes=variable_0based,
        Pmin=Pmin_si,
        Vmax=Vmax_si,
        pipe_names=pipe_names,
        junction_names=junction_names,
        units=units,
    )
    return problem, units, wn


# =====================================================================
#  Run experiment with live progress
# =====================================================================

def _sync_sidebar_state(with_n_runs: bool = True):
    """Copy sidebar widget values into the canonical session_state keys.

    Only widgets that were actually rendered exist (e.g. PSO widgets are
    absent when the user picked "GA" only).
    """
    pairs = [
        ("ns", "ns_input"), ("max_gen", "max_gen_input"),
        ("pmin", "pmin_input"), ("vmax", "vmax_input"),
        ("fixed_pipes", "fixed_pipes_input"), ("exp_seed", "exp_seed_input"),
        ("ga_pc", "ga_pc_input"), ("ga_pm", "ga_pm_input"),
        ("ga_sel", "ga_sel_input"),
        ("pso_w", "pso_w_input"), ("pso_c1", "pso_c1_input"),
        ("pso_c2", "pso_c2_input"),
        ("rs_enabled", "rs_enabled_input"),
        ("rs_patience", "rs_patience_input"),
        ("rs_max", "rs_max_input"),
        ("rs_keep_best", "rs_keep_best_input"),
        ("rs_mode", "rs_mode_input"),
        ("rs_perturb", "rs_perturb_input"),
    ]
    if with_n_runs:
        pairs.append(("n_runs", "n_runs_input"))
    for dst, src in pairs:
        if src in st.session_state:
            st.session_state[dst] = st.session_state[src]


def _build_base_cfg():
    cfg = build_algorithm_config(
        int(st.session_state.ns), int(st.session_state.max_gen)
    )
    cfg.GA.Pc = st.session_state.ga_pc
    cfg.GA.Pm = st.session_state.ga_pm
    cfg.GA.Selection = st.session_state.ga_sel
    cfg.PSO.w = st.session_state.pso_w
    cfg.PSO.c1 = st.session_state.pso_c1
    cfg.PSO.c2 = st.session_state.pso_c2
    cfg.Restart.Enabled = bool(st.session_state.rs_enabled)
    cfg.Restart.Patience = int(st.session_state.rs_patience)
    cfg.Restart.MaxRestarts = int(st.session_state.rs_max)
    cfg.Restart.KeepBest = bool(st.session_state.rs_keep_best)
    cfg.Restart.Mode = str(st.session_state.rs_mode)
    cfg.Restart.PerturbRate = float(st.session_state.rs_perturb)
    if cfg.Restart.Enabled and cfg.Restart.Patience >= cfg.MaxGen:
        raise ValueError(
            f"Restart patience ({cfg.Restart.Patience}) must be smaller "
            f"than MaxGen ({cfg.MaxGen}); otherwise no restart can happen."
        )
    return cfg


# =====================================================================
#  Background jobs: start -> live panel (fragment) -> finalize
# =====================================================================
# The optimisation runs in a worker thread (see app/jobs.py).  The worker
# never calls st.*; the UI polls the job every 0.5 s from a fragment and,
# once the thread has ended (normally OR after Cancel), stores whatever
# result exists and reruns the app.

def _start_run(algo_label: str):
    _sync_sidebar_state(with_n_runs=True)

    # Clear any stale report from a previous experiment
    st.session_state.report_zip = None
    st.session_state.report_filename = None

    problem, units, wn = _prepare_problem()
    base_cfg = _build_base_cfg()
    algos = {"GA": ["GA"], "PSO": ["PSO"]}.get(algo_label, ["GA", "PSO"])
    n_runs = int(st.session_state.n_runs)
    seed = int(st.session_state.exp_seed)

    summary = (
        f"Network: **{len(problem['pipe_names'])}** pipes, "
        f"**{len(problem['junction_names'])}** junctions, "
        f"**{len(problem['VariablePipes'])}** variable, "
        f"**{len(problem['FixedPipes'])}** fixed  |  "
        f"Algorithms: **{', '.join(algos)}** × **{n_runs}** runs, "
        f"NS={st.session_state.ns}, MaxGen={st.session_state.max_gen}"
        + (f"  |  Restart ({base_cfg.Restart.Mode}): patience "
           f"**{base_cfg.Restart.Patience}**, "
           f"max **{base_cfg.Restart.MaxRestarts}**"
           if base_cfg.Restart.Enabled else "")
    )

    def target(progress_cb, cancel_cb):
        return run_experiment(
            problem, base_cfg,
            number_of_runs=n_runs,
            algorithms=algos,
            experiment_seed=seed,
            progress_callback=progress_cb,
            cancel_callback=cancel_cb,
        )

    job = BackgroundJob(
        "run", label="Experiment",
        context={"problem": problem, "units": units, "wn": wn,
                 "algos": algos, "n_runs": n_runs, "summary": summary,
                 "restart_on": bool(base_cfg.Restart.Enabled)},
    )
    job.start(target)
    st.session_state.job = job
    st.session_state.job_notice = None
    st.rerun()      # redraw sidebar: green button -> red "Cancel"


def _start_sweep(param_name, values, algorithms, n_runs):
    _sync_sidebar_state(with_n_runs=False)

    problem, units, wn = _prepare_problem()
    base_cfg = _build_base_cfg()
    seed = int(st.session_state.exp_seed)
    label = _sweep_param_label(param_name, units)

    def target(progress_cb, cancel_cb):
        return run_parameter_sweep(
            problem, base_cfg,
            parameter_name=param_name,
            values=values,
            algorithms=algorithms,
            n_runs=int(n_runs),
            experiment_seed=seed,
            progress_callback=progress_cb,
            cancel_callback=cancel_cb,
        )

    job = BackgroundJob(
        "sweep", label=f"Sweep: {label}",
        context={"param_name": param_name, "values": list(values),
                 "algorithms": list(algorithms), "n_runs": int(n_runs),
                 "summary": (f"Sweeping **{label}** over **{len(values)}** "
                             f"values: `{list(values)}` | Algorithms: "
                             f"**{', '.join(algorithms)}**, runs/point: "
                             f"**{n_runs}**, seed: **{seed}**")},
    )
    job.start(target)
    st.session_state.job = job
    st.session_state.job_notice = None
    st.rerun()


@st.fragment(run_every=0.5)
def _job_panel():
    """Live progress while a job runs; finalises it when the thread ends."""
    job = st.session_state.get("job")
    if job is None:
        return
    job.touch()                       # heartbeat (see BackgroundJob)

    if job.is_done:
        if job.kind == "run":
            _finalize_run(job)
        else:
            _finalize_sweep(job)
        return                        # (finalize ends with st.rerun)

    ctx = job.context
    prog = job.get_progress()
    st.subheader("Live progress")
    st.markdown(ctx.get("summary", ""))

    if job.cancel_requested:
        st.warning(
            "Stopping after the current step — results computed so far "
            "will be kept."
        )

    if job.kind == "run":
        algos, n_runs = ctx["algos"], ctx["n_runs"]
        if "Generation" not in prog:
            st.progress(0.0)
            st.caption("Starting…")
            return
        a_idx = algos.index(prog["Algorithm"]) if prog["Algorithm"] in algos else 0
        gen_frac = min(1.0, prog["Generation"] / max(prog["MaxGenerations"], 1))
        frac = (a_idx * n_runs + (prog["RunIndex"] - 1) + gen_frac) / (
            len(algos) * n_runs
        )
        st.progress(min(1.0, frac))
        feas = "Feasible" if prog["Feasible"] else "Infeasible"
        st.info(
            f"**{prog['RunLabel']}** | Gen **{prog['Generation']}/"
            f"{prog['MaxGenerations']}** | Best cost: "
            f"**{prog['BestCost']:,.0f}** | {feas} | "
            + (f"Restarts: **{prog.get('Restarts', 0)}** | "
               if ctx.get("restart_on") else "")
            + f"elapsed {job.elapsed:.0f} s"
        )
        conv = prog["Convergence"]
        color = "#d9480f" if prog["Algorithm"] == "GA" else "#1971c2"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(conv) + 1), y=conv, mode="lines",
            line=dict(color=color, width=2), name=prog["RunLabel"],
        ))
        for g in prog.get("RestartGens", []):
            fig.add_vline(x=g, line_dash="dot", line_color="gray",
                          opacity=0.7)
        fig.update_layout(
            height=280, margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Generation / Iteration", yaxis_title="Best cost",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    else:  # sweep
        total = prog.get("TotalPoints", len(ctx["values"]))
        pi = prog.get("PointIndex", 1)
        inner = prog.get("ProgressInfo") or {}
        frac_in_point = 0.0
        if inner.get("Generation"):
            n_alg = max(len(ctx["algorithms"]), 1)
            a_idx = (ctx["algorithms"].index(inner["Algorithm"])
                     if inner.get("Algorithm") in ctx["algorithms"] else 0)
            gen_frac = inner["Generation"] / max(inner["MaxGenerations"], 1)
            frac_in_point = (
                a_idx * ctx["n_runs"] + (inner.get("RunIndex", 1) - 1)
                + gen_frac
            ) / (n_alg * ctx["n_runs"])
        if prog.get("Phase") == "PointDone":
            frac_in_point = 1.0
        st.progress(min(1.0, ((pi - 1) + frac_in_point) / max(total, 1)))
        if "ParameterValue" in prog:
            st.info(
                f"Point **{pi}/{total}** | {ctx['param_name']} = "
                f"**{prog['ParameterValue']:g}** | "
                f"elapsed {job.elapsed:.0f} s"
            )
        else:
            st.caption("Starting…")


def _notice(level: str, msg: str, details: str = ""):
    st.session_state.job_notice = (level, msg, details)


def _finalize_run(job: BackgroundJob):
    """Runs in the main script thread after the worker has ended."""
    ctx = job.context
    st.session_state.job = None
    elapsed = job.elapsed

    if job.error is not None:
        _notice("error",
                f"Experiment failed: {type(job.error).__name__}: {job.error}",
                job.error_trace)
        st.rerun()

    result: ExperimentResult = job.result
    problem, units, wn = ctx["problem"], ctx["units"], ctx["wn"]
    cancelled = bool(getattr(result, "cancelled", False))

    best_record, best_algo = _pick_best_run(result)
    if best_record is None:
        errs = [r.error_message for rs in result.runs.values() for r in rs
                if r.status == "Failed" and r.error_message]
        if cancelled:
            _notice("warning",
                    f"Cancelled after {elapsed:.0f} s, before any run "
                    "produced a design — nothing to show.")
        else:
            _notice("error",
                    "No successful run." + (f" First error: {errs[0]}"
                                            if errs else ""))
        st.rerun()

    full_D = problem["InitialD"].copy()
    full_D[problem["VariablePipes"]] = problem["D"][
        _position_to_indices(best_record.position, problem)
    ]
    Pj, Vp, conv_ok = run_hydraulic_simulation(
        wn, full_D, problem["pipe_names"], problem["junction_names"]
    )

    # ---- Persist everything needed to render results ----
    st.session_state.experiment = result
    st.session_state.best_algo = best_algo
    st.session_state.best_run_index = best_record.run_index
    st.session_state.best_position_m = best_record.position
    st.session_state.best_score = best_record.score
    st.session_state.best_feasible = best_record.feasible
    st.session_state.node_pressures = Pj
    st.session_state.pipe_velocities = Vp
    st.session_state.hydraulics_converged = conv_ok

    st.session_state.problem_snapshot = {
        "pipe_names": list(problem["pipe_names"]),
        "junction_names": list(problem["junction_names"]),
        "initial_D": problem["InitialD"].copy(),
        "D_m": problem["D"].copy(),
        "Din": problem["Din"].copy(),
        "Cost": problem["Cost"].copy(),
        "L": problem["L"].copy(),
        "VariablePipes": problem["VariablePipes"].copy(),
        "FixedPipes": problem["FixedPipes"].copy(),
        "Pmin": problem["Pmin"],
        "Vmax": problem["Vmax"],
        "Units": dict(units),
    }

    runs_all = [r for rs in result.runs.values() for r in rs]
    n_ok = sum(r.status == "Success" for r in runs_all)
    n_part = sum(r.status == "Partial" for r in runs_all)
    if cancelled:
        _notice("warning",
                f"Experiment cancelled after {elapsed:.0f} s. Kept {n_ok} "
                f"completed run(s)" + (f" + {n_part} partial run(s)"
                                       if n_part else "")
                + f" out of {len(runs_all)}. The results below are partial.")
    else:
        _notice("success", f"Experiment finished in {elapsed:.1f} s")
    st.rerun()


def _finalize_sweep(job: BackgroundJob):
    st.session_state.job = None

    if job.error is not None:
        _notice("error",
                f"Sweep failed: {type(job.error).__name__}: {job.error}",
                job.error_trace)
        st.rerun()

    result = job.result
    n_pts, n_all = len(result.points), len(result.values)
    if n_pts:
        st.session_state.sweep_result = result
    if result.cancelled:
        if n_pts:
            _notice("warning",
                    f"Sweep cancelled after {job.elapsed:.0f} s — "
                    f"{n_pts} of {n_all} points kept (see the Sweep tab).")
        else:
            _notice("warning", "Sweep cancelled before any point finished.")
    else:
        _notice("success", f"Sweep complete ({n_pts} points, "
                           f"{job.elapsed:.1f} s). See the Sweep tab.")
    st.rerun()


def _show_job_notice():
    notice = st.session_state.get("job_notice")
    if not notice:
        return
    level, msg, details = notice
    {"success": st.success, "warning": st.warning,
     "error": st.error}.get(level, st.info)(msg)
    if details:
        with st.expander("Details"):
            st.code(details)



def _restore_from_snapshot(data: dict):
    """Restore session_state from a loaded snapshot dict.

    Widget keys are queued in ``_pending_restore`` and applied on the
    next render, before widgets are created (see ``_init_widget_state``).
    Non-widget state (experiment, snapshot, hydraulics) is applied
    immediately, then ``st.rerun()`` refreshes the UI.
    """
    import uuid

    prob = data["problem"]
    env = data["environment"]
    cfg = data["config"]
    best = data["best"]
    hyd = data["hydraulics"]

    # ---- Inputs ----
    inputs = data.get("inputs")
    if inputs and inputs.get("inp"):
        tmpdir = Path(tempfile.gettempdir()) / "wds_optimizer_loaded"
        tmpdir.mkdir(parents=True, exist_ok=True)
        uid = uuid.uuid4().hex[:8]

        inp_p = tmpdir / f"{uid}.inp"
        inp_p.write_text(inputs["inp"], encoding="utf-8")
        st.session_state.inp_path = str(inp_p)

        if inputs.get("D_txt"):
            d_p = tmpdir / f"{uid}_D.txt"
            d_p.write_text(inputs["D_txt"], encoding="utf-8")
            st.session_state.d_path = str(d_p)

        if inputs.get("cost_txt"):
            c_p = tmpdir / f"{uid}_Cost.txt"
            c_p.write_text(inputs["cost_txt"], encoding="utf-8")
            st.session_state.cost_path = str(c_p)

    # ---- Canonical config keys (safe to set anytime) ----
    for key in ("ns", "max_gen", "ga_pc", "ga_pm", "ga_sel",
                "pso_w", "pso_c1", "pso_c2", "n_runs", "exp_seed",
                "rs_enabled", "rs_patience", "rs_max", "rs_keep_best",
                "rs_mode", "rs_perturb"):
        if key in cfg and cfg[key] is not None:
            st.session_state[key] = cfg[key]

    units = prob["Units"]
    pmin_si = env.get("pmin_si", env.get("pmin"))
    vmax_si = env.get("vmax_si", env.get("vmax"))
    pmin_disp = pressure_from_si(pmin_si, units)
    vmax_disp = velocity_from_si(vmax_si, units)
    st.session_state.pmin = pmin_disp
    st.session_state.vmax = vmax_disp

    fixed_list = prob.get("FixedPipes", [])
    fixed_str = ", ".join(str(int(x) + 1) for x in fixed_list)
    st.session_state.fixed_pipes = fixed_str

    # ---- Queue widget key updates for next render ----
    widget_updates = {
        "ns_input":          int(cfg.get("ns", st.session_state.ns)),
        "max_gen_input":     int(cfg.get("max_gen", st.session_state.max_gen)),
        "ga_pc_input":       float(cfg.get("ga_pc", st.session_state.ga_pc)),
        "ga_pm_input":       float(cfg.get("ga_pm", st.session_state.ga_pm)),
        "ga_sel_input":      cfg.get("ga_sel", st.session_state.ga_sel),
        "pso_w_input":       float(cfg.get("pso_w", st.session_state.pso_w)),
        "pso_c1_input":      float(cfg.get("pso_c1", st.session_state.pso_c1)),
        "pso_c2_input":      float(cfg.get("pso_c2", st.session_state.pso_c2)),
        "n_runs_input":      int(cfg.get("n_runs", st.session_state.n_runs)),
        "exp_seed_input":    int(cfg.get("exp_seed", st.session_state.exp_seed)),
        # old snapshots have no restart keys -> disabled (as they were run)
        "rs_enabled_input":  bool(cfg.get("rs_enabled") or False),
        "rs_patience_input": int(cfg.get("rs_patience") or 30),
        "rs_max_input":      int(cfg.get("rs_max") or 3),
        "rs_keep_best_input": bool(cfg.get("rs_keep_best") or False),
        # files saved before restart modes existed used random restarts
        "rs_mode_input":     cfg.get("rs_mode") or "random",
        "rs_perturb_input":  float(cfg.get("rs_perturb") or 0.3),
        "pmin_input":        float(pmin_disp),
        "vmax_input":        float(vmax_disp),
        "fixed_pipes_input": fixed_str,
    }

    # Derive algo_input from the loaded experiment's algorithms.
    algos = set(data.get("experiment", {}).get("algorithms", []))
    if algos == {"GA", "PSO"}:
        widget_updates["algo_input"] = "Both"
    elif algos == {"PSO"}:
        widget_updates["algo_input"] = "PSO"
    else:
        widget_updates["algo_input"] = "GA"

    st.session_state._pending_restore = widget_updates

    # ---- Non-widget state (applied immediately) ----
    st.session_state.problem_snapshot = {
        "pipe_names": prob["pipe_names"],
        "junction_names": prob["junction_names"],
        "L": np.array(prob["L"]),
        "initial_D": np.array(prob["initial_D"]),
        "D_m": np.array(prob["D_m"]),
        "Din": np.array(prob["Din"]),
        "Cost": np.array(prob["Cost"]),
        "VariablePipes": np.array(prob["VariablePipes"]),
        "FixedPipes": np.array(prob["FixedPipes"]),
        "Pmin": prob["Pmin"],
        "Vmax": prob["Vmax"],
        "Units": prob["Units"],
    }

    st.session_state.experiment = restore_experiment(data)
    st.session_state.best_algo = best["algorithm"]
    st.session_state.best_run_index = best["run_index"]
    st.session_state.best_position_m = np.array(best["position"])
    st.session_state.node_pressures = np.array(hyd["node_pressures"])
    st.session_state.pipe_velocities = np.array(hyd["pipe_velocities"])
    st.session_state.hydraulics_converged = hyd["converged"]

    best_run = data["experiment"]["runs"][best["algorithm"]][best["run_index"] - 1]
    st.session_state.best_feasible = bool(best_run["feasible"])
    st.session_state.best_score = (
        float(best_run["score"]) if best_run["score"] is not None
        else float("inf")
    )

    st.rerun()

def _render_sweep_tab():
    result = st.session_state.get("sweep_result")
    if result is None:
        st.info("Configure and run a sweep from the sidebar (section 6).")
        return

    units = st.session_state.get("network_units") or {}
    st.subheader(f"Sweep: {_sweep_param_label(result.parameter, units)}")
    st.caption(
        f"Algorithms: {', '.join(result.algorithms)} | "
        f"runs/point: {result.n_runs} | seed: {result.experiment_seed}"
    )

    # Dynamic value column header: include units when relevant
    if result.parameter == "Pmin":
        value_header = f"Pmin ({units.get('PressureUnit', 'm')})"
    elif result.parameter == "Vmax":
        value_header = f"Vmax ({units.get('VelocityUnit', 'm/s')})"
    else:
        value_header = "Value"

    if getattr(result, "cancelled", False):
        st.warning(
            f"Sweep cancelled: {len(result.points)} of "
            f"{len(result.values)} points were reached. Points marked "
            "“Partial” used fewer runs than requested."
        )

    rows = []
    for p in result.points:
        rows.append({
            value_header:     f"{p.value:g}",
            "Status":         p.status,
            "Best (feasible)": (
                f"{p.best_score:,.0f}" if np.isfinite(p.best_score) else "—"
            ),
            "Mean cost": (
                f"{p.mean_score:,.0f}" if np.isfinite(p.mean_score) else "—"
            ),
            "Std":            f"{p.std_score:,.0f}",
            "Feasible %":     f"{p.feasible_pct:.0f}%",
            "Mean time (s)":  f"{p.mean_time:.1f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    # Trend chart (complete points only; partial ones are in the table)
    plot_pts = [p for p in result.points
                if p.status in ("Success", "AllInfeasible")]
    xs = [p.value for p in plot_pts]
    ys = [
        p.mean_score if np.isfinite(p.mean_score) else np.nan
        for p in plot_pts
    ]
    bests = [
        p.best_score if np.isfinite(p.best_score) else np.nan
        for p in plot_pts
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        name="Mean cost", line=dict(color="#1971c2", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=bests, mode="lines+markers",
        name="Best cost", line=dict(color="#d9480f", width=2, dash="dash"),
    ))
    fig.update_layout(
        height=380,
        xaxis_title=_sweep_param_label(result.parameter, units),
        yaxis_title="Cost",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    # CSV export
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ Download sweep (CSV)",
        data=csv,
        file_name=(
            f"wds_sweep_{result.parameter}_"
            f"{datetime.now():%Y%m%d_%H%M%S}.csv"
        ),
        mime="text/csv",
        width="stretch",
    )

def _render_compare_tab():
    st.subheader("Compare two experiments")
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader(
            "Experiment A (.json)", type=["json"], key="cmp_a"
        )
    with col2:
        file_b = st.file_uploader(
            "Experiment B (.json)", type=["json"], key="cmp_b"
        )

    if st.button(
        "🔄 Compare", type="primary",
        disabled=not (file_a and file_b),
        width="stretch",
    ):
        try:
            data_a = load_snapshot(file_a)
            data_b = load_snapshot(file_b)
            st.session_state.compare_result = {
                "a": restore_experiment(data_a),
                "b": restore_experiment(data_b),
                "stats_a": compute_comparison_stats(restore_experiment(data_a)),
                "stats_b": compute_comparison_stats(restore_experiment(data_b)),
            }
        except Exception as e:
            st.error(f"Compare failed: {type(e).__name__}: {e}")
            return

    result = st.session_state.get("compare_result")
    if result is None:
        st.info("Upload two .json experiments and click Compare.")
        return

    exp_a, exp_b = result["a"], result["b"]
    stats_a, stats_b = result["stats_a"], result["stats_b"]
    oa, ob = overall_stats(stats_a), overall_stats(stats_b)

    # ---- Overall comparison ----
    st.markdown("### Overall")
    st.dataframe(
        pd.DataFrame([
            {
                "Metric": "Best (feasible)",
                "A": f"{oa['best']:,.0f}" if np.isfinite(oa["best"]) else "—",
                "B": f"{ob['best']:,.0f}" if np.isfinite(ob["best"]) else "—",
                "Delta (B−A)": (
                    f"{ob['best'] - oa['best']:+,.0f}"
                    if np.isfinite(oa["best"]) and np.isfinite(ob["best"])
                    else "—"
                ),
            },
            {
                "Metric": "Mean (feasible)",
                "A": f"{oa['mean']:,.0f}" if np.isfinite(oa["mean"]) else "—",
                "B": f"{ob['mean']:,.0f}" if np.isfinite(ob["mean"]) else "—",
                "Delta (B−A)": (
                    f"{ob['mean'] - oa['mean']:+,.0f}"
                    if np.isfinite(oa["mean"]) and np.isfinite(ob["mean"])
                    else "—"
                ),
            },
            {
                "Metric": "Feasible runs",
                "A": str(oa["n_feasible"]),                       # ← str()
                "B": str(ob["n_feasible"]),                       # ← str()
                "Delta (B−A)": f"{ob['n_feasible'] - oa['n_feasible']:+d}",
            },
        ]),
        width="stretch", hide_index=True,
    )

    # ---- Per-algorithm ----
    st.markdown("### Per-algorithm")
    common = sorted(set(stats_a) & set(stats_b))
    if not common:
        st.warning("The two experiments do not share any algorithm.")
    else:
        rows = []
        for algo in common:
            sa, sb = stats_a[algo], stats_b[algo]
            for metric, key in [("Best", "best"), ("Mean", "mean"), ("Std", "std")]:
                a, b = sa[key], sb[key]
                rows.append({
                    "Algorithm": algo,
                    "Metric": metric,
                    "A": f"{a:,.0f}" if np.isfinite(a) else "—",
                    "B": f"{b:,.0f}" if np.isfinite(b) else "—",
                    "Delta (B−A)": (
                        f"{b - a:+,.0f}"
                        if np.isfinite(a) and np.isfinite(b) else "—"
                    ),
                })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ---- Convergence overlay ----
    st.markdown("### Convergence")
    fig = go.Figure()
    ca = {"GA": "#f59f00", "PSO": "#15aabf"}
    cb = {"GA": "#d9480f", "PSO": "#1971c2"}
    for label, exp, colors, dash in [
        ("A", exp_a, ca, "solid"),
        ("B", exp_b, cb, "dash"),
    ]:
        for algo, runs in exp.runs.items():
            convs = [
                r.convergence for r in runs
                if r.status == "Success" and r.convergence is not None
            ]
            if not convs:
                continue
            maxlen = max(len(c) for c in convs)
            padded = np.full((len(convs), maxlen), np.nan)
            for i, c in enumerate(convs):
                padded[i, :len(c)] = c
            fig.add_trace(go.Scatter(
                x=np.arange(1, maxlen + 1),
                y=np.nanmean(padded, axis=0),
                mode="lines",
                name=f"{label}/{algo}",
                line=dict(color=colors.get(algo, "gray"),
                          width=2, dash=dash),
            ))
    fig.update_layout(
        height=400, xaxis_title="Generation",
        yaxis_title="Best cost",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    # ---- Distribution ----
    st.markdown("### Distribution")
    fig = go.Figure()
    for label, exp, colors in [("A", exp_a, ca), ("B", exp_b, cb)]:
        for algo, runs in exp.runs.items():
            cs = [
                r.score for r in runs
                if r.status == "Success" and r.feasible
            ]
            if cs:
                fig.add_trace(go.Box(
                    y=cs, name=f"{label}/{algo}",
                    marker_color=colors.get(algo, "gray"),
                ))
    fig.update_layout(
        height=350, yaxis_title="Cost", showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch")

def _pick_best_run(result: ExperimentResult):
    """Select the best successful run (feasible first, then cost)."""
    best_record = None
    best_algo = None
    for algo, runs in result.runs.items():
        for r in runs:
            if r.status not in ("Success", "Partial"):
                continue
            if best_record is None:
                best_record, best_algo = r, algo
                continue
            if r.feasible and not best_record.feasible:
                best_record, best_algo = r, algo
            elif r.feasible == best_record.feasible and r.score < best_record.score:
                best_record, best_algo = r, algo
    return best_record, best_algo


def _position_to_indices(position_m: np.ndarray, problem) -> np.ndarray:
    """Convert a full-diameter vector (m) to option indices (0-based)."""
    D_opt = problem["D"]
    idx = np.zeros(len(problem["VariablePipes"]), dtype=int)
    for k, p in enumerate(problem["VariablePipes"]):
        idx[k] = int(np.argmin(np.abs(D_opt - position_m[p])))
    return idx


# =====================================================================
#  Results tabs
# =====================================================================

def _render_summary_tab():
    result: ExperimentResult | None = st.session_state.experiment
    if result is None:
        st.info("Run an experiment to see the summary here.")
        return

    if getattr(result, "cancelled", False):
        st.warning(
            "This experiment was cancelled — results are partial. "
            "Statistics use only runs that finished; a run stopped midway "
            "(status “Partial”) can still be the best design."
        )

    summary = summarize_experiment(result)
    rows = []
    for algo, s in summary.items():
        rows.append({
            "Algorithm": algo,
            "Runs": len(result.runs[algo]),
            "Success": s["n_success"],
            "Feasible": s["n_feasible"],
            "Mean cost": f"{s['mean']:,.0f}" if np.isfinite(s["mean"]) else "—",
            "Std": f"{s['std']:,.0f}" if np.isfinite(s["std"]) else "—",
            "Best (feasible)": (
                f"{s['best']:,.0f}" if np.isfinite(s["best"]) else "—"
            ),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ---- Convergence curves ----
    st.subheader("Convergence")
    fig = go.Figure()
    colors = {"GA": "#d9480f", "PSO": "#1971c2"}
    for algo, runs in result.runs.items():
        conv_list = [
            r.convergence for r in runs
            if r.status == "Success" and r.convergence is not None
        ]
        if not conv_list:
            continue
        max_len = max(len(c) for c in conv_list)
        padded = np.full((len(conv_list), max_len), np.nan)
        for i, c in enumerate(conv_list):
            padded[i, : len(c)] = c
        mean = np.nanmean(padded, axis=0)
        lo = np.nanmin(padded, axis=0)
        hi = np.nanmax(padded, axis=0)
        x = np.arange(1, max_len + 1)
        fig.add_trace(go.Scatter(
            x=x, y=mean, mode="lines",
            line=dict(color=colors.get(algo, "gray"), width=2),
            name=f"{algo} (mean)",
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([lo, hi[::-1]]),
            fill="toself",
            fillcolor=colors.get(algo, "gray").replace(")", ", 0.15)")
                     if "," in colors.get(algo, "") else "rgba(150,150,150,0.15)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        height=380, xaxis_title="Generation / Iteration",
        yaxis_title="Best cost", margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    # ---- Cost distribution (boxplot) ----
    st.subheader("Final cost distribution")
    rows = []
    for algo, runs in result.runs.items():
        for r in runs:
            if r.status == "Success" and r.feasible:
                rows.append({"Algorithm": algo, "Cost": r.score})
    if not rows:
        st.info("No feasible runs to plot.")
    else:
        df = pd.DataFrame(rows)
        fig = px.box(df, x="Algorithm", y="Cost", color="Algorithm",
                     color_discrete_map=colors, points="all")
        fig.update_layout(height=350, showlegend=False,
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, width="stretch")


def _render_best_design_tab():
    if st.session_state.experiment is None:
        st.info("Run an experiment first.")
        return

    snap = st.session_state.problem_snapshot
    pipe_names = snap["pipe_names"]
    pos = st.session_state.best_position_m
    vel = st.session_state.pipe_velocities
    units = snap["Units"]
    fixed = set(int(x) for x in snap["FixedPipes"])
    v_unit = units["VelocityUnit"]
    v_scale = 1.0 if v_unit == "m/s" else 0.3048   # m/s -> display
    L_unit = units["LengthUnit"]
    L_scale = 1.0 if L_unit == "m" else 1.0 / 0.3048

    # Build table
    rows = []
    for i, name in enumerate(pipe_names):
        L = snap["L"][i]
        d_m = pos[i] if pos is not None else snap["initial_D"][i]
        d_mm = d_m * 1000
        d_in = d_m / 0.0254
        v_si = vel[i] if vel is not None and i < len(vel) else float("nan")
        v_disp = v_si / v_scale if v_scale else v_si
        rows.append({
            "Pipe ID": i + 1,
            "Name": name,
            f"Length ({L_unit})": f"{L / L_scale:.0f}",
            "Diameter (mm)": f"{d_mm:.1f}",
            "Diameter (in)": f"{d_in:.2f}",
            f"Velocity ({v_unit})": f"{v_disp:.3f}",
            "Status": "FIXED" if i in fixed else "variable",
        })
    df = pd.DataFrame(rows)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader("Pipe results")
        st.dataframe(df, width="stretch", hide_index=True)
    with c2:
        score = st.session_state.best_score
        feas = st.session_state.best_feasible
        st.metric("Best cost", f"{score:,.0f}")
        if feas:
            st.success("Feasible design ✓")
        else:
            st.error("Infeasible design")


def _render_hydraulics_tab():
    if st.session_state.experiment is None:
        st.info("Run an experiment first.")
        return

    snap = st.session_state.problem_snapshot
    Pj_si = st.session_state.node_pressures       # meters
    Vp_si = st.session_state.pipe_velocities      # m/s
    pmin_si = snap["Pmin"]
    vmax_si = snap["Vmax"]
    junc_names = snap["junction_names"]
    units = snap["Units"]

    if Pj_si is None or Vp_si is None:
        st.warning("Hydraulic results not available.")
        return

    # Convert to display units
    p_unit = units["PressureUnit"]
    v_unit = units["VelocityUnit"]
    Pj = np.array([pressure_from_si(p, units) for p in Pj_si])
    Vp = np.array([velocity_from_si(v, units) for v in Vp_si])
    pmin = pressure_from_si(pmin_si, units)
    vmax = velocity_from_si(vmax_si, units)

    # ---- Node pressures ----
    st.subheader("Node pressures")
    rows = []
    for k, name in enumerate(junc_names):
        # compare in SI to avoid floating-point drift
        violation = Pj_si[k] < pmin_si
        rows.append({
            "Node ID": k + 1,
            "Name": name,
            f"Pressure ({p_unit})": f"{Pj[k]:.3f}",
            "Status": f"Violation (<{pmin:.2f} {p_unit})"
                      if violation else "OK",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=junc_names, y=Pj,
        marker_color=["#e74c3c" if Pj_si[i] < pmin_si else "#2ecc71"
                      for i in range(len(Pj))],
        text=[f"{p:.2f}" for p in Pj], textposition="outside",
    ))
    fig.add_hline(y=pmin, line_dash="dash", line_color="red",
                  annotation_text=f"Pmin = {pmin:.2f} {p_unit}")
    fig.update_layout(height=320, yaxis_title=f"Pressure ({p_unit})",
                      margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, width="stretch")

        # ---- Pipe velocities ----
    st.subheader("Pipe velocities")
    pipe_names = snap["pipe_names"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pipe_names, y=Vp,
        marker_color=["#e74c3c" if Vp_si[i] > vmax_si else "#3498db"
                      for i in range(len(Vp))],
        text=[f"{v:.2f}" for v in Vp], textposition="outside",
    ))
    fig.add_hline(y=vmax, line_dash="dash", line_color="red",
                  annotation_text=f"Vmax = {vmax:.2f} {v_unit}")
    fig.update_layout(
        height=320,
        yaxis_title=f"Velocity ({v_unit})",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    # ---- Status line ----
    conv_txt = (
        "✓ Converged" if st.session_state.hydraulics_converged
        else "⚠ Not converged"
    )
    st.caption(
        f"Min pressure: {Pj.min():.2f} {p_unit} | "
        f"Max velocity: {Vp.max():.2f} {v_unit} | {conv_txt}"
    )


def _render_map_tab():
    if st.session_state.inp_path is None:
        st.info("Load an .inp file first.")
        return

    mode = st.selectbox(
        "Display mode",
        ["preview", "velocity", "pressure", "diameter"],
        key="map_mode",
        format_func=lambda x: {
            "preview": "Preview (network only)",
            "velocity": "Pipe velocity",
            "pressure": "Junction pressure",
            "diameter": "Pipe diameter",
        }[x],
    )
    show_labels = st.checkbox("Show node labels", value=False)

    wn = _load_network(st.session_state.inp_path)

    # Compute coloring data (in display units)
    pressures = None
    velocities = None
    diameters = None
    pmin = vmax = None
    fixed_1based = []
    units = st.session_state.get("network_units") or {}

    if st.session_state.experiment is not None:
        snap = st.session_state.problem_snapshot
        Pj_si = st.session_state.node_pressures
        Vp_si = st.session_state.pipe_velocities
        pos = st.session_state.best_position_m

        pressures = {
            n: pressure_from_si(p, units)
            for n, p in zip(snap["junction_names"], Pj_si)
        }
        velocities = {
            n: velocity_from_si(v, units)
            for n, v in zip(snap["pipe_names"], Vp_si)
        }
        diameters = {
            n: diameter_to_display(pos[i], units)
            for i, n in enumerate(snap["pipe_names"])
        }
        pmin = pressure_from_si(snap["Pmin"], units)
        vmax = velocity_from_si(snap["Vmax"], units)
        fixed_1based = [int(x) + 1 for x in snap["FixedPipes"]]

    fig = build_network_figure(
        wn, mode=mode,
        pressures=pressures, velocities=velocities, diameters=diameters,
        pmin=pmin, vmax=vmax,
        fixed_pipe_indices=fixed_1based,
        show_labels=show_labels,
        units=units,                    # ← جدید
    )
    st.plotly_chart(fig, width="stretch")

    if st.session_state.experiment is None:
        st.caption(
            "Preview mode: network topology only. "
            "Run an experiment to see velocity/pressure/diameter coloring."
        )


def _render_all_runs_tab():
    if st.session_state.experiment is None:
        st.info("Run an experiment first.")
        return

    result = st.session_state.experiment
    rows = []
    for algo, runs in result.runs.items():
        for r in runs:
            rows.append({
                "Algorithm": algo,
                "Run": r.run_index,
                "Seed": r.seed,
                "Status": r.status,
                "Restarts": len(getattr(r, "restart_gens", [])),
                "Epoch bests": " | ".join(
                    f"{c:,.0f}" if np.isfinite(c) else "–"
                    for c in getattr(r, "epoch_bests", [])
                ),
                "Cost": (
                    f"{r.score:,.0f}" if np.isfinite(r.score) else "—"
                ),
                "Feasible": "✓" if r.feasible else "✗",
                "Time (s)": (
                    f"{r.execution_time:.1f}"
                    if np.isfinite(r.execution_time) else "—"
                ),
                "Error": r.error_message or "",
            })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    # ---- Export buttons ----
    st.subheader("Export")
    c1, c2 = st.columns(2)
    with c1:
        # ASCII-safe + BOM for Excel compatibility
        df_csv = df.copy()
        df_csv["Feasible"] = df_csv["Feasible"].map({"✓": "Yes", "✗": "No"})
        csv = df_csv.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇ Download runs (CSV)",
            data=csv,
            file_name=f"wds_runs_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
            width="stretch",
        )
    with c2:
        xlsx = _build_excel_export(result)
        if xlsx is not None:
            st.download_button(
                "⬇ Download experiment (Excel)",
                data=xlsx,
                file_name=f"wds_experiment_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                width="stretch",
            )


def _build_excel_export(result: ExperimentResult) -> bytes:
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            # Summary sheet
            summary = summarize_experiment(result)
            s_rows = [{"Algorithm": a, **s} for a, s in summary.items()]
            pd.DataFrame(s_rows).to_excel(
                writer, sheet_name="Summary", index=False
            )
            # Per-algorithm runs
            for algo, runs in result.runs.items():
                rows = [{
                    "Run": r.run_index, "Seed": r.seed, "Status": r.status,
                    "Cost": r.score, "Feasible": r.feasible,
                    "Time (s)": r.execution_time,
                    "Error": r.error_message,
                } for r in runs]
                pd.DataFrame(rows).to_excel(
                    writer, sheet_name=f"{algo}_runs", index=False
                )
        return buf.getvalue()
    except Exception as e:
        st.warning(f"Excel export failed: {e}")
        return None


# =====================================================================
#  Main
# =====================================================================

def main():
    init_state()
    _header()

    actions = _render_sidebar()

    if actions["run_clicked"]:
        try:
            _start_run(actions["algo_label"])
        except Exception as e:
            st.error(f"Could not start experiment: {type(e).__name__}: {e}")
            st.exception(e)

    if actions["sweep"] is not None:
        try:
            _start_sweep(**actions["sweep"])
        except Exception as e:
            st.error(f"Could not start sweep: {type(e).__name__}: {e}")
            st.exception(e)

    _show_job_notice()
    if st.session_state.get("job") is not None:
        _job_panel()

    tabs = st.tabs([
        "📊 Summary",
        "🏗 Best Design",
        "💧 Hydraulics",
        "🗺 Network Map",
        "📋 All Runs",
        "🔍 Sweep",
        "🔄 Compare",
    ])
    with tabs[0]:
        _render_summary_tab()
    with tabs[1]:
        _render_best_design_tab()
    with tabs[2]:
        _render_hydraulics_tab()
    with tabs[3]:
        _render_map_tab()
    with tabs[4]:
        _render_all_runs_tab()
    with tabs[5]:
        _render_sweep_tab()
    with tabs[6]:
        _render_compare_tab()


if __name__ == "__main__":
    main()