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


import copy
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
from wds_optimizer.algorithms.registry import (
    ALGORITHM_REGISTRY, list_algorithms,
)
from wds_optimizer.optimization.policies import (
    CONSTRAINT_POLICIES, POLICY_META,
)
from wds_optimizer.api import _parse_network, build_problem
from wds_optimizer.experiments.runner import (
    ExperimentResult,
    run_experiment,
    summarize_experiment,
)
from wds_optimizer.hydraulics.simulation import run_hydraulic_simulation
from wds_optimizer.hydraulics.units import (
    detect_network_units,
    pressure_from_si, pressure_to_si,
    velocity_from_si, velocity_to_si,
    diameter_to_display,
)
from wds_optimizer.app.network_map import build_network_figure
from wds_optimizer.app.state import init_state
from wds_optimizer.app.uploads import read_uploaded
from wds_optimizer.app.jobs import BackgroundJob
from wds_optimizer.analysis import stats as st_
from wds_optimizer.persistence.snapshot import (
    build_snapshot, load_snapshot, restore_experiment,
)
from wds_optimizer.sensitivity.batch import (
    PlanRow, run_batch_sweep, validate_plan, _annotations_for_param,
)
from wds_optimizer.sensitivity.config import (
    build_sensitivity_parameters, normalize_param_name,
    parse_sensitivity_values,
)
from wds_optimizer.reporting.sensitivity import (
    batch_report_markdown, batch_report_csv,
)
from wds_optimizer.reporting.sensitivity import _fmt_cost
from wds_optimizer.sensitivity.runner import run_single_sweep
from wds_optimizer.app.compare import (
    compute_comparison_stats, overall_stats,
)
from wds_optimizer.reporting.report import build_report_zip
from wds_optimizer.history import (
    HistoryEntry, auto_name, new_entry,
    history_to_dict, load_history, to_dict as history_entry_to_dict,
)
from wds_optimizer.plugins import (
    discover_plugins, get_plugin_short_ids, last_report,
    load_plugin_from_file, plugins_dir, reset_plugins,
)

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
def _load_network_cached(inp_path: str):
    """Parse and cache the WNTR network model by path.

    Delegates the parse to ``api._parse_network`` so there is one place
    that knows how to read an .inp.  The cache itself stays here because
    ``@st.cache_resource`` is Streamlit-specific — the API must not
    depend on Streamlit.

    Callers should use ``_load_network`` (below), never this function
    directly, so that each experiment gets its own mutable copy.
    """
    return _parse_network(inp_path)


def _load_network(inp_path: str):
    """Return a fresh, isolated copy of the network model.

    ``run_hydraulic_simulation`` mutates pipe diameters on the model it
    is given, and some WNTR versions also touch pattern / control /
    units state during ``EpanetSimulator.run_sim``.  Returning the
    cached object directly would let those mutations leak into the
    *next* experiment (observed bug: Hanoi → Duhala → Hanoi fails, the
    second Hanoi run starts from the first run's final diameters).

    A deep copy of a few-hundred-pipe model takes ~1 ms — negligible
    next to a single hydraulic solve.
    """
    return copy.deepcopy(_load_network_cached(inp_path))


@st.cache_data(show_spinner=False)
def _get_network_units(inp_path: str):
    """Detect (and cache) the unit system of an .inp file."""
    import wntr
    wn = wntr.network.WaterNetworkModel(inp_path)
    return detect_network_units(wn)

# =====================================================================
#  UI helpers
# =====================================================================

# Algo colors: keyed by short_id.  A plugin algorithm not listed here
# gets a fallback color (gray) at render time.  We keep the mapping
# for the four built-ins only; the visual identity of built-in
# algorithms doesn't need to be derived from META.
ALGO_COLORS = {"GA": "#d9480f", "PSO": "#1971c2", "DE": "#7048e8",
               "SADE": "#0b7285"}
ALGO_COLORS_A = {"GA": "#f59f00", "PSO": "#15aabf", "DE": "#b197fc",
                 "SADE": "#66d9e8"}


def _meta_options(short_id: str, param_key: str) -> tuple:
    """Look up a param's ``options`` list from ALGORITHM_META."""
    meta = ALGORITHM_REGISTRY.get(short_id, {}).get("meta", {})
    for p in meta.get("params", []):
        if p.get("key") == param_key:
            return tuple(p.get("options", ()))
    return ()


SADE_TOLS = _meta_options("SADE", "sade_tol") or (
    1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
)


def _nearest_tol(v):
    """Snap a stored tolerance to the nearest selectable option."""
    return min(SADE_TOLS, key=lambda t: abs(np.log10(t) - np.log10(max(v, 1e-12))))
DE_STRATEGIES = ("rand/1/bin", "best/1/bin", "current-to-best/1/bin")

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
        "Water distribution network optimization using GA, PSO, DE and SADE, "
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
        "max_gen_input":     int(st.session_state.get("max_gen", 500)),
        "pmin_input":        float(st.session_state.get("pmin", 30.0)),
        "pmax_input":        float(st.session_state.get("pmax", 100.0)),
        "vmin_input":        float(st.session_state.get("vmin", 0.0)),
        "vmax_input":        float(st.session_state.get("vmax", 2.5)),
        "use_pmin_input":    bool(st.session_state.get("use_pmin", True)),
        "use_pmax_input":    bool(st.session_state.get("use_pmax", False)),
        "use_vmin_input":    bool(st.session_state.get("use_vmin", False)),
        "use_vmax_input":    bool(st.session_state.get("use_vmax", True)),
        "fixed_pipes_input": st.session_state.get("fixed_pipes", ""),
        "diam_unit_input":   st.session_state.get("diam_unit", "in"),
        "ga_pc_input":       float(st.session_state.get("ga_pc", 0.8)),
        "ga_pm_input":       float(st.session_state.get("ga_pm", 0.03)),
        "ga_sel_input":      st.session_state.get("ga_sel", "tournament"),
        "pso_w_input":       float(st.session_state.get("pso_w", 0.7)),
        "pso_c1_input":      float(st.session_state.get("pso_c1", 1.5)),
        "pso_c2_input":      float(st.session_state.get("pso_c2", 1.5)),
        "n_runs_input":      int(st.session_state.get("n_runs", 3)),
        "exp_seed_input":    int(st.session_state.get("exp_seed", 1)),
        "algo_input":        ["GA"],
        "constraint_policy_input": "deb_prefers",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Seed algorithm-param widget keys from META (covers all registered
    # algorithms — including future plugins — without GUI changes).
    for entry in ALGORITHM_REGISTRY.values():
        meta = entry["meta"]
        for p in meta["params"]:
            widget_key = p["key"] + "_input"
            if widget_key in st.session_state:
                continue
            default = p["default"]
            # Snap sade_tol to nearest allowed option (its widget is a
            # select_slider; only the listed options are valid).
            if p.get("widget") == "select_slider" and p.get("options"):
                default = _nearest_tol(float(default))
            st.session_state[widget_key] = default

def _render_sidebar():
    _init_widget_state()
    # `busy` (is a job running?) is used by several sections —
    # Sensitivity Analysis (4), Experiment (5), Save/Load, etc.
    # Compute it once here, before any widget, so all sections see
    # the same value.
    busy = st.session_state.get("job") is not None
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
            "D.txt unit", ["in", "mm"], key="diam_unit_input"
        )

        if inp_upload and d_upload and cost_upload:
            inp_path = read_uploaded(inp_upload.getvalue(), inp_upload.name)
            d_path = read_uploaded(d_upload.getvalue(), d_upload.name)
            cost_path = read_uploaded(
                cost_upload.getvalue(), cost_upload.name
            )

            # If these are different files than the ones currently
            # loaded, clear any existing experiment so the results
            # tabs don't show stale data from the old network.  The
            # content-addressed upload returns the same path if the
            # bytes are identical, so re-uploading the same files
            # does not clear.
            new_paths = (inp_path, d_path, cost_path)
            old_paths = (
                st.session_state.get("inp_path"),
                st.session_state.get("d_path"),
                st.session_state.get("cost_path"),
            )
            if (st.session_state.get("experiment") is not None
                    and new_paths != old_paths):
                st.session_state.experiment = None
                st.session_state.problem_snapshot = None
                st.session_state.best_algo = None
                st.session_state.best_run_index = None
                st.session_state.best_position_m = None
                st.session_state.best_score = None
                st.session_state.best_feasible = False
                st.session_state.node_pressures = None
                st.session_state.pipe_velocities = None
                st.session_state.hydraulics_converged = False
                _clear_run_ui_state()
                st.info(
                    "Loaded a new network — previous experiment "
                    "results cleared."
                )

            st.session_state.inp_path = inp_path
            st.session_state.d_path = d_path
            st.session_state.cost_path = cost_path
            st.session_state.diam_unit = diam_unit
            st.success("All files loaded.")
        elif st.session_state.inp_path:
            st.info("Using previously loaded files.")

        st.divider()
        st.header("2. Constraints & Population")

        units = None
        if st.session_state.inp_path:
            try:
                units = _get_network_units(st.session_state.inp_path)
            except Exception:
                units = None
        st.session_state.network_units = units

        # Auto-convert all four constraint values when units change.
        prev = st.session_state.get("network_units_prev")
        if units is not None and prev is not None:
            if prev.get("System") != units.get("System"):
                st.session_state.pmin = pressure_from_si(
                    pressure_to_si(st.session_state.pmin, prev), units
                )
                st.session_state.pmax = pressure_from_si(
                    pressure_to_si(st.session_state.pmax, prev), units
                )
                st.session_state.vmin = velocity_from_si(
                    velocity_to_si(st.session_state.vmin, prev), units
                )
                st.session_state.vmax = velocity_from_si(
                    velocity_to_si(st.session_state.vmax, prev), units
                )
                st.session_state.pmin_input = float(st.session_state.pmin)
                st.session_state.pmax_input = float(st.session_state.pmax)
                st.session_state.vmin_input = float(st.session_state.vmin)
                st.session_state.vmax_input = float(st.session_state.vmax)
                st.info(
                    f"Pressure and velocity values converted to "
                    f"{units['System']} units "
                    f"({units['PressureUnit']}, {units['VelocityUnit']})."
                )
        st.session_state.network_units_prev = units

        p_unit_ui = units["PressureUnit"] if units else "m"
        v_unit_ui = units["VelocityUnit"] if units else "m/s"
        if units and units["System"] == "US":
            st.caption(
                f"Network uses US customary units: "
                f"pressure in {p_unit_ui}, velocity in {v_unit_ui}."
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

        # ---- Constraint selector ----
        st.subheader("Constraints")
        st.caption(
            "Enable the constraints that apply to this network. "
            "Disabled constraints are ignored by the optimiser and "
            "by the hydraulic plots — no need to set Vmax to a huge "
            "value to switch it off."
        )

        def _constraint_row(name, unit, use_key, val_key, min_v, step,
                            help_text):
            c1, c2 = st.columns([1, 2])
            with c1:
                enabled = st.checkbox(name, key=use_key,
                                      help=help_text)
            with c2:
                st.number_input(
                    f"{name} ({unit})",
                    min_value=float(min_v), step=float(step),
                    key=val_key,
                    disabled=not enabled,
                    label_visibility="collapsed",
                )
            return enabled

        st.markdown("**Pressure**")
        _constraint_row("Pmin", p_unit_ui, "use_pmin_input",
                        "pmin_input", 0.0, 1.0,
                        f"Minimum pressure at every junction ({p_unit_ui}).")
        _constraint_row("Pmax", p_unit_ui, "use_pmax_input",
                        "pmax_input", 0.0, 1.0,
                        f"Maximum pressure at every junction ({p_unit_ui}).")

        st.markdown("**Velocity**")
        _constraint_row("Vmin", v_unit_ui, "use_vmin_input",
                        "vmin_input", 0.0, 0.1,
                        f"Minimum velocity in every pipe ({v_unit_ui}).")
        _constraint_row("Vmax", v_unit_ui, "use_vmax_input",
                        "vmax_input", 0.1, 0.1,
                        f"Maximum velocity in every pipe ({v_unit_ui}).")

        # ---- Inline validation ----
        _any_on = any(st.session_state.get(f"use_{k}_input")
                      for k in ("pmin", "pmax", "vmin", "vmax"))
        if not _any_on:
            st.error("Enable at least one constraint before running.")
        _p_both = (st.session_state.get("use_pmin_input")
                   and st.session_state.get("use_pmax_input"))
        if _p_both and st.session_state.pmin_input >= st.session_state.pmax_input:
            st.error(
                f"Pmin ({st.session_state.pmin_input}) must be strictly "
                f"smaller than Pmax ({st.session_state.pmax_input})."
            )
        _v_both = (st.session_state.get("use_vmin_input")
                   and st.session_state.get("use_vmax_input"))
        if _v_both and st.session_state.vmin_input >= st.session_state.vmax_input:
            st.error(
                f"Vmin ({st.session_state.vmin_input}) must be strictly "
                f"smaller than Vmax ({st.session_state.vmax_input})."
            )

        st.text_input(
            "Fixed pipe IDs (e.g. '1, 3')",
            key="fixed_pipes_input",
        )

        st.divider()
        st.header("3. Algorithm")

        # Constraint-handling policy (applies to every selected algorithm)
        st.subheader("Constraint handling")
        _policy_names = list(CONSTRAINT_POLICIES)
        # Ensure session_state holds a valid value BEFORE the widget is
        # created.  Modifying a widget's key *before* creation is the
        # supported pattern; modifying it after (or passing both
        # `index=` and `key=`) triggers Streamlit's warning about a
        # widget created with a default and also set via the Session
        # State API.
        if "constraint_policy_input" not in st.session_state:
            st.session_state.constraint_policy_input = "deb_prefers"
        if st.session_state.constraint_policy_input not in _policy_names:
            # Defensive: a policy was removed from CONSTRAINT_POLICIES
            # between sessions (very unlikely; the set is closed).
            st.session_state.constraint_policy_input = (
                _policy_names[0] if _policy_names else "deb_prefers"
            )
        _selected_policy = st.selectbox(
            "How should the optimiser compare infeasible designs?",
            options=_policy_names,
            format_func=lambda n: POLICY_META[n]["display_name"],
            key="constraint_policy_input",
            help="This setting applies globally to every algorithm "
                 "in this experiment.",
        )
        _policy_meta = POLICY_META[_selected_policy]
        st.caption(_policy_meta["description"])
        with st.expander("View code and reference", expanded=False):
            st.code(_policy_meta["code"], language="python")
            st.caption(f"Reference: {_policy_meta['reference']}")
            if _policy_meta.get("notes"):
                st.caption(f"Notes: {_policy_meta['notes']}")

        # Then the existing algorithm multiselect
        algo = st.multiselect(
            "Algorithms to run",
            options=list_algorithms(),
            format_func=_algo_display_label,
            key="algo_input",
            help="Every selected algorithm runs with the same number "
                 "of runs and seeds.  [P] marks user-installed plugins.",
        )
        if not algo:
            st.caption("Select at least one algorithm.")

        _plugin_ids_now = get_plugin_short_ids()
        for short_id in algo:
            entry = ALGORITHM_REGISTRY.get(short_id)
            if entry is None:
                continue
            meta = entry["meta"]
            is_plugin = short_id in _plugin_ids_now
            label = f"{meta['display_name']} parameters"
            if is_plugin:
                label = f"🔌 {label}"
            with st.expander(label, expanded=True):
                st.caption(meta["description"])
                _render_params_from_meta(meta)
                if is_plugin:
                    with st.expander("View source code",
                                     expanded=False):
                        _render_plugin_source(entry["run"])

        # Plugins management (below algorithm selection + params).
        _loaded_ids = get_plugin_short_ids()
        _plugins_header = (
            f"🔌 Plugins ({len(_loaded_ids)} loaded)"
            if _loaded_ids else "🔌 Plugins"
        )
        with st.expander(_plugins_header, expanded=False):
            _render_plugins_panel()

        # ---- 4. Sensitivity Analysis ----
        st.divider()
        st.header("4. Sensitivity Analysis")
        st.caption(
            "Analyze how parameter values affect the result. "
            "Opens a dialog to define a batch of parameter sweeps "
            "for the current network and algorithms."
        )
        if st.button(
            "📊 Open sensitivity analysis…",
            width="stretch",
            disabled=busy,
        ):
            st.session_state.sensitivity_dialog_open = True
            st.rerun()  

        # ---- 5. Experiment ----
        st.divider()
        st.header("5. Experiment")

        # Clear run name after a successful run (sentinel).
        if st.session_state.pop("_clear_run_name", False):
            st.session_state.run_name_input = ""

        st.text_input(
            "Run name (optional)",
            key="run_name_input",
            placeholder="Leave empty for auto-generated name",
            help="A friendly label for this run in History.  "
                 "Auto-generated if left empty.  Rename later "
                 "with ✏️ in the History list.",
        )

        st.number_input("Runs per algorithm", min_value=1, max_value=50,
                        step=1, key="n_runs_input")
        st.number_input("Experiment seed", min_value=0,
                        max_value=2**31 - 1, step=1, key="exp_seed_input")

        st.divider()
        can_run = bool(st.session_state.inp_path)
        # busy is computed at the top of _render_sidebar (v0.8.2).
        run_clicked = _job_button(
            "run", "🚀 Run Experiment", "⏹ Cancel experiment",
            enabled=can_run and bool(st.session_state.get("algo_input")),
        )
        if not can_run:
            st.caption("Load all three input files to enable.")
        # (The old "Clear results" button has been removed: History
        # records every run, and loading new input files above clears
        # stale results automatically.  No user action needed.)

        # ---- 6. Snapshot ----
        st.divider()
        st.header("6. Snapshot (single experiment)")
        st.caption(
            "Save or restore one experiment as a JSON.  For a whole "
            "session's runs, use History (section 7)."
        )

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
                    config=_config_for_snapshot(),
                    inp_content=inp_content,
                    d_content=d_content,
                    cost_content=cost_content,
                )
                # UI state is not part of the experiment itself; record
                # it under an optional "ui" key so older saves (which
                # lack it) remain loadable and newer loads can restore
                # the user's last view.
                ui_state: dict = {}
                if st.session_state.get("selected_run"):
                    ui_state["selected_run"] = list(
                        st.session_state.selected_run)
                if "only_feasible" in st.session_state:
                    ui_state["only_feasible"] = bool(
                        st.session_state.only_feasible)
                if ui_state:
                    snap["ui"] = ui_state
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
        
        # ---- 7. History ----
        st.divider()
        st.header("7. History (all runs)")
        _render_history_section()

        # ---- 8. Reproducibility Report ----
        st.divider()
        st.header("8. Reproducibility Report")
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
                            config=_config_for_snapshot(),
                            reference_cost=(
                                float(st.session_state.get(
                                    "ref_cost_input") or 0.0) or None),
                            success_tol_pct=float(st.session_state.get(
                                "ref_tol_input", 1.0)),
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
        "algo_label": algo,
    }

def _format_sweep_value(v) -> str:
    """Format a sweep value for display.

    Numbers use ``:g``; strings (choice params like constraint_policy)
    are shown verbatim.
    """
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return str(v)

def _algo_display_label(sid: str) -> str:
    """Format an algorithm short_id for the GUI dropdown.

    Plugins are prefixed with ``[P]`` so users can distinguish them
    from built-in algorithms at a glance.  Display uses the
    algorithm's display_name (from META) when available.
    """
    entry = ALGORITHM_REGISTRY.get(sid)
    if entry is None:
        return sid
    display = entry["meta"].get("display_name", sid)
    if sid in get_plugin_short_ids():
        return f"[P] {display}"
    return display

def _discover_plugins_if_needed() -> None:
    """Run plugin discovery once per session.

    Subsequent reruns skip it (the ``_plugins_discovered`` flag is
    set after the first call).  The "🔄 Reload" button in the
    Plugins panel resets the flag to force a re-scan.
    """
    if st.session_state.get("_plugins_discovered"):
        return
    try:
        report = discover_plugins(verbose=False)
        st.session_state._plugins_report = report
        st.session_state._plugins_error = None
    except Exception as e:
        st.session_state._plugins_report = None
        st.session_state._plugins_error = (
            f"{type(e).__name__}: {e}"
        )
    st.session_state._plugins_discovered = True

def _render_plugin_source(run_fn) -> None:
    """Show the source code of a plugin's run() function, read-only."""
    try:
        import inspect
        source = inspect.getsource(run_fn)
        st.code(source, language="python")
    except (OSError, TypeError) as e:
        st.caption(f"Source not available: {e}")

def _render_plugins_panel() -> None:
    """Manage loaded plugins: reload, install from file, view errors."""
    from pathlib import Path

    report = (st.session_state.get("_plugins_report")
              or last_report())
    pdir = plugins_dir()
    plugin_ids = get_plugin_short_ids()

    # ---- First-time security warning ----
    if not st.session_state.get("_plugins_security_ack"):
        st.warning(
            "⚠ Plugins are Python code with full system access. "
            "Only install plugins from sources you trust. "
            "There is no sandbox in v0.5."
        )
        ack = st.checkbox(
            "I understand and accept the risk.",
            key="plugins_security_ack_widget",
        )
        if ack:
            st.session_state._plugins_security_ack = True
            st.rerun()
        return

    # ---- Directory + actions ----
    st.caption(f"Directory: `{pdir}`")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Reload", width="stretch",
                     key="plugins_reload",
                     help="Re-scan the plugins directory."):
            try:
                reset_plugins()
                new_report = discover_plugins(verbose=False)
                st.session_state._plugins_report = new_report
                st.session_state._plugins_error = None
                st.rerun()
            except Exception as e:
                st.error(f"Reload failed: {type(e).__name__}: {e}")
    with c2:
        if st.button("📂 Open folder", width="stretch",
                     key="plugins_open_folder"):
            try:
                pdir.mkdir(parents=True, exist_ok=True)
                import subprocess
                import sys as _sys
                if _sys.platform == "win32":
                    subprocess.Popen(["explorer", str(pdir)])
                elif _sys.platform == "darwin":
                    subprocess.Popen(["open", str(pdir)])
                else:
                    subprocess.Popen(["xdg-open", str(pdir)])
            except Exception as e:
                st.caption(f"Could not open folder: {e}")

    # ---- Install from file ----
    uploaded = st.file_uploader(
        "Load plugin from file",
        type=["py"], key="plugins_upload",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        if st.button(
            "Install uploaded plugin",
            key="plugins_install_upload", width="stretch",
        ):
            try:
                pdir.mkdir(parents=True, exist_ok=True)
                dst = pdir / uploaded.name
                dst.write_bytes(uploaded.getvalue())
                result = load_plugin_from_file(dst, verbose=False)
                if result.status == "loaded":
                    st.success(f"Loaded {result.short_id}")
                    # Refresh the cached report
                    st.session_state._plugins_report = last_report()
                    st.rerun()
                else:
                    st.error(
                        f"{result.status.capitalize()}: {result.error}"
                    )
            except Exception as e:
                st.error(f"Install failed: {type(e).__name__}: {e}")

    # ---- Discovery error (if any) ----
    if st.session_state.get("_plugins_error"):
        st.error(f"Discovery error: {st.session_state._plugins_error}")

    # ---- Loaded plugins ----
    if plugin_ids:
        st.markdown(f"**Loaded ({len(plugin_ids)})**")
        for sid in sorted(plugin_ids):
            meta = ALGORITHM_REGISTRY.get(sid, {}).get("meta", {})
            display = meta.get("display_name", "?")
            st.markdown(f"- `{sid}` — {display}")
    else:
        st.caption("No plugins loaded.")

    # ---- Errors from last discovery ----
    if report.failed:
        st.markdown(f"**Failed ({len(report.failed)})**")
        for path, err in report.failed:
            with st.expander(
                f"❌ {Path(path).name}", expanded=False,
            ):
                st.caption(f"`{path}`")
                st.code(err)
    if report.skipped:
        st.markdown(f"**Skipped ({len(report.skipped)})**")
        for path, reason in report.skipped:
            st.caption(f"`{Path(path).name}` — {reason}")

def _algorithm_params_from_sidebar(algorithms: list[str]) -> dict:
    """Read each algorithm's param values from widget keys.

    Returns ``{short_id: {param_key: value}}``.
    """
    from wds_optimizer.algorithms.registry import ALGORITHM_REGISTRY
    out: dict = {}
    for sid in algorithms:
        entry = ALGORITHM_REGISTRY.get(sid)
        if entry is None:
            continue
        params = {}
        for p in entry["meta"]["params"]:
            wk = p["key"] + "_input"
            params[p["key"]] = st.session_state.get(wk, p["default"])
        out[sid] = params
    return out

def _make_default_plan(algos: list[str]) -> list[dict]:
    """Default sensitivity plan for the given algorithms.

    Each entry is {id, param, values, enabled}.  Keys are chosen
    from the most impactful parameters per algorithm, with sensible
    starting ranges from docs/parameters.md.
    """
    import uuid as _uuid

    defaults = {
        "GA.ga_pc": "0.6, 0.7, 0.8, 0.9",
        "GA.ga_pm": "0.01, 0.03, 0.05, 0.1",
        "GA.ga_sel": "tournament, roulette",
        "PSO.pso_w": "0.5, 0.7, 0.9",
        "PSO.pso_c1": "1.0, 1.5, 2.0",
        "DE.de_f": "0.3, 0.5, 0.7",
        "DE.de_cr": "0.3, 0.5, 0.9",
        "DE.de_strategy": "rand/1/bin, best/1/bin",
        "SADE.sade_stag": "100, 300, 500",
        "SADE.sade_tol": "1e-5, 1e-6, 1e-7",
    }
    rows = []
    for algo in algos:
        for k, v in defaults.items():
            if k.startswith(algo + "."):
                rows.append({
                    "id": _uuid.uuid4().hex,
                    "param": k,
                    "values": v,
                    "enabled": True,
                })
    return rows


@st.dialog("Sensitivity Analysis")
def _sensitivity_dialog():
    """Multi-OAT sensitivity analysis dialog.

    Renders a list of (parameter, values) rows with checkboxes,
    allows add/delete/reset, and starts a batch job on Run.
    """
    import uuid as _uuid

    # Lazy init
    if st.session_state.get("sensitivity_plan") is None:
        st.session_state.sensitivity_plan = _make_default_plan(
            list(st.session_state.get("algo_input", ["GA"]))
        )
    if st.session_state.get("sensitivity_algos") is None:
        st.session_state.sensitivity_algos = list(
            st.session_state.get("algo_input", ["GA"])
        )

    catalog = build_sensitivity_parameters()
    catalog_keys = list(catalog.keys())

    # ---- Algorithm(s) ----
    algos = st.multiselect(
        "Algorithm(s) to analyze",
        options=list_algorithms(),
        format_func=_algo_display_label,
        key="sensitivity_algos_widget",
        default=st.session_state.sensitivity_algos,
    )
    st.session_state.sensitivity_algos = algos

    # ---- Runs per point ----
    runs = st.number_input(
        "Runs per point",
        min_value=1, max_value=50,
        value=int(st.session_state.sensitivity_runs), step=1,
        key="sensitivity_runs_widget",
        help="Use ≥ 5 for reliable statistics.",
    )
    st.session_state.sensitivity_runs = int(runs)

    # ---- Plan rows ----
    st.markdown("**Plan**")

    to_delete = None
    for row in st.session_state.sensitivity_plan:
        rid = row["id"]
        c1, c2, c3, c4 = st.columns([1, 3, 4, 1])

        en_key = f"sen_en_{rid}"
        par_key = f"sen_par_{rid}"
        val_key = f"sen_val_{rid}"
        if en_key not in st.session_state:
            st.session_state[en_key] = bool(row["enabled"])
        if par_key not in st.session_state:
            st.session_state[par_key] = row["param"]
        if val_key not in st.session_state:
            st.session_state[val_key] = row["values"]

        with c1:
            row["enabled"] = st.checkbox(
                "Enabled", key=en_key, label_visibility="collapsed",
            )
        with c2:
            cur = st.session_state[par_key]
            if cur not in catalog_keys:
                cur = catalog_keys[0]
                st.session_state[par_key] = cur
            row["param"] = st.selectbox(
                "Parameter", options=catalog_keys, key=par_key,
                format_func=lambda k: catalog[k]["label"],
                label_visibility="collapsed",
            )
        with c3:
            row["values"] = st.text_input(
                "Values", key=val_key,
                placeholder="e.g. 0.6, 0.7, 0.8, 0.9",
                label_visibility="collapsed",
            )
        with c4:
            if st.button("🗑", key=f"sen_del_{rid}"):
                to_delete = rid

        # Contextual hint for choice params
        info = catalog.get(row["param"], {})
        if info.get("kind") == "choice" and info.get("choices"):
            st.caption(
                f"  ↳ Allowed: {', '.join(map(str, info['choices']))}"
            )

    if to_delete:
        st.session_state.sensitivity_plan = [
            r for r in st.session_state.sensitivity_plan
            if r["id"] != to_delete
        ]
        for k in (f"sen_en_{to_delete}", f"sen_par_{to_delete}",
                  f"sen_val_{to_delete}"):
            st.session_state.pop(k, None)
        st.rerun()

    # ---- Add / Reset ----
    c_add, c_reset = st.columns(2)
    with c_add:
        if st.button("+ Add parameter", width="stretch"):
            new_id = _uuid.uuid4().hex
            default_param = catalog_keys[0]
            default_vals = "0.5, 0.7, 0.9"
            st.session_state[f"sen_en_{new_id}"] = True
            st.session_state[f"sen_par_{new_id}"] = default_param
            st.session_state[f"sen_val_{new_id}"] = default_vals
            st.session_state.sensitivity_plan.append({
                "id": new_id,
                "param": default_param,
                "values": default_vals,
                "enabled": True,
            })
            st.rerun()
    with c_reset:
        if st.button("Reset to defaults", width="stretch"):
            st.session_state.sensitivity_plan = _make_default_plan(algos)
            # Drop stale widget state
            for k in [k for k in list(st.session_state.keys())
                      if k.startswith("sen_")]:
                st.session_state.pop(k, None)
            st.rerun()

    # ---- Estimate ----
    enabled_rows = [r for r in st.session_state.sensitivity_plan
                    if r["enabled"]]
    n_values = 0
    for r in enabled_rows:
        try:
            n_values += len(parse_sensitivity_values(r["values"]))
        except ValueError:
            pass
    st.caption(
        f"Total: {len(enabled_rows)} parameter(s) × ~{n_values} "
        f"value(s) × {runs} run(s) = ~{n_values * int(runs)} runs"
    )

    # ---- Actions ----
    c_run, c_cancel = st.columns(2)
    with c_run:
        if st.button("▶ Run batch", type="primary", width="stretch"):
            try:
                _start_sensitivity_batch(
                    plan_rows=enabled_rows,
                    algos=algos,
                    runs=int(runs),
                )
                st.session_state.sensitivity_dialog_open = False
                st.rerun()
            except ValueError as e:
                st.error(f"Cannot start: {e}")
    with c_cancel:
        if st.button("Cancel", width="stretch"):
            st.session_state.sensitivity_dialog_open = False
            st.rerun()


def _start_sensitivity_batch(*, plan_rows, algos, runs):
    """Validate and start a sensitivity batch as a background job."""
    import copy as _copy

    if not plan_rows:
        raise ValueError("No enabled parameters in the plan.")
    if not algos:
        raise ValueError("Select at least one algorithm.")

    catalog = build_sensitivity_parameters()

    backend_rows: list[PlanRow] = []
    skipped = []
    for r in plan_rows:
        try:
            values = parse_sensitivity_values(r["values"])
        except ValueError as e:
            raise ValueError(f"Cannot parse {r['param']}: {e}")
        if not values:
            continue
        owning = catalog.get(
            normalize_param_name(r["param"]), {}
        ).get("algorithm")
        if owning and owning not in algos:
            skipped.append(r["param"])
            continue
        backend_rows.append(PlanRow(
            param=r["param"],
            algorithm=owning or algos[0],
            values=values,
        ))
    if not backend_rows:
        raise ValueError(
            "No parameters remain after filtering by algorithm "
            + (f"({skipped} skipped)" if skipped else "")
        )
    if skipped:
        st.warning(
            f"Skipped params for unselected algorithms: {skipped}"
        )

    try:
        validate_plan(backend_rows)
    except ValueError as e:
        raise ValueError(str(e))

    problem, units, wn = _prepare_problem()
    base_cfg = _build_base_cfg()
    seed = int(st.session_state.exp_seed)

    entry_context = {
        "inp_path": st.session_state.get("inp_path", ""),
        "d_path": st.session_state.get("d_path", ""),
        "cost_path": st.session_state.get("cost_path", ""),
        "diam_unit": str(st.session_state.get("diam_unit_input", "mm")),
        "constraints": _copy.deepcopy(problem["Constraints"]),
        "units": _copy.deepcopy(units),
        "algorithm_params": _algorithm_params_from_sidebar(algos),
        "constraint_policy": str(base_cfg.constraint_policy),
        "fixed_pipes": str(st.session_state.get("fixed_pipes_input", "")),
    }

    n_values = sum(len(r.values) for r in backend_rows)
    summary = (
        f"**Sensitivity batch:** {len(backend_rows)} parameter(s) × "
        f"~{n_values} value(s) × {runs} run(s) | "
        f"Algorithms: {', '.join(algos)}"
    )

    def target(progress_cb, cancel_cb):
        return run_batch_sweep(
            problem, base_cfg,
            rows=backend_rows,
            n_runs=runs,
            experiment_seed=seed,
            progress_callback=progress_cb,
            cancel_callback=cancel_cb,
        )

    job = BackgroundJob(
        "sensitivity", label="Sensitivity Analysis",
        context={
            "problem": problem, "units": units, "wn": wn,
            "algos": algos, "runs": runs,
            "entry_context": entry_context,
            "summary": summary,
        },
    )
    job.start(target)
    st.session_state.job = job
    st.session_state.job_notice = None


def _finalize_sensitivity(job: BackgroundJob):
    """Runs in main thread after the worker ends."""
    st.session_state.job = None
    elapsed = job.elapsed

    if job.error is not None:
        _notice("error",
                f"Sensitivity batch failed: {type(job.error).__name__}: "
                f"{job.error}", job.error_trace)
        st.rerun()

    result = job.result
    ctx = job.context

    batches = st.session_state.get("sensitivity_batches") or []
    record = {
        "batch": result,
        "benchmark_name": Path(
            ctx["entry_context"]["inp_path"]
        ).stem,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "algos": list(ctx["algos"]),
        "runs": int(ctx["runs"]),
        "cancelled": bool(result.cancelled),
    }
    batches.append(record)
    st.session_state.sensitivity_batches = batches

    try:
        _create_sensitivity_history_entries(
            result, ctx["entry_context"], len(batches) - 1,
        )
    except Exception as e:
        print(f"[finalize_sensitivity] history append failed: {e}")

    n_done = result.points_done
    n_total = result.total_points
    if result.cancelled:
        _notice("warning",
                f"Sensitivity batch cancelled after {n_done}/{n_total} "
                f"points ({elapsed:.1f}s).  Partial results kept.")
    else:
        _notice("success",
                f"Sensitivity batch complete ({n_done} points, "
                f"{elapsed:.1f}s).  See the Sensitivity tab.")

    st.session_state._clear_run_name = True
    st.rerun()


def _create_sensitivity_history_entries(
    batch, entry_context: dict, batch_idx: int,
) -> None:
    """Create one HistoryEntry per point in a sensitivity batch.

    All entries share one sweep_group_id so the History panel groups
    them as a single unit.
    """
    import copy as _copy
    import uuid as _uuid
    from wds_optimizer.hydraulics.units import (
        pressure_to_si, velocity_to_si,
    )

    group_id = _uuid.uuid4().hex
    units = entry_context.get("units") or {}
    user_name = str(st.session_state.get("run_name_input", "")).strip()
    base_name = (
        (user_name + " (sensitivity)") if user_name
        else f"Sensitivity batch #{batch_idx + 1}"
    )

    for pr in batch.params:
        for point in pr.points:
            if point.experiment is None:
                continue

            norm_param = normalize_param_name(pr.param)
            entry_constraints = _copy.deepcopy(
                entry_context.get("constraints") or {}
            )
            entry_algo_params = _copy.deepcopy(
                entry_context.get("algorithm_params") or {}
            )
            entry_policy = entry_context.get(
                "constraint_policy", "deb_prefers",
            )

            if norm_param in ("Pmin", "Pmax", "Vmin", "Vmax"):
                try:
                    if norm_param in ("Pmin", "Pmax"):
                        si = pressure_to_si(float(point.value), units)
                    else:
                        si = velocity_to_si(float(point.value), units)
                    entry_constraints[norm_param] = {
                        "enabled": True, "value": float(si),
                    }
                except Exception:
                    pass
            elif norm_param == "constraint_policy":
                entry_policy = str(point.value)
            elif "." in norm_param:
                aid, pkey = norm_param.split(".", 1)
                if aid in entry_algo_params:
                    entry_algo_params[aid][pkey] = point.value

            entry_name = (
                f"{base_name} · {pr.param}="
                f"{_format_sweep_value(point.value)}"
            )
            entry = new_entry(
                entry_name,
                inp_path=entry_context.get("inp_path", ""),
                d_path=entry_context.get("d_path", ""),
                cost_path=entry_context.get("cost_path", ""),
                diam_unit=entry_context.get("diam_unit", "mm"),
                algorithms=[pr.algorithm],
                n_runs=int(batch.n_runs),
                seed=int(batch.experiment_seed),
                ns=int(getattr(point.experiment, "NS", 100)),
                max_gen=int(getattr(point.experiment, "MaxGen", 500)),
                constraints=entry_constraints,
                algorithm_params=entry_algo_params,
                constraint_policy=entry_policy,
                fixed_pipes=entry_context.get("fixed_pipes", ""),
            )
            entry.experiment = point.experiment
            entry.elapsed_seconds = float(point.mean_time or 0.0)
            entry.status = (
                "Cancelled"
                if getattr(point.experiment, "cancelled", False)
                else "Complete"
            )
            entry.sweep_group_id = group_id
            entry.sweep_meta = {
                "param": pr.param,
                "value": point.value,
                "algorithm": pr.algorithm,
            }
            st.session_state.history.append(entry)


def _render_sensitivity_tab():
    """Report tab for sensitivity batches (current session)."""
    batches = st.session_state.get("sensitivity_batches") or []
    if not batches:
        st.info(
            "No sensitivity batches yet.  Run one from the sidebar "
            "(section 4: Sensitivity Analysis)."
        )
        return

    labels = [
        f"#{i+1}: {b['benchmark_name']} — "
        f"{len(b['batch'].params)} param(s), "
        f"{b['batch'].points_done} points "
        f"({b['timestamp']})"
        for i, b in enumerate(batches)
    ]
    idx = st.selectbox(
        "Batch", range(len(batches)),
        format_func=lambda i: labels[i],
        index=len(batches) - 1,
        key="sensitivity_batch_selector",
    )
    record = batches[idx]
    batch = record["batch"]

    if record.get("cancelled"):
        st.warning(
            "This batch was cancelled; only completed points are shown."
        )

    st.caption(
        f"**{record['benchmark_name']}** | "
        f"Algorithms: {', '.join(record['algos'])} | "
        f"Runs/point: {record['runs']} | "
        f"Seed: {batch.experiment_seed}"
    )

    for pr in batch.params:
        st.subheader(f"`{pr.param}` — {pr.algorithm}")

        rows = []
        for p in pr.points:
            rows.append({
                "Value": _format_sweep_value(p.value),
                "Status": p.status,
                "Best": _fmt_cost(p.best_score),
                "Mean": _fmt_cost(p.mean_score),
                "Std": _fmt_cost(p.std_score),
                "Feasible %": f"{p.feasible_pct:.0f}%",
                "Time (s)": f"{p.mean_time:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch",
                     hide_index=True)

        notes, is_flat = _annotations_for_param(pr, batch.n_runs)
        for note in notes:
            st.caption(f"• {note}")

        if is_flat:
            st.warning(
                "No recommendation — parameter has no measurable "
                "effect on this benchmark."
            )
        elif pr.recommended_value is not None:
            st.success(
                f"Recommended: `{pr.param} = "
                f"{_format_sweep_value(pr.recommended_value)}` "
                f"(mean {_fmt_cost(pr.recommended_mean)})"
            )
        elif pr.reason:
            st.caption(f"Note: {pr.reason}")

    st.divider()
    c1, c2 = st.columns(2)
    md = batch_report_markdown(
        batch, benchmark_name=record["benchmark_name"],
    )
    safe_ts = record["timestamp"].replace(":", "-")
    with c1:
        st.download_button(
            "⬇ Download markdown",
            data=md.encode("utf-8"),
            file_name=f"sensitivity_{safe_ts}.md",
            mime="text/markdown",
            width="stretch",
        )
    with c2:
        st.download_button(
            "⬇ Download CSV",
            data=batch_report_csv(batch),
            file_name=f"sensitivity_{safe_ts}.csv",
            mime="text/csv",
            width="stretch",
        )

def _entry_from_sidebar(name: str) -> HistoryEntry:
    """Build a HistoryEntry (Draft, no result) from current sidebar.

    Reads widget keys (``pmin_input``, ``ga_pc_input``, etc.), not
    canonical keys, so the entry reflects what is on screen — not
    what was last synced by a previous Run.
    """
    from wds_optimizer.algorithms.registry import ALGORITHM_REGISTRY
    from wds_optimizer.hydraulics.units import (
        pressure_to_si, velocity_to_si,
    )

    if not st.session_state.get("inp_path"):
        raise ValueError("Load input files first.")

    units = st.session_state.get("network_units")
    if units is None:
        raise ValueError("Network units not detected — reload .inp.")

    constraints = {
        "Pmin": {
            "enabled": bool(st.session_state.get("use_pmin_input", True)),
            "value": pressure_to_si(
                float(st.session_state.get("pmin_input", 30.0)), units),
        },
        "Pmax": {
            "enabled": bool(st.session_state.get("use_pmax_input", False)),
            "value": pressure_to_si(
                float(st.session_state.get("pmax_input", 100.0)), units),
        },
        "Vmin": {
            "enabled": bool(st.session_state.get("use_vmin_input", False)),
            "value": velocity_to_si(
                float(st.session_state.get("vmin_input", 0.0)), units),
        },
        "Vmax": {
            "enabled": bool(st.session_state.get("use_vmax_input", True)),
            "value": velocity_to_si(
                float(st.session_state.get("vmax_input", 2.5)), units),
        },
    }

    algorithms = list(st.session_state.get("algo_input", []))
    algo_params = _algorithm_params_from_sidebar(algorithms)

    return new_entry(
        name=name,
        inp_path=st.session_state.inp_path,
        d_path=st.session_state.d_path,
        cost_path=st.session_state.cost_path,
        diam_unit=str(st.session_state.get("diam_unit_input", "mm")),
        algorithms=algorithms,
        n_runs=int(st.session_state.get("n_runs_input", 1)),
        seed=int(st.session_state.get("exp_seed_input", 1)),
        ns=int(st.session_state.get("ns_input", 50)),
        max_gen=int(st.session_state.get("max_gen_input", 500)),
        constraints=constraints,
        algorithm_params=algo_params,
        constraint_policy=str(st.session_state.get(
            "constraint_policy_input", "deb_prefers")),
        fixed_pipes=str(st.session_state.get("fixed_pipes_input", "")),
    )


def _append_to_history(result, elapsed: float, cancelled: bool,
                       run_name: str) -> None:
    """Add an entry to history for the just-completed run."""
    import copy as _copy
    history = st.session_state.get("history") or []

    auto = auto_name(
        algorithms=list(result.algorithms),
        inp_path=st.session_state.get("inp_path", ""),
        ns=int(result.NS),
        seed=int(result.experiment_seed),
    )
    name = (run_name or "").strip() or auto

    snap = st.session_state.get("problem_snapshot") or {}
    constraints = _constraints_from_snapshot(snap) if snap else {
        "Pmin": {"enabled": True, "value": 30.0},
        "Pmax": {"enabled": False, "value": 100.0},
        "Vmin": {"enabled": False, "value": 0.0},
        "Vmax": {"enabled": True, "value": 2.5},
    }

    from wds_optimizer.algorithms.registry import ALGORITHM_REGISTRY
    algo_params: dict[str, dict] = {}
    for short_id in result.algorithms:
        entry_meta = ALGORITHM_REGISTRY.get(short_id)
        if entry_meta is None:
            continue
        params: dict = {}
        for p in entry_meta["meta"]["params"]:
            wk = p["key"] + "_input"
            params[p["key"]] = st.session_state.get(wk, p["default"])
        algo_params[short_id] = params

    entry = new_entry(
        name=name,
        inp_path=st.session_state.get("inp_path", ""),
        d_path=st.session_state.get("d_path", ""),
        cost_path=st.session_state.get("cost_path", ""),
        diam_unit=str(st.session_state.get("diam_unit", "mm")),
        algorithms=list(result.algorithms),
        n_runs=int(result.number_of_runs),
        seed=int(result.experiment_seed),
        ns=int(result.NS),
        max_gen=int(result.MaxGen),
        constraints=constraints,
        algorithm_params=algo_params,
        constraint_policy=str(st.session_state.get(
            "constraint_policy_input", "deb_prefers")),
        fixed_pipes=str(st.session_state.get("fixed_pipes_input", "")),
    )
    entry.experiment = result
    entry.elapsed_seconds = float(elapsed)
    entry.status = "Cancelled" if cancelled else "Complete"

    history.append(entry)
    st.session_state.history = history

def _create_sweep_history_entries(result, entry_context: dict) -> None:
    """Add one HistoryEntry per completed sweep point.

    All entries share a sweep_group_id so the History panel can group
    them.  Each entry's config reflects the effective values used for
    that point (constraints in SI, algorithm params after the swept
    value was applied, per-point NS/MaxGen, per-point constraint
    policy).
    """
    import copy as _copy
    import uuid as _uuid
    from wds_optimizer.sensitivity.config import normalize_param_name
    from wds_optimizer.hydraulics.units import (
        pressure_to_si, velocity_to_si,
    )

    group_id = _uuid.uuid4().hex
    units = entry_context.get("units") or {}

    # Base name: user's run_name_input if any, else auto.
    user_name = str(st.session_state.get("run_name_input", "")).strip()
    if user_name:
        base_name = user_name
    else:
        base_name = auto_name(
            algorithms=list(result.algorithms),
            inp_path=entry_context.get("inp_path", ""),
            ns=int(entry_context.get("ns", 0)
                   or st.session_state.get("ns", 50)),
            seed=int(result.experiment_seed),
        )

    norm_param = normalize_param_name(result.parameter)

    for point in result.points:
        if point.experiment is None:
            continue

        # Derive effective constraints and algorithm params for this
        # point from the base config + the swept value.
        entry_constraints = _copy.deepcopy(
            entry_context.get("constraints") or {}
        )
        entry_algo_params = _copy.deepcopy(
            entry_context.get("algorithm_params") or {}
        )
        entry_policy = str(
            getattr(point.experiment, "constraint_policy",
                    entry_context.get("constraint_policy", "deb_prefers"))
        )

        if norm_param in ("Pmin", "Pmax", "Vmin", "Vmax"):
            try:
                if norm_param in ("Pmin", "Pmax"):
                    si = pressure_to_si(float(point.value), units)
                else:
                    si = velocity_to_si(float(point.value), units)
                entry_constraints[norm_param] = {
                    "enabled": True, "value": float(si),
                }
            except Exception:
                pass
        elif norm_param == "constraint_policy":
            entry_policy = str(point.value)
        elif norm_param == "NS":
            # experiment.NS reflects the applied value
            pass
        elif norm_param == "MaxGen":
            pass
        elif norm_param == "n_runs":
            pass
        elif norm_param == "seed":
            pass
        elif "." in norm_param:
            algo_id, pkey = norm_param.split(".", 1)
            if algo_id in entry_algo_params:
                entry_algo_params[algo_id][pkey] = point.value

        entry_name = (
            f"{base_name} \u00b7 {norm_param}="
            f"{_format_sweep_value(point.value)}"
        )
        entry = new_entry(
            entry_name,
            inp_path=entry_context.get("inp_path", ""),
            d_path=entry_context.get("d_path", ""),
            cost_path=entry_context.get("cost_path", ""),
            diam_unit=entry_context.get("diam_unit", "mm"),
            algorithms=list(result.algorithms),
            n_runs=int(result.n_runs),
            seed=int(result.experiment_seed),
            ns=int(getattr(point.experiment, "NS",
                           st.session_state.get("ns", 50))),
            max_gen=int(getattr(point.experiment, "MaxGen",
                               st.session_state.get("max_gen", 500))),
            constraints=entry_constraints,
            algorithm_params=entry_algo_params,
            constraint_policy=entry_policy,
            fixed_pipes=entry_context.get("fixed_pipes", ""),
        )
        entry.experiment = point.experiment
        entry.elapsed_seconds = float(point.mean_time or 0.0)
        entry.status = (
            "Cancelled" if getattr(point.experiment, "cancelled", False)
            else "Complete"
        )
        entry.sweep_group_id = group_id
        entry.sweep_meta = {"param": norm_param, "value": point.value}
        st.session_state.history.append(entry)

def _load_history_config(entry: HistoryEntry) -> None:
    """Load an entry's config into the sidebar.

    Writes embedded inputs to temp files (so an entry is portable even
    if its original paths are gone).  Clears any current experiment so
    tabs don't show stale results.  The entry's *result* is not loaded
    — to see it, use the Compare tab with the entry's saved JSON, or
    just Run again to produce fresh results with the same config.
    """
    import uuid as _uuid
    from wds_optimizer.algorithms.registry import ALGORITHM_REGISTRY
    from wds_optimizer.hydraulics.units import (
        detect_network_units, pressure_from_si, velocity_from_si,
    )

    # ---- Input files ----
    data = history_entry_to_dict(entry, embed_inputs=True)
    embedded = data.get("inputs", {})
    tmpdir = Path(tempfile.gettempdir()) / "wds_optimizer_history"
    tmpdir.mkdir(parents=True, exist_ok=True)
    uid = _uuid.uuid4().hex[:8]

    if embedded.get("inp"):
        p = tmpdir / f"{uid}.inp"
        p.write_text(embedded["inp"], encoding="utf-8")
        new_inp = str(p)
    else:
        new_inp = entry.inp_path

    if embedded.get("D_txt"):
        p = tmpdir / f"{uid}_D.txt"
        p.write_text(embedded["D_txt"], encoding="utf-8")
        new_d = str(p)
    else:
        new_d = entry.d_path

    if embedded.get("cost_txt"):
        p = tmpdir / f"{uid}_Cost.txt"
        p.write_text(embedded["cost_txt"], encoding="utf-8")
        new_cost = str(p)
    else:
        new_cost = entry.cost_path

    # ---- Units on the new file ----
    try:
        import wntr
        new_units = detect_network_units(
            wntr.network.WaterNetworkModel(new_inp))
    except Exception:
        new_units = st.session_state.get("network_units") or {
            "System": "SI", "PressureUnit": "m",
            "VelocityUnit": "m/s", "LengthUnit": "m",
            "DiameterUnit": "mm",
        }

    # ---- Canonical keys ----
    st.session_state.inp_path = new_inp
    st.session_state.d_path = new_d
    st.session_state.cost_path = new_cost
    st.session_state.diam_unit = entry.diam_unit
    st.session_state.network_units = new_units

    # ---- Clear stale experiment ----
    st.session_state.experiment = None
    st.session_state.problem_snapshot = None
    st.session_state.best_algo = None
    st.session_state.best_run_index = None
    st.session_state.best_position_m = None
    st.session_state.best_score = None
    st.session_state.best_feasible = False
    st.session_state.node_pressures = None
    st.session_state.pipe_velocities = None
    st.session_state.hydraulics_converged = False
    _clear_run_ui_state()

    # ---- Widget updates ----
    updates = {
        "algo_input": list(entry.algorithms),
        "ns_input": int(entry.ns),
        "max_gen_input": int(entry.max_gen),
        "n_runs_input": int(entry.n_runs),
        "exp_seed_input": int(entry.seed),
        "constraint_policy_input": str(entry.constraint_policy),
        "fixed_pipes_input": str(entry.fixed_pipes or ""),
        "diam_unit_input": entry.diam_unit,
        "pmin_input": float(pressure_from_si(
            entry.constraints["Pmin"]["value"], new_units)),
        "pmax_input": float(pressure_from_si(
            entry.constraints["Pmax"]["value"], new_units)),
        "vmin_input": float(velocity_from_si(
            entry.constraints["Vmin"]["value"], new_units)),
        "vmax_input": float(velocity_from_si(
            entry.constraints["Vmax"]["value"], new_units)),
        "use_pmin_input": bool(entry.constraints["Pmin"]["enabled"]),
        "use_pmax_input": bool(entry.constraints["Pmax"]["enabled"]),
        "use_vmin_input": bool(entry.constraints["Vmin"]["enabled"]),
        "use_vmax_input": bool(entry.constraints["Vmax"]["enabled"]),
    }
    for short_id, params in entry.algorithm_params.items():
        entry_meta = ALGORITHM_REGISTRY.get(short_id)
        if entry_meta is None:
            continue
        for p in entry_meta["meta"]["params"]:
            if p["key"] in params:
                updates[p["key"] + "_input"] = params[p["key"]]

    st.session_state._pending_restore = updates
    st.rerun()


def _render_history_section() -> None:
    """Render the History panel in the sidebar."""
    import json as _json

    history: list[HistoryEntry] = st.session_state.get("history") or []

    c_save, c_load = st.columns(2)
    with c_save:
        if history:
            st.download_button(
                "💾 Save all",
                data=_json.dumps(
                    history_to_dict(history),
                    indent=2, ensure_ascii=False,
                ).encode("utf-8"),
                file_name=(
                    f"wds_history_"
                    f"{datetime.now():%Y%m%d_%H%M%S}.json"
                ),
                mime="application/json",
                width="stretch",
                help="Download the whole history as one JSON.  "
                     "Reload with 'Load' below.",
            )
        else:
            st.button("💾 Save all", width="stretch",
                      disabled=True, key="hist_save_disabled")
    with c_load:
        hist_upload = st.file_uploader(
            "Load", type=["json"], key="hist_upload",
            label_visibility="collapsed",
        )
    if hist_upload is not None:
        if st.button(
            "Load (replaces current history)",
            width="stretch", key="hist_load_confirm",
        ):
            try:
                st.session_state.history = load_history(hist_upload)
                st.rerun()
            except Exception as e:
                st.error(f"Load failed: {type(e).__name__}: {e}")

    if not history:
        st.caption(
            "No history yet.  Each run is added automatically.  "
            "History is session-local; use 'Save all' to preserve."
        )
        return

    if st.button("🗑 Clear all history", width="stretch",
                 key="hist_clear_all",
                 help="Removes all entries; cannot be undone."):
        st.session_state.history = []
        st.rerun()

    # Group by sweep_group_id; standalone entries stay as-is.
    sweep_groups: dict[str, list[HistoryEntry]] = {}
    standalone: list[HistoryEntry] = []
    for e in history:
        if e.sweep_group_id:
            sweep_groups.setdefault(e.sweep_group_id, []).append(e)
        else:
            standalone.append(e)

    rows: list[tuple[str, str, object]] = []
    for e in standalone:
        rows.append((e.created_at, "single", e))
    for gid, entries in sweep_groups.items():
        ts = entries[0].created_at if entries else ""
        rows.append((ts, "sweep", (gid, entries)))
    rows.sort(key=lambda x: x[0], reverse=True)

    for _, kind, data in rows:
        if kind == "single":
            _render_one_history_row(data)
        else:
            gid, entries = data
            _render_sensitivity_group_row(gid, entries)


def _render_one_history_row(entry: HistoryEntry) -> None:
    # Status + cost
    best_cost = None
    if entry.experiment is not None:
        for runs in entry.experiment.runs.values():
            for r in runs:
                if r.feasible and (
                    best_cost is None or r.score < best_cost
                ):
                    best_cost = r.score

    if entry.status == "Complete":
        badge = "✓"
        cost_str = f"${best_cost:,.0f}" if best_cost is not None \
                   else "infeasible"
    elif entry.status == "Cancelled":
        badge = "⏹"; cost_str = "cancelled"
    elif entry.status == "Failed":
        badge = "✗"; cost_str = "failed"
    else:
        badge = "?"; cost_str = entry.status

    # Rename mode
    if st.session_state.get("history_rename_id") == entry.entry_id:
        rk = f"rename_input_{entry.entry_id}"
        if rk not in st.session_state:
            st.session_state[rk] = entry.name
        new_name = st.text_input(
            "New name",
            key=rk,
            label_visibility="collapsed",
        )
        c_save, c_cancel = st.columns(2)
        with c_save:
            if st.button("Save", width="stretch", type="primary",
                         key=f"rn_save_{entry.entry_id}"):
                entry.name = new_name.strip() or entry.name
                st.session_state.history_rename_id = None
                # Clear the rename input so the next rename session
                # starts from the (possibly new) entry name.
                st.session_state.pop(rk, None)
                st.rerun()
        with c_cancel:
            if st.button("Cancel", width="stretch",
                         key=f"rn_cancel_{entry.entry_id}"):
                st.session_state.history_rename_id = None
                st.session_state.pop(rk, None)
                st.rerun()
        return

    # Normal display
    c_name, c_load, c_rename, c_save, c_del = st.columns([4, 1, 1, 1, 1])
    with c_name:
        st.markdown(f"**{entry.name}** {badge} {cost_str}")
    with c_load:
        if st.button("📂", key=f"hl_{entry.entry_id}",
                     help="Load config into sidebar (edit and re-run)"):
            _load_history_config(entry)
    with c_rename:
        if st.button("✏️", key=f"hr_{entry.entry_id}",
                     help="Rename"):
            st.session_state.history_rename_id = entry.entry_id
            # Clear any stale text from a previous rename session for
            # the same entry — otherwise the input shows the previous
            # typed-but-not-saved name.
            st.session_state.pop(f"rename_input_{entry.entry_id}", None)
            st.rerun()
    with c_save:
        import json as _json
        st.download_button(
            "💾",
            data=_json.dumps(
                history_entry_to_dict(entry),
                indent=2, ensure_ascii=False,
            ).encode("utf-8"),
            file_name=f"{entry.name[:40] or 'entry'}.json",
            mime="application/json",
            key=f"hs_{entry.entry_id}",
            help="Download this entry as JSON (can be uploaded in "
                 "Compare tab or re-loaded here).",
        )
    with c_del:
        if st.button("🗑", key=f"hd_{entry.entry_id}",
                     help="Delete"):
            st.session_state.history_delete_id = entry.entry_id
            st.rerun()

    if st.session_state.get("history_delete_id") == entry.entry_id:
        st.warning(f"Delete **{entry.name}**?  Cannot be undone.")
        cy, cn = st.columns(2)
        with cy:
            if st.button("Yes, delete", width="stretch",
                         type="primary",
                         key=f"hyd_{entry.entry_id}"):
                st.session_state.history = [
                    e for e in st.session_state.history
                    if e.entry_id != entry.entry_id
                ]
                st.session_state.history_delete_id = None
                st.rerun()
        with cn:
            if st.button("Cancel", width="stretch",
                         key=f"hnd_{entry.entry_id}"):
                st.session_state.history_delete_id = None
                st.rerun()

def _render_sensitivity_group_row(
    group_id: str, entries: list["HistoryEntry"],
) -> None:
    """Render a collapsed row for a sweep/batch group.

    Handles both:
    - single-parameter sweeps (all entries share one param),
    - sensitivity batches (multiple params in one group).
    """
    params = set()
    for e in entries:
        if e.sweep_meta and e.sweep_meta.get("param"):
            params.add(e.sweep_meta["param"])

    n_points = len(entries)
    best_cost = float("inf")
    best_value = None
    for e in entries:
        if e.experiment is None:
            continue
        for runs in e.experiment.runs.values():
            for r in runs:
                if r.status in ("Success", "Partial") and r.feasible:
                    if r.score < best_cost:
                        best_cost = r.score
                        best_value = (
                            e.sweep_meta.get("value")
                            if e.sweep_meta else None
                        )

    if len(params) == 1:
        label = f"🔬 Sweep: {next(iter(params))}"
    else:
        label = f"🔬 Sensitivity batch: {len(params)} parameter(s)"

    if np.isfinite(best_cost):
        summary = (
            f"{label} · {n_points} points · best ${best_cost:,.0f}"
            + (f" at {_format_sweep_value(best_value)}"
               if best_value is not None else "")
        )
    else:
        summary = f"{label} · {n_points} points · no feasible"

    with st.expander(summary, expanded=False):
        c_del, c_hint = st.columns([1, 3])
        with c_del:
            if st.button("🗑 Delete group",
                         key=f"sens_del_{group_id}",
                         width="stretch"):
                st.session_state.history = [
                    e for e in st.session_state.history
                    if e.sweep_group_id != group_id
                ]
                st.rerun()
        with c_hint:
            st.caption(
                "Open the Sensitivity tab to see the report and "
                "download it.  Individual points can be loaded here."
            )
        for e in entries:
            _render_one_history_row(e)

def _config_for_snapshot() -> dict:
    """Flat config dict for build_snapshot / build_report_zip.

    Algorithm params (ga_pc, ga_pm, ...) come from session_state
    directly, as before.  ``constraint_policy`` comes from its WIDGET
    key (``constraint_policy_input``), not the canonical key.  Reason:
    the canonical key is only synced at run start by
    _sync_sidebar_state, so if the user changes the dropdown after
    a run and then saves, the canonical key is stale.  Reading the
    widget key means the CURRENT selection is saved.
    """
    keys = (
        "ns", "max_gen", "ga_pc", "ga_pm", "ga_sel",
        "pso_w", "pso_c1", "pso_c2", "n_runs", "exp_seed",
        "de_f", "de_cr", "de_strategy",
        "sade_fmin", "sade_fmax", "sade_crmin",
        "sade_crmax", "sade_conv", "sade_tol",
        "sade_stag",
    )
    cfg = {k: st.session_state[k] for k in keys}
    cfg["constraint_policy"] = str(
        st.session_state.get("constraint_policy_input", "deb_prefers")
    )
    return cfg

def _render_params_from_meta(meta: dict) -> None:
    """Render widgets for every param in ALGORITHM_META['params'].

    The widget type is chosen from ``param['widget']`` if present,
    else from ``param['type']``.  ``depends_on`` (if present) disables
    the widget when another widget's state doesn't match.
    """
    for p in meta["params"]:
        widget_key = p["key"] + "_input"
        disabled = False
        depends = p.get("depends_on")
        if depends:
            dep_key, dep_expected = depends
            disabled = bool(st.session_state.get(dep_key, False)) != bool(dep_expected)

        widget_hint = p.get("widget")

        if widget_hint == "select_slider" and p.get("options"):
            st.select_slider(
                p["display_name"],
                options=list(p["options"]),
                key=widget_key,
                format_func=lambda v: f"{v:.0e}",
                disabled=disabled,
                help=p.get("description", ""),
            )
        elif p["type"] == "float":
            st.number_input(
                p["display_name"],
                min_value=float(p.get("min", 0.0)),
                max_value=float(p.get("max", 1_000_000.0)),
                step=float(p.get("step", 0.01)),
                key=widget_key,
                disabled=disabled,
                help=p.get("description", ""),
            )
        elif p["type"] == "int":
            st.number_input(
                p["display_name"],
                min_value=int(p.get("min", 0)),
                max_value=int(p.get("max", 1_000_000)),
                step=int(p.get("step", 1)),
                key=widget_key,
                disabled=disabled,
                help=p.get("description", ""),
            )
        elif p["type"] == "choice":
            st.selectbox(
                p["display_name"],
                options=p["choices"],
                key=widget_key,
                disabled=disabled,
                help=p.get("description", ""),
            )
        elif p["type"] == "bool":
            st.checkbox(
                p["display_name"],
                key=widget_key,
                disabled=disabled,
                help=p.get("description", ""),
            )

# =====================================================================
#  Preparation: build the Problem dict from loaded files
# =====================================================================

def _prepare_problem():
    """GUI wrapper around ``api.build_problem``.

    Two responsibilities remain here that do not belong in the API:

    1. **Display-unit → SI conversion.**  The user enters Pmin/Pmax/Vmin/
       Vmax in the network's native units (psi, m, ft/s, ...).  The API
       consumes SI only.  This conversion is GUI-side by design — the
       API never sees display units.

    2. **User-friendly validation errors.**  The API validates in SI
       for the safety of CLI/embedded callers; the GUI re-validates in
       display units so error messages show the numbers the user typed
       (e.g. ``Pmin (30.0)`` not ``Pmin (2.11)`` for a psi network).
       GUI-side validation is checked first; the API's is a safety net.
    """
    units = st.session_state.network_units
    if units is None:
        raise ValueError(
            "Network units not detected — load an .inp file first."
        )

    # Display units → SI
    constraints = {
        "Pmin": {"enabled": bool(st.session_state.use_pmin),
                 "value": pressure_to_si(st.session_state.pmin, units)},
        "Pmax": {"enabled": bool(st.session_state.use_pmax),
                 "value": pressure_to_si(st.session_state.pmax, units)},
        "Vmin": {"enabled": bool(st.session_state.use_vmin),
                 "value": velocity_to_si(st.session_state.vmin, units)},
        "Vmax": {"enabled": bool(st.session_state.use_vmax),
                 "value": velocity_to_si(st.session_state.vmax, units)},
    }

    # User-friendly validation, display units (API re-checks in SI)
    if not any(c["enabled"] for c in constraints.values()):
        raise ValueError("Enable at least one constraint before running.")
    if (st.session_state.use_pmin and st.session_state.use_pmax
            and st.session_state.pmin >= st.session_state.pmax):
        raise ValueError(
            f"Pmin ({st.session_state.pmin}) must be smaller than "
            f"Pmax ({st.session_state.pmax})."
        )
    if (st.session_state.use_vmin and st.session_state.use_vmax
            and st.session_state.vmin >= st.session_state.vmax):
        raise ValueError(
            f"Vmin ({st.session_state.vmin}) must be smaller than "
            f"Vmax ({st.session_state.vmax})."
        )

    return build_problem(
        inp_path=st.session_state.inp_path,
        d_path=st.session_state.d_path,
        cost_path=st.session_state.cost_path,
        diam_unit=st.session_state.diam_unit,
        constraints=constraints,
        fixed_pipes=st.session_state.get("fixed_pipes", ""),
    )


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
        ("pmin", "pmin_input"), ("pmax", "pmax_input"),
        ("vmin", "vmin_input"), ("vmax", "vmax_input"),
        ("use_pmin", "use_pmin_input"),
        ("use_pmax", "use_pmax_input"),
        ("use_vmin", "use_vmin_input"),
        ("use_vmax", "use_vmax_input"),
        ("fixed_pipes", "fixed_pipes_input"), ("exp_seed", "exp_seed_input"),
        ("ga_pc", "ga_pc_input"), ("ga_pm", "ga_pm_input"),
        ("ga_sel", "ga_sel_input"),
        ("pso_w", "pso_w_input"), ("pso_c1", "pso_c1_input"),
        ("pso_c2", "pso_c2_input"),
        ("de_f", "de_f_input"), ("de_cr", "de_cr_input"),
        ("de_strategy", "de_strategy_input"),
        ("sade_fmin", "sade_fmin_input"), ("sade_fmax", "sade_fmax_input"),
        ("sade_crmin", "sade_crmin_input"),
        ("sade_crmax", "sade_crmax_input"),
        ("sade_conv", "sade_conv_input"), ("sade_tol", "sade_tol_input"),
        ("sade_stag", "sade_stag_input"),
        ("constraint_policy", "constraint_policy_input"),
    ]
    if with_n_runs:
        pairs.append(("n_runs", "n_runs_input"))
    for dst, src in pairs:
        if src in st.session_state:
            st.session_state[dst] = st.session_state[src]

    # Algorithm params: widget_key → canonical key, driven by META.
    # This keeps the canonical keys (ga_pc, ga_pm, ...) in sync for
    # snapshot save.  Plugin algorithms get their canonical keys too.
    for entry in ALGORITHM_REGISTRY.values():
        for p in entry["meta"]["params"]:
            widget_key = p["key"] + "_input"
            canonical_key = p["key"]
            if widget_key in st.session_state:
                st.session_state[canonical_key] = st.session_state[widget_key]


def _apply_param_to_config(
    cfg, short_id: str, param_meta: dict, value,
) -> None:
    """Map a META param value onto the config object.

    If the param declares ``maps_to``, set the corresponding field on
    the family's dataclass (e.g. ``cfg.GA.Pc``).  Otherwise store the
    value under ``cfg.extra[short_id][param_key]`` for the algorithm
    to read as it likes (this is the plugin path).
    """
    maps_to = param_meta.get("maps_to")
    if maps_to is not None:
        family, field_name = maps_to
        family_cfg = getattr(cfg, family, None)
        if family_cfg is None or not hasattr(family_cfg, field_name):
            raise ValueError(
                f"META param {param_meta['key']!r} maps to "
                f"{family}.{field_name}, which does not exist on "
                f"AlgorithmConfig.  This is a bug in the algorithm's "
                f"ALGORITHM_META."
            )
        setattr(family_cfg, field_name, value)
    else:
        cfg.extra.setdefault(short_id, {})[param_meta["key"]] = value


def _build_base_cfg():
    cfg = build_algorithm_config(
        int(st.session_state.ns), int(st.session_state.max_gen)
    )
    cfg.constraint_policy = str(
        st.session_state.get("constraint_policy_input", "deb_prefers")
    )

    # Populate each selected algorithm's params from session_state.
    selected = st.session_state.get("algo_input", [])
    for short_id in selected:
        entry = ALGORITHM_REGISTRY.get(short_id)
        if entry is None:
            continue
        for p in entry["meta"]["params"]:
            widget_key = p["key"] + "_input"
            value = st.session_state.get(widget_key, p["default"])
            _apply_param_to_config(cfg, short_id, p, value)

    # Restart is disabled in the GUI (see CHANGELOG v0.2.1).
    cfg.Restart.Enabled = False

    # SADE-specific validation (Fmin/Fmax, CRmin/CRmax).
    if "SADE" in selected:
        if cfg.SADE.Fmin > cfg.SADE.Fmax:
            raise ValueError(
                f"SADE Fmin ({cfg.SADE.Fmin}) must be <= "
                f"Fmax ({cfg.SADE.Fmax})."
            )
        if cfg.SADE.CRmin > cfg.SADE.CRmax:
            raise ValueError(
                f"SADE CRmin ({cfg.SADE.CRmin}) must be <= "
                f"CRmax ({cfg.SADE.CRmax})."
            )

    return cfg
# =====================================================================
#  Background jobs: start -> live panel (fragment) -> finalize
# =====================================================================
# The optimisation runs in a worker thread (see app/jobs.py).  The worker
# never calls st.*; the UI polls the job every 0.5 s from a fragment and,
# once the thread has ended (normally OR after Cancel), stores whatever
# result exists and reruns the app.

def _start_run(algo_label: str, run_name: str = ""):
    _sync_sidebar_state(with_n_runs=True)

    # Clear any stale report from a previous experiment
    st.session_state.report_zip = None
    st.session_state.report_filename = None

    problem, units, wn = _prepare_problem()
    base_cfg = _build_base_cfg()
    algos = [a for a in list_algorithms() if a in (algo_label or [])]
    if not algos:
        raise ValueError("Select at least one algorithm.")
    if ("DE" in algos or "SADE" in algos) and int(st.session_state.ns) < 4:
        raise ValueError("DE and SADE need a population (NS) of at least 4.")
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
                 "restart_on": bool(base_cfg.Restart.Enabled),
                 "run_name": run_name},
    )
    job.start(target)
    st.session_state.job = job
    st.session_state.job_notice = None
    st.rerun()      # redraw sidebar: green button -> red "Cancel"

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
        elif job.kind == "sensitivity":
            _finalize_sensitivity(job)
        return

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
            + f"NFE: **{prog.get('NFE', 0):,}** | "
            + (f"Cv: **{prog['Cv']:.2e}** | "
               if prog.get("Algorithm") == "SADE" and "Cv" in prog else "")
            + (f"Restarts: **{prog.get('Restarts', 0)}** | "
               if ctx.get("restart_on") else "")
            + f"elapsed {job.elapsed:.0f} s"
        )
        conv = prog["Convergence"]
        color = ALGO_COLORS.get(prog["Algorithm"], "gray")
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

    elif job.kind == "sensitivity":
        if "Phase" not in prog:
            st.progress(0.0)
            st.caption("Starting…")
            return
        phase = prog.get("Phase")
        pi = prog.get("PointIndex", 1)
        total = prog.get("TotalPoints", 1)
        frac = min(1.0, pi / max(1, total))
        st.progress(frac)
        if phase == "PointDone":
            st.info(
                f"Point **{pi}/{total}** | "
                f"{prog.get('ParameterName', '?')} = "
                f"**{_format_sweep_value(prog.get('ParameterValue'))}**"
                f" | elapsed {job.elapsed:.0f} s"
            )
        elif phase == "ParamDone":
            st.info(
                f"Finished parameter "
                f"{prog.get('ParamIndex', '?')}/"
                f"{prog.get('ParamTotal', '?')} "
                f"({prog.get('ParameterName', '?')})"
                f" | elapsed {job.elapsed:.0f} s"
            )


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

    c = problem["Constraints"]
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
        # Constraint values (SI) + enabled flags
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

    # Release the EPANET toolkit project (and its temp folder) that the
    # fast solver held open during the run.  Without this the handle can
    # linger in memory until GC, and starting a new experiment then
    # leaves two EPANET projects open simultaneously.
    try:
        solver = ctx["problem"].get("_solver")
        if solver:
            solver.close()
    except Exception:
        pass

    # New experiment — invalidate run-specific UI state
    _clear_run_ui_state()

    # Append this run to history (auto-named unless user provided name)
    run_name = str(job.context.get("run_name", "")).strip()
    try:
        _append_to_history(result, elapsed, cancelled, run_name)
        st.session_state._clear_run_name = True
    except Exception as e:
        print(f"[finalize_run] history append failed: {e}")

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
                "de_f", "de_cr", "de_strategy",
                "sade_fmin", "sade_fmax", "sade_crmin", "sade_crmax",
                "sade_conv", "sade_tol", "sade_stag"):
        if key in cfg and cfg[key] is not None:
            st.session_state[key] = cfg[key]

    units = prob["Units"]
    # Value + enabled flags, with backward-compat defaults for snapshots
    # written before Pmax/Vmin existed (Pmin and Vmax were always on).
    pmin_si = env.get("pmin_si", env.get("pmin"))
    pmax_si = env.get("pmax_si", 100.0)
    vmin_si = env.get("vmin_si", 0.0)
    vmax_si = env.get("vmax_si", env.get("vmax"))
    use_pmin = bool(env.get("use_pmin", True))
    use_pmax = bool(env.get("use_pmax", False))
    use_vmin = bool(env.get("use_vmin", False))
    use_vmax = bool(env.get("use_vmax", True))

    pmin_disp = pressure_from_si(pmin_si, units)
    pmax_disp = pressure_from_si(pmax_si, units)
    vmin_disp = velocity_from_si(vmin_si, units)
    vmax_disp = velocity_from_si(vmax_si, units)
    st.session_state.pmin = pmin_disp
    st.session_state.pmax = pmax_disp
    st.session_state.vmin = vmin_disp
    st.session_state.vmax = vmax_disp
    st.session_state.use_pmin = use_pmin
    st.session_state.use_pmax = use_pmax
    st.session_state.use_vmin = use_vmin
    st.session_state.use_vmax = use_vmax

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
        "pmin_input":        float(pmin_disp),
        "pmax_input":        float(pmax_disp),
        "vmin_input":        float(vmin_disp),
        "vmax_input":        float(vmax_disp),
        "use_pmin_input":    use_pmin,
        "use_pmax_input":    use_pmax,
        "use_vmin_input":    use_vmin,
        "use_vmax_input":    use_vmax,
        "fixed_pipes_input": fixed_str,
    }

    widget_updates["constraint_policy_input"] = str(
        cfg.get("constraint_policy") or "deb_prefers"
    )

    # Derive algo_input from the loaded experiment's algorithms.
    saved = set(data.get("experiment", {}).get("algorithms", []))
    widget_updates["algo_input"] = (
        [a for a in list_algorithms() if a in saved] or ["GA"]
    )
    # Algorithm params from META.  For each registered algorithm, for
    # each declared param: if the snapshot has a value for it, use it;
    # else use the META default.  Snap sade_tol to its allowed options.
    for entry in ALGORITHM_REGISTRY.values():
        for p in entry["meta"]["params"]:
            widget_key = p["key"] + "_input"
            canonical_key = p["key"]
            raw = cfg.get(canonical_key)
            if raw is None:
                value = p["default"]
            else:
                value = raw
            if p.get("widget") == "select_slider" and p.get("options"):
                value = _nearest_tol(float(value))
            widget_updates[widget_key] = value

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
        "Pmax": prob.get("Pmax", 100.0),
        "Vmin": prob.get("Vmin", 0.0),
        "Vmax": prob["Vmax"],
        "use_pmin": prob.get("use_pmin", True),
        "use_pmax": prob.get("use_pmax", False),
        "use_vmin": prob.get("use_vmin", False),
        "use_vmax": prob.get("use_vmax", True),
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

    # Snapshot replaces the experiment — drop per-run UI state, then
    # restore the snapshot's saved UI preferences (if any). Older
    # snapshots have no "ui" key and leave the current preferences
    # untouched.
    _clear_run_ui_state()
    saved_ui = data.get("ui", {}) or {}
    sel = saved_ui.get("selected_run")
    if isinstance(sel, (list, tuple)) and len(sel) == 2:
        st.session_state.selected_run = (str(sel[0]), int(sel[1]))
    if "only_feasible" in saved_ui:
        st.session_state.only_feasible = bool(saved_ui["only_feasible"])

    st.rerun()

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
    ca = ALGO_COLORS_A
    cb = ALGO_COLORS
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

    _render_stat_compare(exp_a, exp_b)


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

def _clear_run_ui_state():
    """Drop per-run UI state whenever the experiment is replaced.

    ``selected_run`` (the current dropdown selection), the on-demand
    hydraulics cache and every per-widget ``_sync_marker_*`` must all
    reset together — otherwise a new experiment would briefly render
    with data from the previous one.
    """
    st.session_state.pop("selected_run", None)
    st.session_state.pop("_run_hydraulics_cache", None)
    for k in [k for k in st.session_state.keys()
              if (k.startswith("_sync_marker_")
                  or k.startswith("_sync_of_"))]:
        st.session_state.pop(k, None)
    # NOTE: ``only_feasible`` is a user-level UI preference, not
    # experiment state — it is intentionally left intact so the filter
    # survives Clear Results / new experiments.

def _best_cost_among_feasible(experiment) -> float:
    """Lowest cost among feasible, successful/partial runs (+inf if none)."""
    best = float("inf")
    for runs in experiment.runs.values():
        for r in runs:
            if (r.status in ("Success", "Partial")
                    and r.feasible
                    and np.isfinite(r.score)):
                best = min(best, r.score)
    return best


def _is_best_cost(score: float, best_cost: float, rtol: float = 1e-6) -> bool:
    """True if ``score`` matches ``best_cost`` within relative tolerance.

    A small tolerance is used because two runs that converged to the same
    design can differ by floating-point noise; visually they are the same
    solution and should both be flagged.
    """
    if not (np.isfinite(score) and np.isfinite(best_cost)):
        return False
    if best_cost == 0.0:
        return score == 0.0
    return abs(score - best_cost) <= rtol * abs(best_cost)


def _format_run_option(algo: str, run, is_best: bool) -> str:
    """Dropdown label: ``⭐ GA / run 3 — 38,642,000  (✓ Success)``."""
    star = "⭐ " if is_best else "   "
    feas = "✓" if run.feasible else "✗"
    cost = f"{run.score:,.0f}" if np.isfinite(run.score) else "—"
    return f"{star}{algo} / run {run.run_index} — {cost}  ({feas} {run.status})"


def _get_run_hydraulics(algo: str, run_index: int):
    """Compute (or fetch from cache) the hydraulics of one run.

    Cache is stored in ``st.session_state["_run_hydraulics_cache"]`` and
    invalidated in ``_finalize_run`` / ``_restore_from_snapshot`` so a new
    experiment never returns stale pressures.  The diameter vector is
    reconstructed with the same round-trip as ``_finalize_run`` so the
    numbers agree exactly with what the "best" path used.
    """
    cache = st.session_state.setdefault("_run_hydraulics_cache", {})
    key = (algo, int(run_index))
    if key in cache:
        return cache[key]

    experiment = st.session_state.experiment
    snap = st.session_state.problem_snapshot
    run = experiment.runs[algo][run_index - 1]

    full_D = snap["initial_D"].copy()
    var_idx = _position_to_indices(run.position, {
        "D": snap["D_m"], "VariablePipes": snap["VariablePipes"],
    })
    full_D[snap["VariablePipes"]] = snap["D_m"][var_idx]

    wn = _load_network(st.session_state.inp_path)
    Pj, Vp, ok = run_hydraulic_simulation(
        wn, full_D, snap["pipe_names"], snap["junction_names"]
    )
    cache[key] = (Pj, Vp, ok)
    return Pj, Vp, ok

def _constraints_from_snapshot(snap):
    """Extract the four constraint entries from a problem_snapshot.

    Falls back to the pre-Pmax/Vmin defaults (Pmin on, Vmax on, others
    off) so old snapshots render correctly.
    """
    return {
        "Pmin": {"enabled": bool(snap.get("use_pmin", True)),
                 "value": float(snap["Pmin"])},
        "Pmax": {"enabled": bool(snap.get("use_pmax", False)),
                 "value": float(snap.get("Pmax", 100.0))},
        "Vmin": {"enabled": bool(snap.get("use_vmin", False)),
                 "value": float(snap.get("Vmin", 0.0))},
        "Vmax": {"enabled": bool(snap.get("use_vmax", True)),
                 "value": float(snap["Vmax"])},
    }

def _render_run_selector(widget_key: str):
    """Render the run-selector dropdown, feasibility filter and reset
    button, keeping both tabs in sync.

    ``selected_run`` and ``only_feasible`` are plain session_state keys
    (single source of truth).  Per-tab widget keys
    (``widget_key`` for the dropdown, ``only_feasible_<widget_key>`` for
    the checkbox) are kept in step with those canonical values through
    per-widget sync-markers.  This lets two tabs each own their dropdown
    without DuplicateWidgetID clashes, while the user sees the same
    selection everywhere.

    Returns ``(algo, run_index, run_record, best_cost)`` or
    ``(None, None, None, None)`` when there is nothing to inspect.
    """
    experiment = st.session_state.experiment
    if experiment is None:
        return None, None, None, None

    all_runs = [
        (a, r)
        for a in experiment.algorithms
        for r in experiment.runs.get(a, [])
        if r.status in ("Success", "Partial")
    ]
    if not all_runs:
        return None, None, None, None

    best_cost = _best_cost_among_feasible(experiment)

    # ---- "Only feasible" state (canonical + per-tab widget) ----
    of_widget_key = f"only_feasible_{widget_key}"
    of_marker = f"_sync_of_{widget_key}"
    if "only_feasible" not in st.session_state:
        st.session_state["only_feasible"] = False

    def _on_of_change():
        st.session_state["only_feasible"] = bool(
            st.session_state.get(of_widget_key, False)
        )

    n_feasible = sum(1 for _, r in all_runs if r.feasible)
    if n_feasible == 0:
        # No feasible runs — force the filter off and hide the checkbox.
        if st.session_state["only_feasible"]:
            st.session_state["only_feasible"] = False
            st.session_state[of_marker] = False
        st.caption(
            "No feasible runs in this experiment — showing all runs."
        )
    elif n_feasible < len(all_runs):
        # Push the canonical value into this tab's widget when the
        # *other* tab (or a snapshot load) changed it.
        if st.session_state.get(of_marker) != st.session_state["only_feasible"]:
            st.session_state[of_widget_key] = st.session_state["only_feasible"]
            st.session_state[of_marker] = st.session_state["only_feasible"]
        st.checkbox(
            "Only feasible", key=of_widget_key, on_change=_on_of_change,
        )

    # ---- Filter and validate the selection ----
    only_feasible = st.session_state["only_feasible"]
    shown = (
        [(a, r) for a, r in all_runs if r.feasible]
        if only_feasible else all_runs
    )
    if not shown:                    # defensive fallback, should not fire
        shown = all_runs
    valid_keys = {(a, r.run_index) for a, r in shown}

    best_key = (
        st.session_state.get("best_algo"),
        st.session_state.get("best_run_index"),
    )
    reset_note = None
    if st.session_state.get("selected_run") not in valid_keys:
        preferred = (
            best_key if best_key in valid_keys
            else (shown[0][0], shown[0][1].run_index)
        )
        prev = st.session_state.get("selected_run")
        st.session_state.selected_run = preferred
        if prev is not None and only_feasible:
            reset_note = (
                "Selection reset to the best feasible run "
                "(the previous selection is infeasible)."
            )

    # ---- Sync the dropdown widget with the canonical selection ----
    sel_marker = f"_sync_marker_{widget_key}"
    if st.session_state.get(sel_marker) != st.session_state.selected_run:
        st.session_state[widget_key] = st.session_state.selected_run
        st.session_state[sel_marker] = st.session_state.selected_run

    def _on_sel_change():
        st.session_state.selected_run = st.session_state[widget_key]

    # ---- Dropdown + Reset button in one row ----
    c_sel, c_btn = st.columns([6, 1])
    with c_sel:
        st.selectbox(
            "Run",
            options=[(a, r.run_index) for a, r in shown],
            format_func=lambda k: _format_run_option(
                k[0], experiment.runs[k[0]][k[1] - 1],
                is_best=_is_best_cost(
                    experiment.runs[k[0]][k[1] - 1].score, best_cost),
            ),
            key=widget_key,
            on_change=_on_sel_change,
            label_visibility="collapsed",
        )
    with c_btn:
        at_best = (
            st.session_state.get("selected_run") == best_key
            and best_key in valid_keys
        )
        st.button(
            "↩ Reset",
            key=f"reset_best_{widget_key}",
            disabled=(at_best or best_key not in valid_keys),
            help=(
                "Jump back to the canonical best run of the experiment. "
                "When several runs tie on cost, this returns to the one "
                "picked by the summary — all tied-best runs remain "
                "⭐-marked in the dropdown."
            ),
            on_click=lambda: st.session_state.__setitem__(
                "selected_run", best_key),
        )

    st.caption(
        "⭐ marks runs that match the global best cost."
        + (f"  {reset_note}" if reset_note else "")
    )

    algo, run_index = st.session_state.selected_run
    return algo, run_index, experiment.runs[algo][run_index - 1], best_cost


def _render_run_summary(algo, run_index, run, best_cost, units,
                        Pj_si, Vp_si, conv_ok):
    """Summary metric cards + convergence banner (shared by both tabs).

    Four compact metric cards in one row plus a full-width banner.  This
    replaces the previous five-column layout, which clipped the
    Hydraulics card on standard screen widths.
    """
    p_unit = units["PressureUnit"]
    v_unit = units["VelocityUnit"]
    Pj = np.array([pressure_from_si(p, units) for p in Pj_si])
    Vp = np.array([velocity_from_si(v, units) for v in Vp_si])

    is_star = _is_best_cost(run.score, best_cost)
    title = f"Summary — {algo} / run {run_index}"
    if is_star:
        title += "   ⭐ best-known"
    st.subheader(title)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Cost",
            f"{run.score:,.0f}" if np.isfinite(run.score) else "—",
        )
    with c2:
        st.metric(
            "Feasibility",
            "✓ Feasible" if run.feasible else "✗ Infeasible",
        )
    with c3:
        st.metric(f"Min pressure ({p_unit})", f"{Pj.min():.2f}")
    with c4:
        st.metric(f"Max velocity ({v_unit})", f"{Vp.max():.2f}")

    if conv_ok:
        st.success("✓ Hydraulics converged")
    else:
        st.warning("⚠ Hydraulics did not converge")

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
            "NFE per run": (
                f"{s['nfe_mean']:,.0f}" if np.isfinite(s["nfe_mean"])
                else "—"
            ),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ---- Convergence curves ----
    st.subheader("Convergence")
    fig = go.Figure()
    colors = ALGO_COLORS
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


def _stats_table(df, index_col):
    """Show a wide stats table with metrics as rows (easier to read)."""
    st.dataframe(df.set_index(index_col).T, width="stretch")


def _render_statistics_tab():
    st.subheader("Statistical analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Best-known (reference) cost — optional",
            min_value=0.0, value=0.0, step=1000.0, format="%.2f",
            key="ref_cost_input",
            help="Literature or best-known optimum for this network. "
                 "Enables gap-to-reference and success-rate statistics. "
                 "0 = not used.",
        )
    with c2:
        st.number_input(
            "Success tolerance (% above reference)",
            min_value=0.0, max_value=100.0, value=1.0, step=0.5,
            key="ref_tol_input",
            help="A run counts as a success if it is feasible and its cost "
                 "is within this percentage of the reference cost.",
        )

    result: ExperimentResult | None = st.session_state.experiment
    if result is None:
        st.info("Run an experiment to see statistics here.")
        return

    ref = float(st.session_state.get("ref_cost_input") or 0.0) or None
    tol = float(st.session_state.get("ref_tol_input", 1.0))
    analysis = st_.analyze_experiment(result, ref, tol)

    for note in analysis["notes"]:
        st.warning(note)

    st.markdown("### Descriptive statistics")
    _stats_table(st_.descriptive_dataframe(analysis), "Algorithm")
    st.caption(
        "Cost statistics use finished, feasible runs only. "
        "CI = 95% confidence interval; CV = coefficient of variation; "
        "Shapiro p < 0.05 suggests non-normal costs, so rank-based tests "
        "are used below."
    )

    if analysis["pairwise"]:
        st.markdown("### Comparison between algorithms")
        om = analysis.get("omnibus")
        if om is not None:
            if np.isfinite(om["p"]):
                st.markdown(
                    f"Kruskal-Wallis across all algorithms: "
                    f"**p = {st_.fmt_p(om['p'])}**"
                    + (f" (H = {om['H']:.2f})" if np.isfinite(om["H"]) else "")
                    + (" → at least one algorithm differs."
                       if om["p"] < analysis["alpha"]
                       else " → no significant difference overall.")
                )
            else:
                st.caption(f"Kruskal-Wallis: {om['note']}")
        _stats_table(st_.tests_dataframe(analysis["pairwise"]), "Comparison")
        for r in analysis["pairwise"]:
            st.markdown(f"**{r['verdict']}**")
        for f in analysis["feasibility"]:
            st.caption(
                f"Feasibility rate — {f['x']}: {f['rate_x']:.0f}%, "
                f"{f['y']}: {f['rate_y']:.0f}% "
                f"(Fisher exact p = {st_.fmt_p(f['p'])})."
            )

    # ---- ECDF of final costs ----
    st.markdown("### Empirical distribution of final costs")
    colors = ALGO_COLORS
    fig = go.Figure()
    for algo, runs in result.runs.items():
        c = np.sort(st_.feasible_costs(runs))
        if c.size:
            fig.add_trace(go.Scatter(
                x=c, y=np.arange(1, c.size + 1) / c.size, mode="lines",
                line=dict(shape="hv", color=colors.get(algo, "gray"),
                          width=2),
                name=algo,
            ))
    if ref:
        fig.add_vline(x=ref, line_dash="dot", line_color="gray",
                      annotation_text="reference")
    fig.update_layout(
        height=330, xaxis_title="Final cost (feasible runs)",
        yaxis_title="Fraction of runs ≤ cost",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("A curve further LEFT / above means cheaper results.")

    sade_runs = [r for r in result.runs.get("SADE", [])
                 if r.status in ("Success", "Partial")
                 and (getattr(r, "extra", None) or {}).get("cv")]
    if sade_runs:
        with st.expander("SADE adaptation diagnostics"):
            idx = st.selectbox(
                "Run", range(len(sade_runs)),
                format_func=lambda i: f"run {sade_runs[i].run_index} "
                                      f"(cost {sade_runs[i].score:,.0f})",
                key="sade_diag_run",
            )
            ex = sade_runs[idx].extra
            gens = np.arange(1, len(ex["cv"]) + 1)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=gens, y=ex["cv"], name="Cv",
                                      line=dict(color="#0b7285")))
            fig2.update_layout(height=260, yaxis_type="log",
                               xaxis_title="Generation",
                               yaxis_title="Cv of population cost",
                               margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig2, width="stretch")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=gens, y=ex["f_mean"],
                                      name="mean F"))
            fig3.add_trace(go.Scatter(x=gens, y=ex["cr_mean"],
                                      name="mean CR"))
            fig3.update_layout(height=260, xaxis_title="Generation",
                               yaxis_title="Population mean",
                               margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig3, width="stretch")
            st.caption(
                "Cv = std/|mean| of the population's cost (the paper's "
                "convergence measure; the run stops when it falls below "
                "the tolerance). Mean F and CR show how the self-adaptation "
                "moves inside its ranges."
            )

    with st.expander("Methods and references (for your paper)"):
        st.write(st_.methods_text(analysis["alpha"]))
        for r in st_.REFERENCES:
            st.markdown(f"- {r}")
        st.caption("Please verify the references before citing.")


def _render_stat_compare(exp_a, exp_b):
    """Statistical comparison of two saved experiments (Compare tab)."""
    st.markdown("### Statistical comparison (A vs B)")
    common = sorted(set(exp_a.runs) & set(exp_b.runs))
    if not common:
        return
    rows = [
        st_.compare_samples(
            st_.feasible_costs(exp_a.runs[a]),
            st_.feasible_costs(exp_b.runs[a]), f"A/{a}", f"B/{a}")
        for a in common
    ]
    st_.adjust_and_explain(rows)
    _stats_table(st_.tests_dataframe(rows), "Comparison")
    for r in rows:
        st.markdown(f"**{r['verdict']}**")
    for a in common:
        f = st_.compare_feasibility(exp_a.runs[a], exp_b.runs[a])
        st.caption(
            f"{a} feasibility — A: {f['rate_x']:.0f}%, B: "
            f"{f['rate_y']:.0f}% (Fisher exact p = {st_.fmt_p(f['p'])})."
        )
    st.caption(
        "Mann-Whitney U on feasible final costs; A12 = probability that "
        "an A run is cheaper than a B run (ties count half). p-values are "
        "Holm-adjusted across algorithms."
    )

def _render_design_hydraulics_tab():
    """Inspect a single run: pipe diameters, velocities, node pressures.

    Defaults to the overall best run; the dropdown at the top lets the
    user inspect any other run.  ⭐ markers flag runs that match the
    global best cost.
    """
    if st.session_state.experiment is None:
        st.info("Run an experiment first.")
        return

    snap = st.session_state.problem_snapshot
    if snap is None:
        st.warning("Problem snapshot not available — re-run the experiment.")
        return

    units = snap["Units"]
    pipe_names = snap["pipe_names"]
    junc_names = snap["junction_names"]
    fixed = set(int(x) for x in snap["FixedPipes"])
    p_unit = units["PressureUnit"]
    v_unit = units["VelocityUnit"]
    L_unit = units["LengthUnit"]
    L_scale = 1.0 if L_unit == "m" else 1.0 / 0.3048
    c = _constraints_from_snapshot(snap)
    pmin = pressure_from_si(c["Pmin"]["value"], units) if c["Pmin"]["enabled"] else None
    pmax = pressure_from_si(c["Pmax"]["value"], units) if c["Pmax"]["enabled"] else None
    vmin = velocity_from_si(c["Vmin"]["value"], units) if c["Vmin"]["enabled"] else None
    vmax = velocity_from_si(c["Vmax"]["value"], units) if c["Vmax"]["enabled"] else None

    st.subheader("Select a run")
    algo, run_index, run, best_cost = _render_run_selector(
        "run_inspector_selector"
    )
    if algo is None:
        st.info("No completed runs to inspect.")
        return

    try:
        Pj_si, Vp_si, conv_ok = _get_run_hydraulics(algo, run_index)
    except Exception as e:
        st.error(f"Could not compute hydraulics: {type(e).__name__}: {e}")
        return
    if Pj_si is None or Vp_si is None:
        st.warning("Hydraulic results not available for this run.")
        return

    Pj = np.array([pressure_from_si(p, units) for p in Pj_si])
    Vp = np.array([velocity_from_si(v, units) for v in Vp_si])

    _render_run_summary(algo, run_index, run, best_cost, units,
                        Pj_si, Vp_si, conv_ok)

    # ---- Pipe design (diameters + velocities in one table) ----
    st.subheader("Pipe design")
    pos = run.position
    rows = []
    for i, name in enumerate(pipe_names):
        L = snap["L"][i]
        d_m = pos[i] if pos is not None else snap["initial_D"][i]
        v_disp = Vp[i] if i < len(Vp) else float("nan")
        v_status = "OK"
        if c["Vmin"]["enabled"] and Vp_si[i] < c["Vmin"]["value"]:
            v_status = f"< Vmin ({vmin:.2f})"
        elif c["Vmax"]["enabled"] and Vp_si[i] > c["Vmax"]["value"]:
            v_status = f"> Vmax ({vmax:.2f})"
        rows.append({
            "Pipe ID": i + 1,
            "Name": name,
            f"Length ({L_unit})": f"{L / L_scale:.0f}",
            "Diameter (mm)": f"{d_m * 1000:.1f}",
            "Diameter (in)": f"{d_m / 0.0254:.2f}",
            f"Velocity ({v_unit})": f"{v_disp:.3f}",
            "Velocity check": v_status,
            "Status": "FIXED" if i in fixed else "variable",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ---- Node pressures (table + chart) ----
    st.subheader("Node pressures")
    rows = []
    for k, name in enumerate(junc_names):
        status = "OK"
        if c["Pmin"]["enabled"] and Pj_si[k] < c["Pmin"]["value"]:
            status = f"< Pmin ({pmin:.2f} {p_unit})"
        elif c["Pmax"]["enabled"] and Pj_si[k] > c["Pmax"]["value"]:
            status = f"> Pmax ({pmax:.2f} {p_unit})"
        rows.append({
            "Node ID": k + 1,
            "Name": name,
            f"Pressure ({p_unit})": f"{Pj[k]:.3f}",
            "Status": status,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    fig = go.Figure()
    def _p_viol(k):
        return ((c["Pmin"]["enabled"] and Pj_si[k] < c["Pmin"]["value"])
                or (c["Pmax"]["enabled"] and Pj_si[k] > c["Pmax"]["value"]))
    fig.add_trace(go.Bar(
        x=junc_names, y=Pj,
        marker_color=["#e74c3c" if _p_viol(i) else "#2ecc71"
                      for i in range(len(Pj))],
        text=[f"{p:.2f}" for p in Pj], textposition="outside",
    ))
    if pmin is not None:
        fig.add_hline(y=pmin, line_dash="dash", line_color="red",
                      annotation_text=f"Pmin = {pmin:.2f} {p_unit}")
    if pmax is not None:
        fig.add_hline(y=pmax, line_dash="dash", line_color="red",
                      annotation_text=f"Pmax = {pmax:.2f} {p_unit}")
    fig.update_layout(height=320, yaxis_title=f"Pressure ({p_unit})",
                      margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, width="stretch")

    # ---- Pipe velocities (chart; exact values in the pipe table above) ----
    st.subheader("Pipe velocities")
    fig = go.Figure()
    def _v_viol(i):
        return ((c["Vmin"]["enabled"] and Vp_si[i] < c["Vmin"]["value"])
                or (c["Vmax"]["enabled"] and Vp_si[i] > c["Vmax"]["value"]))
    fig.add_trace(go.Bar(
        x=pipe_names, y=Vp,
        marker_color=["#e74c3c" if _v_viol(i) else "#3498db"
                      for i in range(len(Vp))],
        text=[f"{v:.2f}" for v in Vp], textposition="outside",
    ))
    if vmin is not None:
        fig.add_hline(y=vmin, line_dash="dash", line_color="red",
                      annotation_text=f"Vmin = {vmin:.2f} {v_unit}")
    if vmax is not None:
        fig.add_hline(y=vmax, line_dash="dash", line_color="red",
                      annotation_text=f"Vmax = {vmax:.2f} {v_unit}")
    fig.update_layout(
        height=320, yaxis_title=f"Velocity ({v_unit})",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        f"Min pressure: {Pj.min():.2f} {p_unit} | "
        f"Max velocity: {Vp.max():.2f} {v_unit} | "
        + ("✓ Converged" if conv_ok else "⚠ Not converged")
    )

def _render_map_tab():
    if st.session_state.inp_path is None:
        st.info("Load an .inp file first.")
        return

    experiment = st.session_state.experiment
    snap = st.session_state.problem_snapshot

    # Default values used in preview mode (no experiment yet)
    pos = None
    Pj_si = None
    Vp_si = None

    if experiment is not None and snap is not None:
        st.subheader("Select a run to display")
        algo, run_index, run, best_cost = _render_run_selector(
            "map_run_selector"
        )
        if algo is None:
            st.info("No completed runs to display.")
            return

        try:
            Pj_si, Vp_si, conv_ok = _get_run_hydraulics(algo, run_index)
        except Exception as e:
            st.error(
                f"Could not compute hydraulics: {type(e).__name__}: {e}"
            )
            return
        if Pj_si is None or Vp_si is None:
            st.warning("Hydraulic results not available for this run.")
            return

        _render_run_summary(algo, run_index, run, best_cost,
                            snap["Units"], Pj_si, Vp_si, conv_ok)
        pos = run.position
        st.divider()

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
    units = st.session_state.get("network_units") or {}

    # Compute coloring data (in display units) for the selected run
    pressures = None
    velocities = None
    diameters = None
    pmin = vmax = None
    fixed_1based = []

    if experiment is not None and snap is not None and pos is not None:
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
        fixed_1based = [int(x) + 1 for x in snap["FixedPipes"]]

    # Pass all four constraints to the map (None = disabled, so the
    # figure omits its colouring influence and its hover label).
    c = _constraints_from_snapshot(snap) if snap is not None else None
    pmin_for_map = (pressure_from_si(c["Pmin"]["value"], units)
                    if c and c["Pmin"]["enabled"] else None)
    pmax_for_map = (pressure_from_si(c["Pmax"]["value"], units)
                    if c and c["Pmax"]["enabled"] else None)
    vmin_for_map = (velocity_from_si(c["Vmin"]["value"], units)
                    if c and c["Vmin"]["enabled"] else None)
    vmax_for_map = (velocity_from_si(c["Vmax"]["value"], units)
                    if c and c["Vmax"]["enabled"] else None)
    fig = build_network_figure(
        wn, mode=mode,
        pressures=pressures, velocities=velocities, diameters=diameters,
        pmin=pmin_for_map, pmax=pmax_for_map,
        vmin=vmin_for_map, vmax=vmax_for_map,
        fixed_pipe_indices=fixed_1based,
        show_labels=show_labels,
        units=units,
    )
    st.plotly_chart(fig, width="stretch")

    if experiment is None:
        st.caption(
            "Preview mode: network topology only. "
            "Run an experiment to see velocity/pressure/diameter coloring."
        )


def _render_all_runs_tab():
    if st.session_state.experiment is None:
        st.info("Run an experiment first.")
        return

    result = st.session_state.experiment
    best_cost = _best_cost_among_feasible(result)
    rows = []
    for algo, runs in result.runs.items():
        for r in runs:
            rows.append({
                "Algorithm": algo,
                "Best": (
                    "⭐" if _is_best_cost(r.score, best_cost) else ""
                ),
                "Run": r.run_index,
                "Seed": r.seed,
                "Status": r.status,
                "Generations run": (getattr(r, "extra", {}) or {}).get(
                    "generations"),
                "Stopped by Cv": (getattr(r, "extra", {}) or {}).get(
                    "converged"),
                "NFE": getattr(r, "nfe", 0) or None,
                "Simulations": getattr(r, "n_sim", 0) or None,
                "NFE to best": getattr(r, "nfe_to_best", 0) or None,
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
                    "NFE": getattr(r, "nfe", 0),
                    "Hydraulic simulations": getattr(r, "n_sim", 0),
                    "NFE to best": getattr(r, "nfe_to_best", 0),
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
    _discover_plugins_if_needed()

    actions = _render_sidebar()

    if actions["run_clicked"]:
        try:
            _start_run(actions["algo_label"])
        except Exception as e:
            st.error(f"Could not start experiment: {type(e).__name__}: {e}")
            st.exception(e)

    # Sensitivity dialog (modal).  Shown only when the sidebar
    # "Open sensitivity…" button set the flag.
    if st.session_state.get("sensitivity_dialog_open"):
        _sensitivity_dialog()

    _show_job_notice()
    if st.session_state.get("job") is not None:
        _job_panel()

    tabs = st.tabs([
        "📊 Summary",
        "📈 Statistics",
        "🎯 Sensitivity",
        "🔎 Run Inspector",
        "🗺 Network Map",
        "📋 All Runs",
        "🔄 Compare",
    ])
    with tabs[0]:
        _render_summary_tab()
    with tabs[1]:
        _render_statistics_tab()
    with tabs[2]:
        _render_sensitivity_tab()
    with tabs[3]:
        _render_design_hydraulics_tab()
    with tabs[4]:
        _render_map_tab()
    with tabs[5]:
        _render_all_runs_tab()
    with tabs[6]:
        _render_compare_tab()


if __name__ == "__main__":
    main()