"""WDS Optimizer — Streamlit application (Phase 2)."""

from __future__ import annotations

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
from wds_optimizer.optimization.problem import build_optimization_problem
from wds_optimizer.app.network_map import build_network_figure
from wds_optimizer.app.state import init_state, reset_experiment

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


# =====================================================================
#  UI helpers
# =====================================================================

def _header():
    st.title("💧 WDS Optimizer")
    st.caption(
        "Water distribution network optimization using GA and PSO, "
        "powered by WNTR/EPANET. Python port of the MATLAB toolkit."
    )


def _render_sidebar():
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
        st.number_input(
            "Population / Swarm size (NS)", min_value=4, max_value=500,
            step=1, key="ns_input",
            value=int(st.session_state.get("ns", 50)),
        )
        st.number_input(
            "Max generations / iterations", min_value=5, max_value=5000,
            step=1, key="max_gen_input",
            value=int(st.session_state.get("max_gen", 100)),
        )
        st.number_input(
            "Min pressure (m)", min_value=0.0, step=1.0,
            key="pmin_input",
            value=float(st.session_state.get("pmin", 30.0)),
        )
        st.number_input(
            "Max velocity (m/s)", min_value=0.1, step=0.1,
            key="vmax_input",
            value=float(st.session_state.get("vmax", 2.5)),
        )
        st.text_input(
            "Fixed pipe IDs (e.g. '1, 3')",
            key="fixed_pipes_input",
            value=st.session_state.get("fixed_pipes", ""),
        )

        st.divider()
        st.header("3. Algorithm")
        algo = st.radio(
            "Algorithm", ["GA", "PSO", "Both"],
            index=0, key="algo_input", horizontal=True,
        )
        if algo in ("GA", "Both"):
            with st.expander("GA parameters", expanded=True):
                st.number_input(
                    "Pc (crossover)", 0.0, 1.0, step=0.05,
                    key="ga_pc_input",
                    value=float(st.session_state.get("ga_pc", 0.8)),
                )
                st.number_input(
                    "Pm (mutation)", 0.0, 1.0, step=0.01,
                    key="ga_pm_input",
                    value=float(st.session_state.get("ga_pm", 0.03)),
                )
                st.selectbox(
                    "Selection",
                    ["tournament", "roulette"], index=0,
                    key="ga_sel_input",
                )
        if algo in ("PSO", "Both"):
            with st.expander("PSO parameters", expanded=True):
                st.number_input(
                    "w (inertia)", 0.0, 2.0, step=0.05,
                    key="pso_w_input",
                    value=float(st.session_state.get("pso_w", 0.7)),
                )
                st.number_input(
                    "c1 (cognitive)", 0.0, 3.0, step=0.1,
                    key="pso_c1_input",
                    value=float(st.session_state.get("pso_c1", 1.5)),
                )
                st.number_input(
                    "c2 (social)", 0.0, 3.0, step=0.1,
                    key="pso_c2_input",
                    value=float(st.session_state.get("pso_c2", 1.5)),
                )

        st.divider()
        st.header("4. Experiment")
        st.number_input(
            "Runs per algorithm", min_value=1, max_value=50, step=1,
            key="n_runs_input",
            value=int(st.session_state.get("n_runs", 3)),
        )
        st.number_input(
            "Experiment seed", min_value=0, max_value=2**31 - 1, step=1,
            key="exp_seed_input",
            value=int(st.session_state.get("exp_seed", 1)),
        )

        st.divider()
        can_run = bool(st.session_state.inp_path)
        run_clicked = st.button(
            "🚀 Run Experiment",
            type="primary",
            width="stretch",
            disabled=not can_run,
        )
        if not can_run:
            st.caption("Load all three input files to enable.")
        if st.session_state.experiment is not None:
            if st.button("🗑 Clear results", width="stretch"):
                reset_experiment()
                st.rerun()

    return run_clicked, algo


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

    problem = build_optimization_problem(
        wn=wn, Din=Din, Cost=Cost, D_m=D_m,
        full_lengths=lengths, initial_D=initial_D,
        fixed_pipes=fixed_0based,
        variable_pipes=variable_0based,
        Pmin=st.session_state.pmin,
        Vmax=st.session_state.vmax,
        pipe_names=pipe_names,
        junction_names=junction_names,
        units=units,
    )
    return problem, units, wn


# =====================================================================
#  Run experiment with live progress
# =====================================================================

def _execute_run(algo_label: str):
    # --- Sync sidebar widgets into canonical session_state keys. ---
    # Only sync widgets that were actually rendered.  When the user
    # picks "GA" only, the PSO expander is not rendered and its
    # `*_input` keys do not exist in session_state.
    def _sync(dst, src):
        if src in st.session_state:
            st.session_state[dst] = st.session_state[src]

    _sync("ns",          "ns_input")
    _sync("max_gen",     "max_gen_input")
    _sync("pmin",        "pmin_input")
    _sync("vmax",        "vmax_input")
    _sync("fixed_pipes", "fixed_pipes_input")
    _sync("n_runs",      "n_runs_input")
    _sync("exp_seed",    "exp_seed_input")
    # GA widgets
    _sync("ga_pc",       "ga_pc_input")
    _sync("ga_pm",       "ga_pm_input")
    _sync("ga_sel",      "ga_sel_input")
    # PSO widgets
    _sync("pso_w",       "pso_w_input")
    _sync("pso_c1",      "pso_c1_input")
    _sync("pso_c2",      "pso_c2_input")

    with st.status("Preparing problem...", expanded=True) as status:
        problem, units, wn = _prepare_problem()

        st.write(
            f"Network: **{len(problem['pipe_names'])}** pipes, "
            f"**{len(problem['junction_names'])}** junctions, "
            f"**{len(problem['VariablePipes'])}** variable, "
            f"**{len(problem['FixedPipes'])}** fixed."
        )
        st.write(
            f"Diameters: **{len(problem['D'])}** options "
            f"({problem['D'][0] * 1000:.1f} – "
            f"{problem['D'][-1] * 1000:.1f} mm)"
        )
        st.write(f"Units: {units['FlowUnit']} / {units['System']}")

        # Algorithm config — use CANONICAL keys after sync.
        base_cfg = build_algorithm_config(
            int(st.session_state.ns), int(st.session_state.max_gen)
        )
        base_cfg.GA.Pc        = st.session_state.ga_pc
        base_cfg.GA.Pm        = st.session_state.ga_pm
        base_cfg.GA.Selection = st.session_state.ga_sel
        base_cfg.PSO.w        = st.session_state.pso_w
        base_cfg.PSO.c1       = st.session_state.pso_c1
        base_cfg.PSO.c2       = st.session_state.pso_c2

        if algo_label == "GA":
            algos = ["GA"]
        elif algo_label == "PSO":
            algos = ["PSO"]
        else:
            algos = ["GA", "PSO"]

        st.write(
            f"Algorithms: **{', '.join(algos)}** × "
            f"**{st.session_state.n_runs}** runs, "
            f"NS={st.session_state.ns}, "
            f"MaxGen={st.session_state.max_gen}"
        )
        status.update(label="Running...", state="running")

    # ---- Live progress widgets ----
    st.subheader("Live progress")
    progress_bar = st.progress(0.0)
    status_line = st.empty()
    chart_placeholder = st.empty()

    def progress_cb(info):
        gen = info["Generation"]
        maxg = info["MaxGenerations"]
        progress_bar.progress(min(1.0, gen / maxg))
        feas_txt = "Feasible" if info["Feasible"] else "Infeasible"
        status_line.info(
            f"**{info['RunLabel']}** | Gen **{gen}/{maxg}** | "
            f"Best cost: **{info['BestCost']:,.0f}** | {feas_txt}"
        )
        conv = info["Convergence"]
        color = "#d9480f" if info["Algorithm"] == "GA" else "#1971c2"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(conv) + 1), y=conv,
            mode="lines", line=dict(color=color, width=2),
            name=info["RunLabel"],
        ))
        fig.update_layout(
            height=280, margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Generation / Iteration",
            yaxis_title="Best cost",
            showlegend=False,
        )
        chart_placeholder.plotly_chart(fig, width="stretch")

    t0 = time.perf_counter()
    result: ExperimentResult = run_experiment(
        problem, base_cfg,
        number_of_runs=int(st.session_state.n_runs),
        algorithms=algos,
        experiment_seed=int(st.session_state.exp_seed),
        progress_callback=progress_cb,
    )
    elapsed = time.perf_counter() - t0
    progress_bar.progress(1.0)
    status_line.success(f"Experiment finished in {elapsed:.1f} s")

    # ---- Pick best run and compute final hydraulics ----
    best_record, best_algo = _pick_best_run(result)
    if best_record is None:
        st.error("No successful run. Check the All Runs tab for details.")
        return

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

    st.rerun()


def _pick_best_run(result: ExperimentResult):
    """Select the best successful run (feasible first, then cost)."""
    best_record = None
    best_algo = None
    for algo, runs in result.runs.items():
        for r in runs:
            if r.status != "Success":
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

    # Build table
    rows = []
    for i, name in enumerate(pipe_names):
        L = snap["L"][i]
        d_m = pos[i] if pos is not None else snap["initial_D"][i]
        d_mm = d_m * 1000
        d_in = d_m / 0.0254
        v = vel[i] if vel is not None and i < len(vel) else float("nan")
        rows.append({
            "Pipe ID": i + 1,
            "Name": name,
            "Length (m)": f"{L:.0f}",
            "Diameter (mm)": f"{d_mm:.1f}",
            "Diameter (in)": f"{d_in:.2f}",
            "Velocity (m/s)": f"{v:.3f}",
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
    Pj = st.session_state.node_pressures
    Vp = st.session_state.pipe_velocities
    pmin = snap["Pmin"]
    vmax = snap["Vmax"]
    junc_names = snap["junction_names"]

    if Pj is None or Vp is None:
        st.warning("Hydraulic results not available.")
        return

    # ---- Node pressures ----
    st.subheader("Node pressures")
    rows = []
    for k, name in enumerate(junc_names):
        p = Pj[k]
        status = "Violation ✓" if p < pmin else "OK"
        rows.append({
            "Node ID": k + 1,
            "Name": name,
            "Pressure (m)": f"{p:.3f}",
            "Status": status,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch",
                 hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=junc_names, y=Pj,
        marker_color=["#e74c3c" if p < pmin else "#2ecc71" for p in Pj],
        text=[f"{p:.2f}" for p in Pj], textposition="outside",
    ))
    fig.add_hline(y=pmin, line_dash="dash", line_color="red",
                  annotation_text=f"Pmin = {pmin} m")
    fig.update_layout(height=320, yaxis_title="Pressure (m)",
                      margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, width="stretch")

    # ---- Pipe velocities ----
    st.subheader("Pipe velocities")
    pipe_names = snap["pipe_names"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pipe_names, y=Vp,
        marker_color=["#e74c3c" if v > vmax else "#3498db" for v in Vp],
        text=[f"{v:.2f}" for v in Vp], textposition="outside",
    ))
    fig.add_hline(y=vmax, line_dash="dash", line_color="red",
                  annotation_text=f"Vmax = {vmax} m/s")
    fig.update_layout(height=320, yaxis_title="Velocity (m/s)",
                      margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, width="stretch")

    # ---- Status line ----
    conv_txt = (
        "✓ Converged" if st.session_state.hydraulics_converged
        else "⚠ Not converged"
    )
    st.caption(
        f"Min pressure: {Pj.min():.2f} m | "
        f"Max velocity: {Vp.max():.2f} m/s | {conv_txt}"
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

    # Compute coloring data
    pressures = None
    velocities = None
    diameters = None
    pmin = vmax = None
    fixed_1based = []

    if st.session_state.experiment is not None:
        snap = st.session_state.problem_snapshot
        Pj = st.session_state.node_pressures
        Vp = st.session_state.pipe_velocities
        pos = st.session_state.best_position_m

        pressures = dict(zip(snap["junction_names"], Pj))
        velocities = dict(zip(snap["pipe_names"], Vp))
        diameters = dict(zip(snap["pipe_names"],
                             [pos[i] for i in range(len(snap["pipe_names"]))]))
        pmin = snap["Pmin"]
        vmax = snap["Vmax"]
        fixed_1based = [int(x) + 1 for x in snap["FixedPipes"]]

    fig = build_network_figure(
        wn, mode=mode,
        pressures=pressures, velocities=velocities, diameters=diameters,
        pmin=pmin, vmax=vmax,
        fixed_pipe_indices=fixed_1based,
        show_labels=show_labels,
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
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download runs (CSV)",
            data=csv,
            file_name=f"wds_runs_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
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

    run_clicked, algo_label = _render_sidebar()

    if run_clicked:
        try:
            _execute_run(algo_label)
        except Exception as e:
            st.error(f"Experiment failed: {type(e).__name__}: {e}")
            st.exception(e)

    tabs = st.tabs([
        "📊 Summary",
        "🏗 Best Design",
        "💧 Hydraulics",
        "🗺 Network Map",
        "📋 All Runs",
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


if __name__ == "__main__":
    main()