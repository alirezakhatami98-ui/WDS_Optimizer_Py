"""Reproducibility report: Markdown + figures + data → ZIP."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import wntr

from .figures import (
    make_convergence_png,
    make_cost_distribution_png,
    make_network_map_png,
    make_pressure_png,
    make_velocity_png,
)
from ..analysis import stats as st_
from ..experiments.runner import summarize_experiment
from ..persistence.snapshot import build_snapshot
from ..hydraulics.units import (
    diameter_to_display, pressure_from_si, velocity_from_si,
)


def build_report_zip(
    *,
    inp_path, d_path, cost_path,
    problem_snapshot, experiment,
    best_algo, best_run_index,
    best_position_m, node_pressures, pipe_velocities,
    hydraulics_converged,
    config,
    include_inputs=True,
    reference_cost=None,
    success_tol_pct=1.0,
) -> bytes:
    """Build a complete reproducibility ZIP in memory."""
    buf = io.BytesIO()

    # ---- Load network for map ----
    try:
        wn = wntr.network.WaterNetworkModel(str(inp_path))
    except Exception as e:
        print(f"[build_report_zip] WNTR load error: {e}")
        wn = None

    pipe_names = list(problem_snapshot["pipe_names"])
    units = problem_snapshot["Units"]
    p_unit = units.get("PressureUnit", "m")
    v_unit = units.get("VelocityUnit", "m/s")
    d_unit = units.get("DiameterUnit", "mm")
    junction_names = list(problem_snapshot["junction_names"])
    pmin = problem_snapshot["Pmin"]
    vmax = problem_snapshot["Vmax"]
    fixed_pipes = [int(i) for i in problem_snapshot["FixedPipes"]]

    # ---- Figures ----
    figs: dict[str, bytes] = {}

    def _try(name, fn, *args, **kw):
        try:
            figs[name] = fn(*args, **kw)
        except Exception as e:
            print(f"[build_report_zip] figure {name} error: {e}")

    _try("convergence.png", make_convergence_png, experiment)
    _try("cost_distribution.png", make_cost_distribution_png, experiment)
    # Constraint values (SI) + enabled flags; pass None for disabled
    # constraints so figures omit the line and violation colouring.
    c = {
        "Pmin": {"enabled": bool(problem_snapshot.get("use_pmin", True)),
                 "value": pmin},
        "Pmax": {"enabled": bool(problem_snapshot.get("use_pmax", False)),
                 "value": float(problem_snapshot.get("Pmax", 100.0))},
        "Vmin": {"enabled": bool(problem_snapshot.get("use_vmin", False)),
                 "value": float(problem_snapshot.get("Vmin", 0.0))},
        "Vmax": {"enabled": bool(problem_snapshot.get("use_vmax", True)),
                 "value": vmax},
    }
    pmin_fig = c["Pmin"]["value"] if c["Pmin"]["enabled"] else None
    pmax_fig = c["Pmax"]["value"] if c["Pmax"]["enabled"] else None
    vmin_fig = c["Vmin"]["value"] if c["Vmin"]["enabled"] else None
    vmax_fig = c["Vmax"]["value"] if c["Vmax"]["enabled"] else None

    _try("node_pressures.png", make_pressure_png,
         junction_names, node_pressures, pmin_fig, units,
         pmax_si=pmax_fig)
    _try("pipe_velocities.png", make_velocity_png,
         pipe_names, pipe_velocities, vmax_fig, units,
         vmin_si=vmin_fig)
    if wn is not None:
        for mode in ("velocity", "diameter", "pressure"):
            # Pass all four constraints (None = disabled).  The figure
            # mirrors the app's interactive map: single-constraint
            # colouring preserves the previous behaviour; two-sided
            # uses a green=OK / red=violation severity scale.
            _try(f"network_map_{mode}.png", make_network_map_png,
                 wn, pipe_names, junction_names, best_position_m,
                 pipe_velocities, node_pressures,
                 mode=mode, fixed_pipe_indices=fixed_pipes,
                 pmin=pmin_fig, pmax=pmax_fig,
                 vmin=vmin_fig, vmax=vmax_fig, units=units)

    # ---- Snapshot ----
    try:
        inp_content = Path(inp_path).read_text(encoding="utf-8",
                                                errors="replace")
        d_content = Path(d_path).read_text(encoding="utf-8",
                                            errors="replace")
        cost_content = Path(cost_path).read_text(encoding="utf-8",
                                                  errors="replace")
        snapshot = build_snapshot(
            problem_snapshot=problem_snapshot, experiment=experiment,
            best_algo=best_algo, best_run_index=best_run_index,
            best_position_m=best_position_m,
            node_pressures=node_pressures,
            pipe_velocities=pipe_velocities,
            hydraulics_converged=hydraulics_converged,
            config=config,
            inp_content=inp_content,
            d_content=d_content,
            cost_content=cost_content,
        )
        experiment_json = json.dumps(
            snapshot, indent=2, ensure_ascii=False
        ).encode("utf-8")
    except Exception as e:
        print(f"[build_report_zip] snapshot error: {e}")
        experiment_json = b"{}"

    # ---- Statistics ----
    try:
        analysis = st_.analyze_experiment(
            experiment, reference_cost, success_tol_pct)
        stats_desc_csv = _to_csv(
            st_.descriptive_dataframe(analysis, formatted=False)
            .to_dict("records"))
        stats_tests_csv = _to_csv(
            st_.tests_dataframe(analysis["pairwise"], formatted=False)
            .to_dict("records"))
    except Exception as e:
        print(f"[build_report_zip] statistics error: {e}")
        analysis, stats_desc_csv, stats_tests_csv = None, "", ""

    # ---- CSVs ----
    summary = summarize_experiment(experiment)

    summary_rows = []
    for algo, s in summary.items():
        summary_rows.append({
            "Algorithm": algo,
            "Runs": len(experiment.runs[algo]),
            "Success": s["n_success"],
            "Feasible": s["n_feasible"],
            "Mean": f"{s['mean']:.2f}" if np.isfinite(s["mean"]) else "",
            "Std": f"{s['std']:.2f}" if np.isfinite(s["std"]) else "",
            "Best": f"{s['best']:.2f}" if np.isfinite(s["best"]) else "",
            "NFE_mean": (f"{s['nfe_mean']:.0f}"
                         if np.isfinite(s["nfe_mean"]) else ""),
        })

    runs_rows = []
    for algo, runs in experiment.runs.items():
        for r in runs:
            runs_rows.append({
                "Algorithm": algo, "Run": r.run_index, "Seed": r.seed,
                "Status": r.status,
                "Generations_run": (getattr(r, "extra", {}) or {}).get(
                    "generations", ""),
                "Stopped_by_Cv": (getattr(r, "extra", {}) or {}).get(
                    "converged", ""),
                "Stopped_by_stagnation": (getattr(r, "extra", {}) or {}).get(
                    "stopped_by_stagnation", ""),
                "Final_Cv": (f"{r.extra['cv'][-1]:.3e}"
                             if (getattr(r, "extra", None) or {}).get("cv")
                             else ""),
                "NFE": getattr(r, "nfe", 0),
                "Hydraulic_simulations": getattr(r, "n_sim", 0),
                "NFE_to_best": getattr(r, "nfe_to_best", 0),
                "Restarts": len(getattr(r, "restart_gens", [])),
                "Restart_at_gen": " ".join(
                    str(g) for g in getattr(r, "restart_gens", [])
                ),
                "Epoch_best_costs": " | ".join(
                    f"{c:.2f}" if np.isfinite(c) else "none"
                    for c in getattr(r, "epoch_bests", [])
                ),
                "Cost": f"{r.score:.2f}" if np.isfinite(r.score) else "",
                "Feasible": "Yes" if r.feasible else "No",
                "Time_s": (f"{r.execution_time:.3f}"
                           if np.isfinite(r.execution_time) else ""),
                "Error": r.error_message,
            })

    pipes_rows = []
    for i, pname in enumerate(pipe_names):
        d_m = best_position_m[i]
        v_si = pipe_velocities[i]
        v_disp = velocity_from_si(v_si, units)
        # velocity column name: Velocity_m_s or Velocity_ft_s
        v_col = f"Velocity_{v_unit.replace('/', '_')}"
        pipes_rows.append({
            "Pipe_ID": i + 1, "Name": pname,
            f"Diameter_{d_unit}": f"{diameter_to_display(d_m, units):.3f}",
            "Diameter_mm": f"{d_m * 1000:.2f}",
            "Diameter_in": f"{d_m / 0.0254:.3f}",
            v_col: f"{v_disp:.4f}",
            "Fixed": "Yes" if i in fixed_pipes else "No",
        })

    nodes_rows = []
    for i, nname in enumerate(junction_names):
        p_si = node_pressures[i]
        p_disp = pressure_from_si(p_si, units)
        p_col = f"Pressure_{p_unit}"
        nodes_rows.append({
            "Node_ID": i + 1, "Name": nname,
            p_col: f"{p_disp:.4f}",
            "Status": "Violation" if p_si < pmin else "OK",
        })

    # ---- Markdown + README ----
    md = _build_markdown(
        experiment=experiment, summary=summary,
        best_algo=best_algo, best_run_index=best_run_index,
        best_position_m=best_position_m,
        node_pressures=node_pressures, pipe_velocities=pipe_velocities,
        hydraulics_converged=hydraulics_converged,
        problem_snapshot=problem_snapshot, config=config,
        pipe_names=pipe_names, analysis=analysis,
    )
    readme = _build_readme(experiment, best_algo, best_run_index)

    # ---- Pack ZIP ----
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("REPORT.md", md.encode("utf-8"))
        zf.writestr("README.txt", readme.encode("utf-8"))
        zf.writestr("experiment.json", experiment_json)

        if include_inputs:
            for src, dst in [
                (inp_path, "inputs/network.inp"),
                (d_path, "inputs/D.txt"),
                (cost_path, "inputs/Cost.txt"),
            ]:
                try:
                    zf.writestr(dst, Path(src).read_bytes())
                except Exception as e:
                    print(f"[build_report_zip] input copy error ({dst}): {e}")

        zf.writestr("data/summary.csv",
                    _to_csv(summary_rows).encode("utf-8-sig"))
        zf.writestr("data/runs.csv",
                    _to_csv(runs_rows).encode("utf-8-sig"))
        if stats_desc_csv:
            zf.writestr("data/statistics_descriptive.csv",
                        stats_desc_csv.encode("utf-8-sig"))
        if stats_tests_csv:
            zf.writestr("data/statistics_tests.csv",
                        stats_tests_csv.encode("utf-8-sig"))
        zf.writestr("data/pipes.csv",
                    _to_csv(pipes_rows).encode("utf-8-sig"))
        zf.writestr("data/nodes.csv",
                    _to_csv(nodes_rows).encode("utf-8-sig"))

        for name, data in figs.items():
            zf.writestr(f"figures/{name}", data)

    return buf.getvalue()


# =====================================================================
#  Internals
# =====================================================================

def _to_csv(rows) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _md_table(df) -> list:
    """DataFrame -> GitHub markdown table lines."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return out


def _stats_markdown(analysis) -> list:
    L = ["## Statistical analysis\n"]
    for n in analysis["notes"]:
        L.append(f"> **Note:** {n}\n")

    L.append("### Descriptive statistics\n")
    d = st_.descriptive_dataframe(analysis).set_index("Algorithm").T
    d.insert(0, "Metric", d.index)
    L.extend(_md_table(d))
    L.append("")
    if analysis.get("reference_cost"):
        L.append(f"Reference (best-known) cost: "
                 f"{analysis['reference_cost']:,.2f}; success = finished, "
                 f"feasible and within {analysis['tol_pct']:g}% of it.\n")

    if analysis["pairwise"]:
        L.append("### Comparison between algorithms\n")
        om = analysis.get("omnibus")
        if om is not None:
            if np.isfinite(om["p"]):
                L.append(f"Kruskal-Wallis across all algorithms: "
                         f"p = {st_.fmt_p(om['p'])}"
                         + (f" (H = {om['H']:.2f})"
                            if np.isfinite(om["H"]) else "") + ".\n")
            else:
                L.append(f"Kruskal-Wallis across all algorithms: "
                         f"{om['note']}\n")
        t = st_.tests_dataframe(analysis["pairwise"]).set_index(
            "Comparison").T
        t.insert(0, "Metric", t.index)
        L.extend(_md_table(t))
        L.append("")
        for r in analysis["pairwise"]:
            L.append(f"- {r['verdict']}")
        L.append("")
        for f in analysis["feasibility"]:
            L.append(
                f"- Feasibility rate {f['x']}: {f['rate_x']:.0f}%, "
                f"{f['y']}: {f['rate_y']:.0f}% "
                f"(Fisher exact p = {st_.fmt_p(f['p'])}).")
        L.append("")

    L.append("### Statistical methods\n")
    L.append(st_.methods_text(analysis["alpha"]) + "\n")
    L.append("References:\n")
    for ref in st_.REFERENCES:
        L.append(f"- {ref}")
    L.append("")
    return L


def _algorithm_param_lines(experiment, config) -> list:
    """One bullet per algorithm that was run, with its parameters."""
    def g(key, default="?"):
        v = config.get(key)
        return default if v is None else v

    out = []
    for algo in experiment.algorithms:
        if algo == "GA":
            out.append(f"- **GA parameters:** crossover Pc = {g('ga_pc')}, "
                       f"mutation Pm = {g('ga_pm')}, "
                       f"selection = {g('ga_sel')}")
        elif algo == "PSO":
            out.append(f"- **PSO parameters:** inertia w = {g('pso_w')}, "
                       f"c1 = {g('pso_c1')}, c2 = {g('pso_c2')}")
        elif algo == "DE":
            out.append(f"- **DE parameters:** F = {g('de_f')}, "
                       f"CR = {g('de_cr')}, strategy = {g('de_strategy')}")
        elif algo == "SADE":
            stop = ("stop when Cv < " + str(g('sade_tol'))
                    if g('sade_conv', True) else "fixed MaxGen")
            # StagnationMax (added later than the original fields).
            # Old experiment files that predate this field fall back to
            # "not recorded" rather than guessing 0 or 300.
            stag_raw = config.get('sade_stag')
            if stag_raw is None:
                stag_txt = ", stagnation safeguard: not recorded"
            else:
                stag = int(stag_raw)
                stag_txt = (f", stagnation safeguard: {stag} generations"
                            if stag > 0 else
                            ", stagnation safeguard: disabled")
            out.append(f"- **SADE parameters:** F in [{g('sade_fmin')}, "
                       f"{g('sade_fmax')}], CR in [{g('sade_crmin')}, "
                       f"{g('sade_crmax')}] (self-adapted per individual), "
                       f"termination: {stop}{stag_txt}")
    return out


def _restart_text(experiment) -> str:
    rs = getattr(experiment, "restart", None) or {}
    if not rs.get("enabled"):
        return "disabled"
    mode = rs.get("mode", "random")     # files written before modes: random
    if mode == "perturb":
        how = (f"perturb best design "
               f"({100 * float(rs.get('perturb_rate', 0.3)):.0f}% of pipes "
               f"re-drawn)")
    else:
        how = ("random population, best kept: "
               + ("yes" if rs.get("keep_best") else "no"))
    return (f"enabled (patience {rs.get('patience')} generations, "
            f"max {rs.get('max_restarts')} restarts per run, {how}; "
            f"total MaxGen unchanged)")


def _build_markdown(*, experiment, summary, best_algo, best_run_index,
                    best_position_m, node_pressures, pipe_velocities,
                    hydraulics_converged, problem_snapshot, config,
                    pipe_names, analysis=None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_run = experiment.runs[best_algo][best_run_index - 1]
    pmin = problem_snapshot["Pmin"]
    vmax = problem_snapshot["Vmax"]
    D = problem_snapshot["D_m"]
    units = problem_snapshot["Units"]
    p_unit = units.get("PressureUnit", "m")
    v_unit = units.get("VelocityUnit", "m/s")
    d_unit = units.get("DiameterUnit", "mm")
    pmin_disp = pressure_from_si(pmin, units)
    vmax_disp = velocity_from_si(vmax, units)

    L = []
    L.append("# WDS Optimizer — Reproducibility Report\n")
    L.append(f"**Generated:** {ts}  ")
    L.append(f"**Experiment ID:** `{experiment.experiment_id}`\n")
    if getattr(experiment, "cancelled", False):
        L.append("> **⚠ This experiment was cancelled by the user — "
                 "results are partial.** Statistics below use only runs "
                 "that finished (status `Success`). A run stopped midway "
                 "(status `Partial`) keeps its best-so-far design and may "
                 "be the best design reported. See `data/runs.csv` for "
                 "per-run status.\n")

    L.append("## Configuration\n")
    L.append(f"- **Network:** {len(pipe_names)} pipes, "
             f"{len(problem_snapshot['junction_names'])} junctions")
    L.append(f"- **Variable pipes:** {len(problem_snapshot['VariablePipes'])}")
    L.append(f"- **Fixed pipes (0-based):** "
             f"{list(map(int, problem_snapshot['FixedPipes']))}")
    L.append(f"- **Population / Swarm (NS):** {config.get('ns')}")
    L.append(f"- **Max generations / iterations:** {config.get('max_gen')}")
    L.append(f"- **Runs per algorithm:** {experiment.number_of_runs}")
    L.append(f"- **Experiment seed:** {experiment.experiment_seed}")
    L.append(f"- **Algorithms:** {', '.join(experiment.algorithms)}")
    L.append(f"- **Restart on stagnation:** {_restart_text(experiment)}")
    for line in _algorithm_param_lines(experiment, config):
        L.append(line)
    pmin_disp = pressure_from_si(
        problem_snapshot["Pmin"], units)
    pmax_disp = pressure_from_si(
        problem_snapshot.get("Pmax", 100.0), units)
    vmin_disp = velocity_from_si(
        problem_snapshot.get("Vmin", 0.0), units)
    vmax_disp = velocity_from_si(
        problem_snapshot["Vmax"], units)
    for key, label, disp, unit in [
        ("use_pmin", "Pmin", pmin_disp, p_unit),
        ("use_pmax", "Pmax", pmax_disp, p_unit),
        ("use_vmin", "Vmin", vmin_disp, v_unit),
        ("use_vmax", "Vmax", vmax_disp, v_unit),
    ]:
        on = bool(problem_snapshot.get(key, key in ("use_pmin", "use_vmax")))
        state = "on" if on else "off"
        L.append(f"- **{label}:** {disp:.2f} {unit} ({state})")
    L.append(f"- **Diameter options:** {len(D)} "
             f"({D[0]*1000:.1f} – {D[-1]*1000:.1f} mm)\n")

    L.append("## Experiment summary\n")
    L.append("| Algorithm | Runs | Success | Feasible | Mean cost | "
             "Std | Best | NFE per run |")
    L.append("|---|---|---|---|---|---|---|---|")
    for algo, s in summary.items():
        nfe = f"{s['nfe_mean']:,.0f}" if np.isfinite(s["nfe_mean"]) else "—"
        L.append(f"| {algo} | {len(experiment.runs[algo])} | "
                 f"{s['n_success']} | {s['n_feasible']} | "
                 f"{s['mean']:.0f} | {s['std']:.0f} | {s['best']:.0f} | "
                 f"{nfe} |")
    L.append("")
    L.append("> **Note on `Stopped_by_Cv` / `Stopped_by_stagnation` "
             "(data/runs.csv):** SADE terminates either when the "
             "population's coefficient of variation falls below the "
             "tolerance (`Stopped_by_Cv = True`) or when the epoch's "
             "best cost has not improved for `StagnationMax` generations "
             "(`Stopped_by_stagnation = True`). Other algorithms "
             "terminate by MaxGen and report empty cells here.\n")

    if analysis is not None:
        L.extend(_stats_markdown(analysis))

    L.append("## Best design\n")
    L.append(f"- **Algorithm:** {best_algo}")
    L.append(f"- **Run:** {best_run_index}")
    L.append(f"- **Run status:** {best_run.status}")
    L.append(f"- **Seed:** {best_run.seed}")
    L.append(f"- **Cost:** {best_run.score:.2f}")
    L.append(f"- **Feasible:** {'Yes' if best_run.feasible else 'No'}")
    L.append(f"- **Execution time:** {best_run.execution_time:.2f} s")
    if getattr(best_run, "nfe", 0) > 0:
        L.append(f"- **NFE:** {best_run.nfe:,} "
                 f"(hydraulic simulations: {best_run.n_sim:,}; "
                 f"best found at NFE {best_run.nfe_to_best:,})")
    L.append(f"- **Hydraulics converged:** "
             f"{'Yes' if hydraulics_converged else 'No'}\n")

    L.append(f"### Pipe diameters and velocities\n")
    L.append(f"| Pipe | Name | Diameter ({d_unit}) | "
             f"Diameter (mm) | Velocity ({v_unit}) |")
    L.append("|---|---|---|---|---|")
    for i, pname in enumerate(pipe_names):
        d_m = best_position_m[i]
        v_disp = velocity_from_si(pipe_velocities[i], units)
        L.append(f"| {i+1} | {pname} | "
                 f"{diameter_to_display(d_m, units):.2f} | "
                 f"{d_m*1000:.1f} | "
                 f"{v_disp:.3f} |")
    L.append("")

    L.append(f"### Node pressures\n")
    L.append(f"| Node | Name | Pressure ({p_unit}) | Status |")
    L.append("|---|---|---|---|")
    use_pmin = bool(problem_snapshot.get("use_pmin", True))
    use_pmax = bool(problem_snapshot.get("use_pmax", False))
    pmin_si_val = problem_snapshot["Pmin"]
    pmax_si_val = problem_snapshot.get("Pmax", 100.0)
    for i, nname in enumerate(problem_snapshot["junction_names"]):
        p_si = node_pressures[i]
        p_disp = pressure_from_si(p_si, units)
        if use_pmin and p_si < pmin_si_val:
            status = "Violation (< Pmin)"
        elif use_pmax and p_si > pmax_si_val:
            status = "Violation (> Pmax)"
        else:
            status = "OK"
        L.append(f"| {i+1} | {nname} | {p_disp:.3f} | {status} |")
    L.append("")

    L.append("## Figures\n")
    for title, path in [
        ("Convergence", "convergence.png"),
        ("Cost distribution", "cost_distribution.png"),
        ("Node pressures", "node_pressures.png"),
        ("Pipe velocities", "pipe_velocities.png"),
        ("Network map — velocity", "network_map_velocity.png"),
        ("Network map — diameter", "network_map_diameter.png"),
        ("Network map — pressure", "network_map_pressure.png"),
    ]:
        L.append(f"### {title}\n")
        L.append(f"![{title}](figures/{path})\n")

    L.append("## Reproduction\n")
    L.append("1. Load `experiment.json` in the Streamlit app "
             "(sidebar → *5. Save/Load*).")
    L.append("2. Or run `run_experiment()` from Python with the same "
             "`experiment_seed`. Per-run seeds are in `data/runs.csv`.")
    L.append("3. `.inp` and input files are embedded in "
             "`experiment.json` and also present under `inputs/`.\n")
    L.append("See `README.txt` for step-by-step instructions.\n")

    return "\n".join(L)


def _build_readme(experiment, best_algo, best_run_index) -> str:
    return f"""\
WDS Optimizer — Reproducibility Report
======================================

This archive contains everything needed to reproduce the experiment
and verify the results.

Contents
--------
  REPORT.md           Complete human-readable summary (Markdown).
  README.txt          This file.
  experiment.json     Full experiment state (loadable in the app).
  inputs/             Input files: network.inp, D.txt, Cost.txt.
  data/               Result tables (CSV, UTF-8-BOM → Excel-friendly):
                        summary.csv    Per-algorithm statistics
                        runs.csv       All individual runs
                        statistics_descriptive.csv   Per-algorithm statistics
                        statistics_tests.csv         Mann-Whitney / A12 tests
                        pipes.csv      Best-design pipe results
                        nodes.csv      Best-design node results
  figures/            Static PNG figures (150 dpi).

{"NOTE: this experiment was CANCELLED by the user; results are partial." + chr(10) if getattr(experiment, "cancelled", False) else ""}Configuration
-------------
  Algorithms        : {', '.join(experiment.algorithms)}
  Restart           : {_restart_text(experiment)}
  Runs per algo     : {experiment.number_of_runs}
  Experiment seed   : {experiment.experiment_seed}
  NS                : {experiment.NS}
  MaxGen            : {experiment.MaxGen}
  Pmin              : {experiment.Pmin} m    [{'on' if experiment.use_pmin else 'off'}]
  Pmax              : {experiment.Pmax} m    [{'on' if experiment.use_pmax else 'off'}]
  Vmin              : {experiment.Vmin} m/s  [{'on' if experiment.use_vmin else 'off'}]
  Vmax              : {experiment.Vmax} m/s  [{'on' if experiment.use_vmax else 'off'}]

Best result
-----------
  Algorithm         : {best_algo}
  Run index         : {best_run_index}

Reproduction — Option 1 (app)
-----------------------------
  1. Start the app: `streamlit run src/wds_optimizer/app/streamlit_app.py`
  2. Sidebar → "5. Save / Load" → upload `experiment.json` → Load.
  3. Full state restored (inputs, parameters, results, hydraulics).

Reproduction — Option 2 (Python)
--------------------------------
  1. Copy `inputs/*` into your project folder.
  2. Import modules from `wds_optimizer.*`.
  3. Run with `experiment_seed = {experiment.experiment_seed}`.
     Per-run seeds are in `data/runs.csv` (column "Seed").

Notes
-----
  * Determinism: same seed + same inputs yields the same cost sequence
    (within floating-point tolerance).
  * MATLAB vs Python: the RNG differs (MATLAB: Mersenne Twister global;
    Python: PCG64 per call). Results are statistically equivalent but
    not bit-for-bit identical.
"""