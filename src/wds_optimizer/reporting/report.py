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
    _try("node_pressures.png", make_pressure_png,
         junction_names, node_pressures, pmin, units)
    _try("pipe_velocities.png", make_velocity_png,
         pipe_names, pipe_velocities, vmax, units)
    if wn is not None:
        for mode in ("velocity", "diameter", "pressure"):
            _try(f"network_map_{mode}.png", make_network_map_png,
                 wn, pipe_names, junction_names, best_position_m,
                 pipe_velocities, node_pressures,
                 mode=mode, fixed_pipe_indices=fixed_pipes,
                 pmin=pmin, vmax=vmax, units=units)

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
        })

    runs_rows = []
    for algo, runs in experiment.runs.items():
        for r in runs:
            runs_rows.append({
                "Algorithm": algo, "Run": r.run_index, "Seed": r.seed,
                "Status": r.status,
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
        pipe_names=pipe_names,
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


def _build_markdown(*, experiment, summary, best_algo, best_run_index,
                    best_position_m, node_pressures, pipe_velocities,
                    hydraulics_converged, problem_snapshot, config,
                    pipe_names) -> str:
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
    L.append(f"- **Pmin:** {pmin_disp:.2f} {p_unit} "
             f"({pmin:.3f} m)")
    L.append(f"- **Vmax:** {vmax_disp:.2f} {v_unit} "
             f"({vmax:.3f} m/s)")
    L.append(f"- **Diameter options:** {len(D)} "
             f"({D[0]*1000:.1f} – {D[-1]*1000:.1f} mm)\n")

    L.append("## Experiment summary\n")
    L.append("| Algorithm | Runs | Success | Feasible | Mean cost | "
             "Std | Best |")
    L.append("|---|---|---|---|---|---|---|")
    for algo, s in summary.items():
        L.append(f"| {algo} | {len(experiment.runs[algo])} | "
                 f"{s['n_success']} | {s['n_feasible']} | "
                 f"{s['mean']:.0f} | {s['std']:.0f} | {s['best']:.0f} |")
    L.append("")

    L.append("## Best design\n")
    L.append(f"- **Algorithm:** {best_algo}")
    L.append(f"- **Run:** {best_run_index}")
    L.append(f"- **Seed:** {best_run.seed}")
    L.append(f"- **Cost:** {best_run.score:.2f}")
    L.append(f"- **Feasible:** {'Yes' if best_run.feasible else 'No'}")
    L.append(f"- **Execution time:** {best_run.execution_time:.2f} s")
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
    for i, nname in enumerate(problem_snapshot["junction_names"]):
        p_si = node_pressures[i]
        p_disp = pressure_from_si(p_si, units)
        status = "Violation" if p_si < pmin else "OK"
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
                        pipes.csv      Best-design pipe results
                        nodes.csv      Best-design node results
  figures/            Static PNG figures (150 dpi).

Configuration
-------------
  Algorithms        : {', '.join(experiment.algorithms)}
  Runs per algo     : {experiment.number_of_runs}
  Experiment seed   : {experiment.experiment_seed}
  NS                : {experiment.NS}
  MaxGen            : {experiment.MaxGen}
  Pmin              : {experiment.Pmin} m
  Vmax              : {experiment.Vmax} m/s

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