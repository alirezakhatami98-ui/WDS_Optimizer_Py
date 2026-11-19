"""Sensitivity analysis pipeline for WDS Optimizer.

Runs a predefined set of parameter sweeps on the benchmarks and
writes the results as JSON plus a generated Markdown report.

Usage
-----
    # Run the quick plan (twoloop only, ~5 min)
    python scripts/sensitivity_analysis.py --quick

    # Run the full plan (hanoi + twoloop, 10 runs, 500 gen, ~2-3 h)
    python scripts/sensitivity_analysis.py --full

    # One specific sweep
    python scripts/sensitivity_analysis.py \\
        --benchmark hanoi --algo GA --param GA.ga_pc \\
        --values 0.6,0.7,0.8,0.9 --runs 5 --max-gen 300

    # Inspect the plan without running
    python scripts/sensitivity_analysis.py --quick --dry-run

    # Rebuild docs/sensitivity_analysis.md from existing JSONs
    python scripts/sensitivity_analysis.py --regenerate-only

Design principles
-----------------
- No new backend.  Uses wds_optimizer.sweep.runner.run_parameter_sweep
  (v0.6.0), which already handles all parameter classes (global,
  constraint, algorithm, plugin).
- Reproducible.  Fixed seed by default (42).  Each JSON records the
  parameter, values, n_runs, and timestamp.
- Idempotent.  Markdown is regenerated from JSONs; re-running the
  script overwrites cleanly.
- Testable.  main(argv) returns an int; can be invoked in-process.
- Not for the GUI.  This is a maintenance/authoring tool.  Users run
  sweeps interactively via the GUI's Sweep tab; sensitivity analysis
  is what *we* use to write the parameter recommendations that ship
  with the tool.

Output layout
-------------
    docs/sensitivity_data/
        hanoi/
            GA/
                GA_ga_pc.json
                GA_ga_pm.json
            PSO/
                ...
        twoloop/
            ...
    docs/sensitivity_analysis.md  (generated)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------
#  Benchmark definitions (mirrors benchmarks/<name>/README.md)
# ---------------------------------------------------------------------

BENCHMARK_CONFIGS: dict[str, dict] = {
    "hanoi": {
        "inp": "benchmarks/hanoi/hanoi.inp",
        "d": "benchmarks/hanoi/D.txt",
        "cost": "benchmarks/hanoi/Cost.txt",
        "diam_unit": "in",
        "constraints": {
            "Pmin": {"enabled": True,  "value": 30.0},
            "Pmax": {"enabled": False, "value": 100.0},
            "Vmin": {"enabled": False, "value": 0.0},
            "Vmax": {"enabled": False, "value": 2.5},
        },
        "ns": 200,
        "max_gen": 300,
        "paper_best": 6.081e6,
    },
    "twoloop": {
        "inp": "benchmarks/twoloop/twoloop.inp",
        "d": "benchmarks/twoloop/D.txt",
        "cost": "benchmarks/twoloop/Cost.txt",
        "diam_unit": "in",
        "constraints": {
            "Pmin": {"enabled": True,  "value": 30.0},
            "Pmax": {"enabled": False, "value": 100.0},
            "Vmin": {"enabled": False, "value": 0.0},
            "Vmax": {"enabled": False, "value": 2.5},
        },
        "ns": 100,
        "max_gen": 200,
        "paper_best": 419_000.0,
    },
}


# ---------------------------------------------------------------------
#  Predefined plans
# ---------------------------------------------------------------------

# (benchmark, algorithm, parameter, values)
PLAN_QUICK: list[tuple] = [
    ("twoloop", "GA",   "GA.ga_pc",        [0.6, 0.7, 0.8, 0.9]),
    ("twoloop", "GA",   "GA.ga_pm",        [0.01, 0.03, 0.05, 0.1]),
    ("twoloop", "GA",   "GA.ga_sel",       ["tournament", "roulette"]),
    ("twoloop", "PSO",  "PSO.pso_w",       [0.5, 0.7, 0.9]),
    ("twoloop", "PSO",  "PSO.pso_c1",      [1.0, 1.5, 2.0]),
    ("twoloop", "DE",   "DE.de_f",         [0.3, 0.5, 0.7]),
    ("twoloop", "DE",   "DE.de_cr",        [0.3, 0.5, 0.9]),
    ("twoloop", "SADE", "SADE.sade_stag",  [100, 300, 500]),
]

PLAN_FULL: list[tuple] = [
    # Hanoi — larger network, longer runs
    ("hanoi", "GA",   "GA.ga_pc",          [0.5, 0.6, 0.7, 0.8, 0.9]),
    ("hanoi", "GA",   "GA.ga_pm",          [0.01, 0.03, 0.05, 0.1]),
    ("hanoi", "GA",   "GA.ga_sel",         ["tournament", "roulette"]),
    ("hanoi", "PSO",  "PSO.pso_w",         [0.5, 0.7, 0.9]),
    ("hanoi", "PSO",  "PSO.pso_c1",        [1.0, 1.5, 2.0]),
    ("hanoi", "PSO",  "PSO.pso_c2",        [1.0, 1.5, 2.0]),
    ("hanoi", "DE",   "DE.de_f",           [0.3, 0.5, 0.7]),
    ("hanoi", "DE",   "DE.de_cr",          [0.3, 0.5, 0.7, 0.9]),
    ("hanoi", "DE",   "DE.de_strategy",
        ["rand/1/bin", "best/1/bin", "current-to-best/1/bin"]),
    ("hanoi", "SADE", "SADE.sade_fmin",    [0.05, 0.1, 0.2]),
    ("hanoi", "SADE", "SADE.sade_crmin",   [0.05, 0.1, 0.2]),
    ("hanoi", "SADE", "SADE.sade_tol",     [1e-5, 1e-6, 1e-7]),
    ("hanoi", "SADE", "SADE.sade_stag",    [100, 300, 500]),
    # Twoloop — repeat for comparison
    ("twoloop", "GA",   "GA.ga_pc",        [0.5, 0.6, 0.7, 0.8, 0.9]),
    ("twoloop", "SADE", "SADE.sade_stag",  [100, 300, 500]),
    # Global parameters
    ("hanoi", "GA",   "NS",     [100, 200, 300]),
    ("hanoi", "GA",   "MaxGen", [200, 300, 500]),
    ("hanoi", "SADE", "n_runs", [1, 5, 10]),
]


# ---------------------------------------------------------------------
#  Value formatting / JSON conversion
# ---------------------------------------------------------------------

def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _value_to_jsonable(v: Any) -> Any:
    """int, float, str, bool are all JSON-serializable as-is."""
    if isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def _parse_values_arg(raw: str) -> list:
    """Reuse the sweep parser (handles quotes, int/float/string)."""
    from wds_optimizer.sweep.config import parse_sweep_values
    return parse_sweep_values(raw)


# ---------------------------------------------------------------------
#  Plan building
# ---------------------------------------------------------------------

def _build_plan(args) -> list[dict]:
    """Turn CLI args into a list of sweep dicts."""
    if args.param:
        if not args.benchmark or not args.algo or not args.values:
            raise ValueError(
                "--param requires --benchmark, --algo, and --values."
            )
        return [{
            "benchmark": args.benchmark,
            "algo": args.algo,
            "param": args.param,
            "values": _parse_values_arg(args.values),
            "runs": args.runs if args.runs is not None else 1,
            "max_gen": args.max_gen,   # None → BENCHMARK_CONFIGS default
        }]

    if args.quick and args.full:
        raise ValueError("Use --quick or --full, not both.")
    if not (args.quick or args.full):
        raise ValueError(
            "Specify --quick, --full, or --param/--benchmark/--algo/--values."
        )

    template = PLAN_QUICK if args.quick else PLAN_FULL
    default_runs = 3 if args.quick else 10
    default_max_gen = 200 if args.quick else 500

    runs = args.runs if args.runs is not None else default_runs
    max_gen = args.max_gen if args.max_gen is not None else default_max_gen

    return [
        {
            "benchmark": b, "algo": a, "param": p,
            "values": list(v), "runs": runs, "max_gen": max_gen,
        }
        for (b, a, p, v) in template
    ]


# ---------------------------------------------------------------------
#  Saving sweep results
# ---------------------------------------------------------------------

def _save_sweep_json(
    result, out_path: Path, benchmark: str, runs: int,
) -> None:
    """Write the aggregated sweep result as JSON (no ExperimentResult)."""
    data = {
        "benchmark": benchmark,
        "algorithm": (
            result.algorithms[0] if result.algorithms else "?"
        ),
        "parameter": result.parameter,
        "values": [_value_to_jsonable(v) for v in result.values],
        "n_runs": result.n_runs,
        "experiment_seed": result.experiment_seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "points": [
            {
                "value": _value_to_jsonable(p.value),
                "status": p.status,
                "best_score": (p.best_score
                               if np.isfinite(p.best_score) else None),
                "mean_score": (p.mean_score
                               if np.isfinite(p.mean_score) else None),
                "std_score": (p.std_score
                              if np.isfinite(p.std_score) else None),
                "feasible_pct": p.feasible_pct,
                "mean_time_s": p.mean_time,
                "error": p.error,
            }
            for p in result.points
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
#  Markdown generation
# ---------------------------------------------------------------------

def generate_markdown(output_dir: Path, md_path: Path) -> None:
    """Rebuild the sensitivity report from all JSONs in output_dir."""
    lines = [
        "# Sensitivity Analysis",
        "",
        "This file is auto-generated by `scripts/sensitivity_analysis.py`.",
        "Do not edit by hand — changes will be overwritten.",
        "",
        "Refresh with:",
        "",
        "```bash",
        "python scripts/sensitivity_analysis.py --regenerate-only",
        "```",
        "",
    ]

    if not output_dir.exists():
        lines.append("_No sensitivity data yet._")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return

    jsons = sorted(output_dir.rglob("*.json"))
    groups: dict[tuple, dict] = {}
    for jp in jsons:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            key = (data["benchmark"], data["algorithm"], data["parameter"])
            groups[key] = data
        except Exception as e:
            print(f"[sensitivity] skipping {jp}: {e}", file=sys.stderr)

    if not groups:
        lines.append("_No sensitivity data yet._")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return

    by_benchmark: dict[str, list[tuple]] = {}
    for (bench, algo, param), data in groups.items():
        by_benchmark.setdefault(bench, []).append((algo, param, data))

    for bench in sorted(by_benchmark):
        lines.append(f"\n## Benchmark: {bench}\n")
        bench_cfg = BENCHMARK_CONFIGS.get(bench, {})
        if "paper_best" in bench_cfg:
            lines.append(
                f"_Paper best cost: "
                f"${bench_cfg['paper_best']:,.0f}_\n"
            )
        by_algo: dict[str, list] = {}
        for algo, param, data in by_benchmark[bench]:
            by_algo.setdefault(algo, []).append((param, data))
        for algo in sorted(by_algo):
            lines.append(f"\n### {algo}\n")
            for param, data in sorted(by_algo[algo]):
                lines.extend(_param_section_md(param, data))

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _param_section_md(param: str, data: dict) -> list[str]:
    """One parameter's table + recommendation."""
    lines = [
        f"\n#### `{param}`\n",
        f"_n_runs = {data['n_runs']}, "
        f"seed = {data['experiment_seed']}, "
        f"generated {data['timestamp']}_\n",
        "| Value | Best | Mean | Std | Feasible % | Time (s/pt) |",
        "|-------|------|------|-----|------------|-------------|",
    ]
    best_mean: Optional[float] = None
    best_value: Any = None
    for p in data["points"]:
        v_str = _format_value(p["value"])
        best_str = (f"${p['best_score']:,.0f}"
                    if p["best_score"] is not None else "—")
        mean_str = (f"${p['mean_score']:,.0f}"
                    if p["mean_score"] is not None else "—")
        std_str = (f"${p['std_score']:,.0f}"
                   if p["std_score"] is not None else "—")
        feas_str = f"{p['feasible_pct']:.0f}%"
        time_str = f"{p['mean_time_s']:.1f}"
        lines.append(
            f"| {v_str} | {best_str} | {mean_str} | "
            f"{std_str} | {feas_str} | {time_str} |"
        )
        if p["mean_score"] is not None:
            if best_mean is None or p["mean_score"] < best_mean:
                best_mean = p["mean_score"]
                best_value = p["value"]

    if best_mean is not None:
        lines.append(
            f"\n**Recommended:** `{param} = {_format_value(best_value)}` "
            f"(lowest mean cost ${best_mean:,.0f})"
        )
    return lines


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sensitivity_analysis",
        description=(
            "Run sensitivity sweeps on benchmarks and generate "
            "docs/sensitivity_analysis.md.  "
            "Authoring tool, not for end users."
        ),
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true",
                   help="Run the predefined quick plan (twoloop only).")
    g.add_argument("--full", action="store_true",
                   help="Run the full plan (hanoi + twoloop, longer).")
    g.add_argument("--regenerate-only", action="store_true",
                   help="Skip execution; rebuild markdown from "
                        "existing JSONs.")

    p.add_argument("--benchmark", choices=sorted(BENCHMARK_CONFIGS),
                   help="For single-sweep mode.")
    p.add_argument("--algo", help="For single-sweep mode (e.g. GA).")
    p.add_argument("--param",
                   help="For single-sweep mode (e.g. GA.ga_pc).")
    p.add_argument("--values",
                   help="Comma-separated (e.g. 0.6,0.7,0.8).")

    p.add_argument("--runs", type=int, default=None,
                   help="Runs per point.  Default depends on plan.")
    p.add_argument("--max-gen", type=int, default=None,
                   help="Max generations per run.")
    p.add_argument("--seed", type=int, default=42,
                   help="Experiment seed (default 42).")

    p.add_argument("--output-dir", default="docs/sensitivity_data",
                   help="Where to write JSONs.")
    p.add_argument("--markdown", default="docs/sensitivity_analysis.md",
                   help="Where to write the markdown report.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit; do not run.")
    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 10


def _run(args) -> int:
    output_dir = Path(args.output_dir)
    md_path = Path(args.markdown)

    if args.regenerate_only:
        generate_markdown(output_dir, md_path)
        print(f"Regenerated {md_path}")
        return 0

    plan = _build_plan(args)

    if args.dry_run:
        print(f"Dry-run: {len(plan)} sweep(s) would run:")
        for s in plan:
            print(f"  {s['benchmark']:8s} / {s['algo']:4s} / "
                  f"{s['param']:20s}  values={s['values']}  "
                  f"runs={s['runs']}  max_gen={s['max_gen']}")
        return 0

    # Deferred imports: only when actually running.
    from wds_optimizer.api import build_problem
    from wds_optimizer.algorithms.config import build_algorithm_config
    from wds_optimizer.sweep.runner import run_parameter_sweep

    # Cache problem + config per benchmark so we don't reload the
    # network for every sweep of the same benchmark.
    problems: dict = {}
    cfgs: dict = {}

    for sweep in plan:
        bench = sweep["benchmark"]
        algo = sweep["algo"]
        param = sweep["param"]
        values = sweep["values"]
        runs = sweep["runs"]
        bcfg = BENCHMARK_CONFIGS[bench]
        max_gen = sweep["max_gen"] or bcfg["max_gen"]

        if bench not in problems:
            problem, units, wn = build_problem(
                inp_path=bcfg["inp"],
                d_path=bcfg["d"],
                cost_path=bcfg["cost"],
                diam_unit=bcfg["diam_unit"],
                constraints=bcfg["constraints"],
                fixed_pipes="",
            )
            problems[bench] = problem
            cfg = build_algorithm_config(NS=bcfg["ns"], MaxGen=max_gen)
            cfg.Seed = args.seed
            cfg.Cache = {}
            cfgs[bench] = cfg
        else:
            cfgs[bench].MaxGen = max_gen

        print(f"\n[{bench} / {algo}] {param} over {values}  "
              f"(runs={runs}, max_gen={max_gen})", flush=True)

        try:
            result = run_parameter_sweep(
                problems[bench], cfgs[bench],
                parameter_name=param,
                values=values,
                algorithms=[algo],
                n_runs=runs,
                experiment_seed=args.seed,
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        for p in result.points:
            marker = "✓" if p.status == "Success" else "✗"
            best = (f"${p.best_score:,.0f}"
                    if np.isfinite(p.best_score) else "—")
            print(f"  {marker} {_format_value(p.value):>10s}  "
                  f"best={best:>14s}  "
                  f"feas={p.feasible_pct:5.1f}%  "
                  f"({p.mean_time:.1f}s)")

        out_path = (
            output_dir / bench / algo
            / f"{param.replace('.', '_')}.json"
        )
        _save_sweep_json(result, out_path, bench, runs)
        print(f"  -> {out_path}")

    generate_markdown(output_dir, md_path)
    print(f"\nRegenerated {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())