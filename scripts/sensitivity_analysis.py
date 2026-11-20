"""Sensitivity analysis pipeline for WDS Optimizer.

Runs a predefined set of parameter sweeps on the benchmarks and
writes the results as JSON plus a generated Markdown report.

Since v0.8.1 this script uses the batch API
(wds_optimizer.sensitivity.batch.run_batch_sweep) and the shared
report generator (wds_optimizer.reporting.sensitivity) so that the
GUI (v0.8.2) produces the same format.

CLI and JSON output are unchanged from v0.7.2.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


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
    ("twoloop", "GA",   "GA.ga_pc",        [0.5, 0.6, 0.7, 0.8, 0.9]),
    ("twoloop", "SADE", "SADE.sade_stag",  [100, 300, 500]),
    ("hanoi", "GA",   "NS",     [100, 200, 300]),
    ("hanoi", "GA",   "MaxGen", [200, 300, 500]),
    ("hanoi", "SADE", "n_runs", [1, 5, 10]),
]


# ---------------------------------------------------------------------
#  JSON helpers (unchanged format from v0.7.1)
# ---------------------------------------------------------------------

def _value_to_jsonable(v: Any) -> Any:
    if isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def _save_batch_param_json(
    pr, batch, output_dir: Path, benchmark: str,
) -> Path:
    """Write one BatchParamResult as a per-parameter JSON.

    Format is the same as v0.7.1 (plus optional recommendation
    fields), so existing sensitivity_data files are compatible.
    """
    out_path = (
        output_dir / benchmark / pr.algorithm
        / f"{pr.param.replace('.', '_')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "benchmark": benchmark,
        "algorithm": pr.algorithm,
        "parameter": pr.param,
        "values": [_value_to_jsonable(v) for v in pr.values],
        "n_runs": batch.n_runs,
        "experiment_seed": batch.experiment_seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "recommended_value": (
            _value_to_jsonable(pr.recommended_value)
            if pr.recommended_value is not None else None
        ),
        "recommended_mean": pr.recommended_mean,
        "reason": pr.reason,
        "points": [
            {
                "value": _value_to_jsonable(p.value),
                "status": p.status,
                "best_score": (
                    p.best_score if np.isfinite(p.best_score) else None
                ),
                "mean_score": (
                    p.mean_score if np.isfinite(p.mean_score) else None
                ),
                "std_score": (
                    p.std_score if np.isfinite(p.std_score) else None
                ),
                "feasible_pct": p.feasible_pct,
                "mean_time_s": p.mean_time,
                "error": p.error,
            }
            for p in pr.points
        ],
    }
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------
#  Plan building
# ---------------------------------------------------------------------

def _build_plan(args) -> list[dict]:
    if args.param:
        if not args.benchmark or not args.algo or not args.values:
            raise ValueError(
                "--param requires --benchmark, --algo, and --values."
            )
        from wds_optimizer.sensitivity.config import (
            parse_sensitivity_values,
        )
        return [{
            "benchmark": args.benchmark,
            "algo": args.algo,
            "param": args.param,
            "values": parse_sensitivity_values(args.values),
            "runs": args.runs if args.runs is not None else 1,
            "max_gen": args.max_gen,
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
#  Execution
# ---------------------------------------------------------------------

def _run(args) -> int:
    from wds_optimizer.sensitivity.batch import (
        PlanRow, run_batch_sweep, validate_plan,
    )
    from wds_optimizer.reporting.sensitivity import (
        batch_report_markdown, load_batch_from_jsons,
    )
    from wds_optimizer.api import build_problem
    from wds_optimizer.algorithms.config import build_algorithm_config

    output_dir = Path(args.output_dir)
    md_path = Path(args.markdown)

    if args.regenerate_only:
        batch = load_batch_from_jsons(output_dir)
        if batch is None:
            md = "# Sensitivity Analysis\n\n_No sensitivity data yet._\n"
        else:
            md = batch_report_markdown(batch)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        print(f"Regenerated {md_path}")
        return 0

    plan = _build_plan(args)

    if args.dry_run:
        print(f"Dry-run: {len(plan)} sweep(s) would run:")
        for s in plan:
            print(
                f"  {s['benchmark']:8s} / {s['algo']:4s} / "
                f"{s['param']:20s}  values={s['values']}  "
                f"runs={s['runs']}  max_gen={s['max_gen']}"
            )
        return 0

    # Group by benchmark; one batch per benchmark.
    by_bench: dict[str, list[dict]] = {}
    for s in plan:
        by_bench.setdefault(s["benchmark"], []).append(s)

    md_sections: list[str] = [
        "# Sensitivity Analysis", "",
        "_Auto-generated by scripts/sensitivity_analysis.py._",
        "_Regenerate with: "
        "`python scripts/sensitivity_analysis.py --regenerate-only`._",
        "",
    ]

    problems: dict = {}

    for bench, sweeps in by_bench.items():
        bcfg = BENCHMARK_CONFIGS[bench]
        if bench not in problems:
            problem, _, _ = build_problem(
                inp_path=bcfg["inp"],
                d_path=bcfg["d"],
                cost_path=bcfg["cost"],
                diam_unit=bcfg["diam_unit"],
                constraints=bcfg["constraints"],
                fixed_pipes="",
            )
            problems[bench] = problem

        # Rows for this benchmark
        rows = [
            PlanRow(
                param=s["param"],
                algorithm=s["algo"],
                values=s["values"],
            )
            for s in sweeps
        ]
        # Validate before running
        try:
            validate_plan(rows)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        # Config: take max_gen/runs from the first sweep (all sweeps of
        # a benchmark in our plans use the same max_gen/runs).
        max_gen = sweeps[0]["max_gen"] or bcfg["max_gen"]
        runs = sweeps[0]["runs"]
        cfg = build_algorithm_config(
            NS=bcfg["ns"], MaxGen=max_gen,
        )
        cfg.Seed = args.seed
        cfg.Cache = {}

        print(
            f"\n[{bench}] {len(rows)} row(s), "
            f"runs={runs}, max_gen={max_gen}",
            flush=True,
        )
        batch = run_batch_sweep(
            problems[bench], cfg,
            rows=rows,
            n_runs=runs,
            experiment_seed=args.seed,
        )

        for pr in batch.params:
            for p in pr.points:
                marker = "✓" if p.status == "Success" else "✗"
                best = (
                    f"${p.best_score:,.0f}"
                    if np.isfinite(p.best_score) else "—"
                )
                print(
                    f"  {marker} {pr.param:20s} "
                    f"{str(p.value):>10s}  "
                    f"best={best:>14s}  "
                    f"feas={p.feasible_pct:5.1f}%  "
                    f"({p.mean_time:.1f}s)"
                )
            out_path = _save_batch_param_json(
                pr, batch, output_dir, bench,
            )
            print(f"  -> {out_path}")

    # Regenerate markdown from JSONs (ensures consistency)
    full_batch = load_batch_from_jsons(output_dir)
    if full_batch is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(
            batch_report_markdown(full_batch),
            encoding="utf-8",
        )
        print(f"\nRegenerated {md_path}")
    return 0


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sensitivity_analysis",
        description="Run sensitivity sweeps on benchmarks.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--regenerate-only", action="store_true")

    p.add_argument("--benchmark", choices=sorted(BENCHMARK_CONFIGS))
    p.add_argument("--algo")
    p.add_argument("--param")
    p.add_argument("--values")
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--max-gen", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="docs/sensitivity_data")
    p.add_argument("--markdown", default="docs/sensitivity_analysis.md")
    p.add_argument("--dry-run", action="store_true")
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


if __name__ == "__main__":
    sys.exit(main())