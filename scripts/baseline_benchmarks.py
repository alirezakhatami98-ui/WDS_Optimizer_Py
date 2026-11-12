"""Baseline benchmark run: how do the 4 algorithms perform today?

Not a test.  A script to gather numbers before writing verification
tests.  Outputs a clean table of best / mean / feasible% per algorithm
per benchmark.

Usage:
    python scripts/baseline_benchmarks.py

Runtime: ~10-20 minutes (depends on machine).
Results are written to scripts/baseline_results.json for later use.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from wds_optimizer.api import OptimizationRequest, run_optimization


BENCHMARKS = {
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
        # Paper's best-known cost from Suribabu (2010):
        "paper_best": 6.081e6,
        # Network has 34 variable pipes; NS=200 per paper heuristic
        "ns": 200,
        "max_gen": 300,
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
        # Paper's best-known cost from Alperovits & Shamir (1977):
        "paper_best": 419_000.0,
        # 8 pipes; small NS is enough
        "ns": 100,
        "max_gen": 200,
    },
}

ALGORITHMS = ["GA", "PSO", "DE", "SADE"]
N_RUNS = 3
SEED = 42


def main():
    results = {}
    for bname, bcfg in BENCHMARKS.items():
        print(f"\n{'=' * 70}")
        print(f"Benchmark: {bname}  (paper best = ${bcfg['paper_best']:,.0f})")
        print(f"{'=' * 70}")
        results[bname] = {}

        for algo in ALGORITHMS:
            print(f"\n  Running {algo} (n_runs={N_RUNS}) ...", flush=True)
            t0 = time.perf_counter()
            req = OptimizationRequest(
                inp_path=bcfg["inp"],
                d_path=bcfg["d"],
                cost_path=bcfg["cost"],
                diam_unit=bcfg["diam_unit"],
                algorithms=[algo],
                n_runs=N_RUNS,
                seed=SEED,
                ns=bcfg["ns"],
                max_gen=bcfg["max_gen"],
                constraints=bcfg["constraints"],
            )
            try:
                result = run_optimization(req)
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}")
                results[bname][algo] = {"error": str(e)}
                continue

            dt = time.perf_counter() - t0
            runs = result.experiment.runs[algo]
            costs = [r.score for r in runs
                     if r.status in ("Success", "Partial") and r.feasible]
            n_feas = len(costs)
            best = min(costs) if costs else float("inf")
            mean = sum(costs) / len(costs) if costs else float("inf")

            gap_pct = (100 * (best - bcfg["paper_best"]) / bcfg["paper_best"]
                       if costs else float("inf"))

            results[bname][algo] = {
                "best": best,
                "mean": mean,
                "n_feasible": n_feas,
                "n_runs": len(runs),
                "gap_pct": gap_pct,
                "time_s": round(dt, 1),
            }

            print(f"    best   = ${best:>12,.0f}   gap = {gap_pct:>+6.1f}%")
            print(f"    mean   = ${mean:>12,.0f}")
            print(f"    feasible: {n_feas}/{len(runs)}   time: {dt:.1f}s")

    # Summary table
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Benchmark':<12} {'Algo':<6} {'Best':>14} {'Gap%':>8} "
          f"{'Feas':>6} {'Time':>6}")
    print(f"{'-' * 70}")
    for bname, r in results.items():
        for algo in ALGORITHMS:
            if algo not in r or "error" in r[algo]:
                print(f"{bname:<12} {algo:<6} ERROR")
                continue
            e = r[algo]
            print(f"{bname:<12} {algo:<6} ${e['best']:>13,.0f} "
                  f"{e['gap_pct']:>+7.1f}% "
                  f"{e['n_feasible']:>2}/{e['n_runs']} "
                  f"{e['time_s']:>5.1f}s")

    out = Path("scripts/baseline_results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)