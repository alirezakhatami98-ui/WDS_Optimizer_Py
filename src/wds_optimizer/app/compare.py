"""Compare two experiments."""

import numpy as np


def compute_comparison_stats(exp) -> dict:
    """Per-algorithm statistics for one experiment."""
    result = {}
    for algo, runs in exp.runs.items():
        ok = [r for r in runs if r.status == "Success"]
        feas = [r for r in ok if r.feasible]
        costs = np.array([r.score for r in feas]) if feas else np.array([])
        times = np.array([r.execution_time for r in ok]) if ok else np.array([])
        result[algo] = {
            "n_total": len(runs),
            "n_success": len(ok),
            "n_feasible": len(feas),
            "best": float(costs.min()) if costs.size else float("nan"),
            "mean": float(costs.mean()) if costs.size else float("nan"),
            "std": float(costs.std(ddof=1)) if costs.size > 1 else 0.0,
            "mean_time": float(times.mean()) if times.size else float("nan"),
            "costs": costs.tolist(),
        }
    return result


def overall_stats(per_algo: dict) -> dict:
    all_costs = []
    for s in per_algo.values():
        all_costs.extend(s["costs"])
    costs = np.array(all_costs)
    return {
        "n_feasible": int(costs.size),
        "best": float(costs.min()) if costs.size else float("nan"),
        "mean": float(costs.mean()) if costs.size else float("nan"),
        "std": float(costs.std(ddof=1)) if costs.size > 1 else 0.0,
    }