"""Markdown and CSV report generation for sensitivity batches.

Shared between the sensitivity_analysis.py script and (in v0.8.2)
the GUI's Sensitivity tab.  One source of truth for report layout.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..sensitivity.batch import (
    BatchParamResult, BatchPoint, BatchSweepResult,
)


# ---------------------------------------------------------------------
#  Formatting helpers
# ---------------------------------------------------------------------

def _fmt_value(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _fmt_cost(c) -> str:
    if c is None:
        return "—"
    try:
        c = float(c)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(c):
        return "—"
    return f"${c:,.0f}"


# ---------------------------------------------------------------------
#  Markdown report
# ---------------------------------------------------------------------

def batch_report_markdown(
    batch: BatchSweepResult,
    *,
    benchmark_name: str = "",
    paper_best: Optional[float] = None,
    title: str = "Sensitivity Analysis",
) -> str:
    """Return a complete Markdown report for a batch."""
    lines = [
        f"# {title}",
        "",
        f"_n_runs = {batch.n_runs}, "
        f"seed = {batch.experiment_seed}, "
        f"{len(batch.params)} parameter sweep(s)._",
    ]
    if benchmark_name:
        lines.append(f"\n**Benchmark:** {benchmark_name}")
    if paper_best is not None:
        lines.append(f"**Paper best cost:** {_fmt_cost(paper_best)}")
    if batch.cancelled:
        lines.append(
            "\n**Note:** this batch was cancelled; only completed "
            "points are shown."
        )
    lines.append("")

    for pr in batch.params:
        lines.extend(_param_section(pr))

    return "\n".join(lines)


def _param_section(pr: BatchParamResult) -> list[str]:
    lines = [
        f"\n## `{pr.param}`  ({pr.algorithm})",
        "",
        "| Value | Best | Mean | Std | Feasible % | Time (s) |",
        "|-------|------|------|-----|------------|----------|",
    ]
    for p in pr.points:
        lines.append(
            f"| {_fmt_value(p.value)} | {_fmt_cost(p.best_score)} | "
            f"{_fmt_cost(p.mean_score)} | {_fmt_cost(p.std_score)} | "
            f"{p.feasible_pct:.0f}% | {p.mean_time:.1f} |"
        )

    if pr.recommended_value is not None:
        lines.append(
            f"\n**Recommended:** `{pr.param} = "
            f"{_fmt_value(pr.recommended_value)}` "
            f"(mean {_fmt_cost(pr.recommended_mean)}; {pr.reason})"
        )
    elif pr.reason:
        lines.append(f"\n**Note:** {pr.reason}")

    return lines


# ---------------------------------------------------------------------
#  CSV report
# ---------------------------------------------------------------------

def batch_report_csv(batch: BatchSweepResult) -> bytes:
    """Return a flat CSV, one row per (parameter, algorithm, value)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "parameter", "algorithm", "value",
        "status", "best", "mean", "std",
        "feasible_pct", "mean_time_s", "error",
    ])
    for pr in batch.params:
        for p in pr.points:
            w.writerow([
                pr.param,
                pr.algorithm,
                _fmt_value(p.value),
                p.status,
                (f"{p.best_score:.2f}"
                 if np.isfinite(p.best_score) else ""),
                (f"{p.mean_score:.2f}"
                 if np.isfinite(p.mean_score) else ""),
                (f"{p.std_score:.2f}"
                 if np.isfinite(p.std_score) else ""),
                f"{p.feasible_pct:.1f}",
                f"{p.mean_time:.2f}",
                p.error,
            ])
    # utf-8-sig so Excel opens it with the correct encoding
    return buf.getvalue().encode("utf-8-sig")


# ---------------------------------------------------------------------
#  Load batch from JSONs (for --regenerate-only in the script)
# ---------------------------------------------------------------------

def load_batch_from_jsons(output_dir: Path) -> Optional[BatchSweepResult]:
    """Reconstruct a BatchSweepResult from saved JSONs in output_dir.

    Each JSON corresponds to one BatchParamResult (one parameter,
    one algorithm).  Field names match those written by
    sensitivity_analysis.py.

    Returns None when there are no JSONs.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    jsons = sorted(output_dir.rglob("*.json"))
    if not jsons:
        return None

    params: list[BatchParamResult] = []
    n_runs = 1
    seed = 42

    for jp in jsons:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[sensitivity] skipping {jp}: {e}")
            continue

        n_runs = int(data.get("n_runs", n_runs))
        seed = int(data.get("experiment_seed", seed))

        pr = BatchParamResult(
            param=data["parameter"],
            algorithm=data["algorithm"],
            values=list(data["values"]),
        )
        for pd in data.get("points", []):
            point = BatchPoint(
                param=data["parameter"],
                algorithm=data["algorithm"],
                value=pd["value"],
                status=pd.get("status", "Failed"),
                best_score=(
                    float(pd["best_score"])
                    if pd.get("best_score") is not None
                    else float("inf")
                ),
                mean_score=(
                    float(pd["mean_score"])
                    if pd.get("mean_score") is not None
                    else float("nan")
                ),
                std_score=(
                    float(pd["std_score"])
                    if pd.get("std_score") is not None
                    else float("nan")
                ),
                feasible_pct=float(pd.get("feasible_pct", 0.0)),
                mean_time=float(pd.get("mean_time_s", 0.0)),
                error=pd.get("error", ""),
            )
            pr.points.append(point)

        # Use saved recommendation if present; else recompute.
        if "recommended_value" in data:
            pr.recommended_value = data["recommended_value"]
            pr.recommended_mean = data.get("recommended_mean")
            pr.reason = data.get("reason", "")
        else:
            from ..sensitivity.batch import _compute_recommendation
            _compute_recommendation(pr)

        params.append(pr)

    if not params:
        return None

    total = sum(len(pr.points) for pr in params)
    return BatchSweepResult(
        n_runs=n_runs,
        experiment_seed=seed,
        params=params,
        total_points=total,
        points_done=total,
    )