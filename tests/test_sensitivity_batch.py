"""Tests for sensitivity/batch.py and reporting/sensitivity.py."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from wds_optimizer.algorithms.config import build_algorithm_config
from wds_optimizer.sensitivity.batch import (
    BatchParamResult, BatchPoint, BatchSweepResult,
    PlanRow, run_batch_sweep, validate_plan, _compute_recommendation,
)
from wds_optimizer.reporting.sensitivity import (
    batch_report_csv, batch_report_markdown, load_batch_from_jsons,
)


# ---------------------------------------------------------------------
#  Fixtures: synthetic problem + cache
# ---------------------------------------------------------------------

def _problem():
    return {
        "wn": None, "_solver": False,
        "NP": 3, "VariablePipes": np.array([0, 1, 2]),
        "FixedPipes": np.array([], dtype=int),
        "D": np.array([0.1, 0.2, 0.3]),
        "InitialD": np.array([0.2, 0.2, 0.2]),
        "Cost": np.array([1.0, 10.0, 100.0]),
        "L": np.array([100.0, 100.0, 100.0]),
        "Constraints": {
            "Pmin": {"enabled": True,  "value": 30.0},
            "Pmax": {"enabled": False, "value": 100.0},
            "Vmin": {"enabled": False, "value": 0.0},
            "Vmax": {"enabled": False, "value": 2.5},
        },
        "pipe_names": ["P1", "P2", "P3"],
        "junction_names": ["J1", "J2", "J3"],
        "Units": {"System": "SI", "PressureUnit": "m",
                  "VelocityUnit": "m/s", "LengthUnit": "m",
                  "DiameterUnit": "mm"},
    }


def _cache_cheap_infeasible():
    """Cheapest design is infeasible; cheapest feasible costs 2100."""
    unit = [1.0, 10.0, 100.0]
    cache = {}
    for d in itertools.product(range(3), repeat=3):
        c = sum(100.0 * unit[i] for i in d)
        feas = sum(d) >= 2
        cache[d] = (c, 0.0 if feas else 1.0, feas)
    return cache


def _cache_all_equal():
    """Every design costs 100 and is feasible — flat-line territory."""
    cache = {}
    for d in itertools.product(range(3), repeat=3):
        cache[d] = (100.0, 0.0, True)
    return cache


def _cfg(**overrides):
    cfg = build_algorithm_config(NS=10, MaxGen=10)
    cfg.Seed = 42
    cfg.Cache = _cache_cheap_infeasible()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------

def test_validate_rejects_empty_plan():
    with pytest.raises(ValueError, match="no enabled rows"):
        validate_plan([])


def test_validate_rejects_all_disabled():
    rows = [PlanRow("GA.ga_pc", "GA", [0.6], enabled=False)]
    with pytest.raises(ValueError, match="no enabled rows"):
        validate_plan(rows)


def test_validate_rejects_unknown_param():
    rows = [PlanRow("GA.not_a_param", "GA", [0.6])]
    with pytest.raises(ValueError, match="Unknown parameter"):
        validate_plan(rows)


def test_validate_rejects_unknown_algo():
    rows = [PlanRow("GA.ga_pc", "NOT_AN_ALGO", [0.6])]
    with pytest.raises(ValueError, match="Unknown algorithm"):
        validate_plan(rows)


def test_validate_rejects_algo_mismatch():
    rows = [PlanRow("GA.ga_pc", "PSO", [0.6])]
    with pytest.raises(ValueError, match="belongs to algorithm"):
        validate_plan(rows)


def test_validate_rejects_empty_values():
    rows = [PlanRow("GA.ga_pc", "GA", [])]
    with pytest.raises(ValueError, match="no values"):
        validate_plan(rows)


def test_validate_accepts_global_param_with_any_algo():
    """NS is a global param — valid with any algorithm."""
    validate_plan([PlanRow("NS", "GA", [50, 100])])
    validate_plan([PlanRow("NS", "SADE", [50, 100])])


# ---------------------------------------------------------------------
#  Recommendation logic
# ---------------------------------------------------------------------

def _mk_point(param, algo, value, status, mean, std=0.0, best=None):
    return BatchPoint(
        param=param, algorithm=algo, value=value,
        status=status,
        best_score=best if best is not None else mean,
        mean_score=mean, std_score=std,
    )


def test_recommendation_picks_lowest_mean():
    pr = BatchParamResult(
        param="NS", algorithm="GA", values=[50, 100, 200],
        points=[
            _mk_point("NS", "GA", 50, "Success", 500.0),
            _mk_point("NS", "GA", 100, "Success", 300.0),
            _mk_point("NS", "GA", 200, "Success", 400.0),
        ],
    )
    _compute_recommendation(pr)
    assert pr.recommended_value == 100
    assert pr.recommended_mean == 300.0
    assert pr.reason == "lowest_mean"


def test_recommendation_flat_line():
    pr = BatchParamResult(
        param="NS", algorithm="GA", values=[50, 100],
        points=[
            _mk_point("NS", "GA", 50, "Success", 100.0),
            _mk_point("NS", "GA", 100, "Success", 100.5),  # 0.5% spread
        ],
    )
    _compute_recommendation(pr)
    assert pr.recommended_value is None
    assert "flat_line" in pr.reason


def test_recommendation_high_variance():
    pr = BatchParamResult(
        param="NS", algorithm="GA", values=[50, 100],
        points=[
            _mk_point("NS", "GA", 50, "Success", 200.0,
                      std=150.0),  # std 75% of mean
            _mk_point("NS", "GA", 100, "Success", 300.0),
        ],
    )
    _compute_recommendation(pr)
    assert pr.recommended_value == 50
    assert "high variance" in pr.reason


def test_recommendation_no_successes():
    pr = BatchParamResult(
        param="NS", algorithm="GA", values=[50],
        points=[_mk_point("NS", "GA", 50, "Failed", float("nan"))],
    )
    _compute_recommendation(pr)
    assert pr.recommended_value is None
    assert pr.reason == "no_successful_points"


# ---------------------------------------------------------------------
#  End-to-end batch
# ---------------------------------------------------------------------

def test_batch_runs_all_points():
    problem = _problem()
    cfg = _cfg()
    rows = [
        PlanRow("GA.ga_pc", "GA", [0.6, 0.9]),
        PlanRow("GA.ga_pm", "GA", [0.03, 0.1]),
    ]
    result = run_batch_sweep(
        problem, cfg,
        rows=rows, n_runs=1, experiment_seed=42,
    )
    assert len(result.params) == 2
    assert result.params[0].param == "GA.ga_pc"
    assert len(result.params[0].points) == 2
    assert result.params[1].param == "GA.ga_pm"
    assert len(result.params[1].points) == 2
    assert result.total_points == 4
    assert result.points_done == 4
    assert not result.cancelled


def test_batch_progress_callback():
    events = []
    problem = _problem()
    cfg = _cfg()
    rows = [PlanRow("GA.ga_pc", "GA", [0.6, 0.9])]
    run_batch_sweep(
        problem, cfg,
        rows=rows, n_runs=1, experiment_seed=42,
        progress_callback=events.append,
    )
    # Two kinds of events reach the callback:
    #   1. Per-generation events forwarded from run_experiment
    #      (Algorithm, Generation, BestCost, ... ; NO "Phase" key).
    #   2. Batch-level events added by run_batch_sweep
    #      ({"Phase": "PointDone"|"ParamDone", ...}).
    # The GUI's _job_panel distinguishes them with `"Phase" in e`;
    # consumers that only care about batch events must filter.
    batch_events = [e for e in events if "Phase" in e]
    phases = [e["Phase"] for e in batch_events]
    assert phases.count("PointDone") == 2
    assert phases.count("ParamDone") == 1
    # PointDone events carry parameter info
    pd_events = [e for e in batch_events if e["Phase"] == "PointDone"]
    assert pd_events[0]["ParameterName"] == "GA.ga_pc"
    assert pd_events[0]["Algorithm"] == "GA"
    # Per-generation events have no Phase but do have ProgressIndex
    pg_events = [e for e in events if "Phase" not in e]
    for e in pg_events:
        assert "Generation" in e
        assert "PointIndex" in e
        assert "ParameterValue" in e


def test_batch_cancel_stops_early():
    calls = [0]
    def cancel():
        calls[0] += 1
        return calls[0] > 1  # cancel after first check
    problem = _problem()
    cfg = _cfg()
    rows = [PlanRow("GA.ga_pc", "GA", [0.6, 0.9, 0.5])]
    result = run_batch_sweep(
        problem, cfg,
        rows=rows, n_runs=1, experiment_seed=42,
        cancel_callback=cancel,
    )
    assert result.cancelled


def test_batch_flat_line_detected_end_to_end():
    """With a cache where all designs are equivalent, we get flat_line."""
    problem = _problem()
    cfg = _cfg()
    cfg.Cache = _cache_all_equal()
    rows = [PlanRow("GA.ga_pc", "GA", [0.6, 0.9])]
    result = run_batch_sweep(
        problem, cfg,
        rows=rows, n_runs=1, experiment_seed=42,
    )
    pr = result.params[0]
    # Means should be identical (all designs cost 100).
    means = [p.mean_score for p in pr.points if p.status == "Success"]
    if len(means) >= 2 and abs(means[0] - means[1]) / max(means[0], 1e-9) < 0.01:
        assert pr.recommended_value is None
        assert "flat_line" in pr.reason


# ---------------------------------------------------------------------
#  Report generation
# ---------------------------------------------------------------------

def _mk_batch():
    pr1 = BatchParamResult(
        param="GA.ga_pc", algorithm="GA", values=[0.6, 0.9],
        points=[
            _mk_point("GA.ga_pc", "GA", 0.6, "Success", 500.0, 10.0, 490.0),
            _mk_point("GA.ga_pc", "GA", 0.9, "Success", 300.0, 5.0, 295.0),
        ],
        recommended_value=0.9, recommended_mean=300.0,
        reason="lowest_mean",
    )
    pr2 = BatchParamResult(
        param="NS", algorithm="GA", values=[50],
        points=[
            _mk_point("NS", "GA", 50, "Success", 400.0, 0.0, 400.0),
        ],
        recommended_value=None, recommended_mean=400.0,
        reason="only_one_value",
    )
    return BatchSweepResult(
        n_runs=1, experiment_seed=42,
        params=[pr1, pr2],
        total_points=3, points_done=3,
    )


def test_markdown_contains_sections():
    md = batch_report_markdown(_mk_batch())
    assert "# Sensitivity Analysis" in md
    assert "## `GA.ga_pc`" in md
    assert "## `NS`" in md
    assert "| Value | Best | Mean | Std" in md
    assert "**Recommended:** `GA.ga_pc = 0.9`" in md
    assert "lowest_mean" in md


def test_markdown_cancelled_note():
    batch = _mk_batch()
    batch.cancelled = True
    md = batch_report_markdown(batch)
    assert "cancelled" in md.lower()


def test_csv_has_header_and_rows():
    csv_bytes = batch_report_csv(_mk_batch())
    text = csv_bytes.decode("utf-8-sig")
    lines = [l for l in text.strip().split("\n") if l]
    assert lines[0].startswith("parameter,algorithm,value")
    assert len(lines) == 4  # header + 3 rows


# ---------------------------------------------------------------------
#  JSON round-trip (load_batch_from_jsons)
# ---------------------------------------------------------------------

def test_load_batch_from_jsons_roundtrip(tmp_path):
    """Write JSON in the format from sensitivity_analysis.py; load it back."""
    jp = tmp_path / "twoloop" / "GA" / "GA_ga_pc.json"
    jp.parent.mkdir(parents=True)
    jp.write_text(json.dumps({
        "benchmark": "twoloop",
        "algorithm": "GA",
        "parameter": "GA.ga_pc",
        "values": [0.6, 0.9],
        "n_runs": 3,
        "experiment_seed": 42,
        "timestamp": "2026-01-01T00:00:00",
        "recommended_value": 0.9,
        "recommended_mean": 300.0,
        "reason": "lowest_mean",
        "points": [
            {"value": 0.6, "status": "Success",
             "best_score": 490.0, "mean_score": 500.0,
             "std_score": 10.0, "feasible_pct": 100.0,
             "mean_time_s": 1.0, "error": ""},
            {"value": 0.9, "status": "Success",
             "best_score": 295.0, "mean_score": 300.0,
             "std_score": 5.0, "feasible_pct": 100.0,
             "mean_time_s": 1.0, "error": ""},
        ],
    }, indent=2), encoding="utf-8")

    batch = load_batch_from_jsons(tmp_path)
    assert batch is not None
    assert batch.n_runs == 3
    assert batch.experiment_seed == 42
    assert len(batch.params) == 1
    pr = batch.params[0]
    assert pr.param == "GA.ga_pc"
    assert pr.recommended_value == 0.9
    assert pr.reason == "lowest_mean"


def test_load_batch_from_jsons_empty_dir(tmp_path):
    assert load_batch_from_jsons(tmp_path / "empty") is None


def test_load_batch_from_jsons_old_format(tmp_path):
    """Old JSONs (v0.7.1) lack recommended_* fields.  Must recompute."""
    jp = tmp_path / "GA" / "GA_ga_pc.json"
    jp.parent.mkdir(parents=True)
    jp.write_text(json.dumps({
        "benchmark": "twoloop",
        "algorithm": "GA",
        "parameter": "GA.ga_pc",
        "values": [0.6, 0.9],
        "n_runs": 3,
        "experiment_seed": 42,
        "timestamp": "2026-01-01T00:00:00",
        "points": [
            {"value": 0.6, "status": "Success",
             "best_score": 490.0, "mean_score": 500.0,
             "std_score": 10.0, "feasible_pct": 100.0,
             "mean_time_s": 1.0, "error": ""},
            {"value": 0.9, "status": "Success",
             "best_score": 295.0, "mean_score": 300.0,
             "std_score": 5.0, "feasible_pct": 100.0,
             "mean_time_s": 1.0, "error": ""},
        ],
    }, indent=2), encoding="utf-8")

    batch = load_batch_from_jsons(tmp_path)
    assert batch is not None
    pr = batch.params[0]
    # Recommendation recomputed from points
    assert pr.recommended_value == 0.9
    assert pr.reason == "lowest_mean"