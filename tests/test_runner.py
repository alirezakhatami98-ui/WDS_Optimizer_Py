"""Regression tests for run_experiment (experiments/runner.py).

Motivating bug (user-reported):

    run_experiment(...)
    → File "runner.py", line 104, in run_experiment
        Pmin=problem["Pmin"],
      KeyError: 'Pmin'

The Pmax/Vmin commit moved all constraint information from top-level
``problem["Pmin"]`` / ``problem["Vmax"]`` keys into a single
``problem["Constraints"]`` dict.  runner.py was not updated in the same
commit and broke on the very next run.  These tests pin the fix and the
surrounding behaviour.

Design
------
The four algorithm functions (run_ga / run_pso / run_de / run_sade)
are patched with stubs so no real hydraulic simulation is required.
Test runtime: < 1 s.

The ``_problem`` helper deliberately builds a dict WITHOUT top-level
``Pmin``/``Vmax``.  If runner.py ever reverts to ``problem["Pmin"]``,
every test using this helper fails with KeyError — catching the
regression rather than silently passing.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from wds_optimizer.algorithms.config import build_algorithm_config
from wds_optimizer.algorithms.ga import GACancelled
from wds_optimizer.experiments.runner import (
    ExperimentResult,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _problem(
    *, NP: int = 5,
    pmin=30.0, pmax=100.0, vmin=0.0, vmax=2.5,
    use_pmin=True, use_pmax=False, use_vmin=False, use_vmax=True,
) -> dict:
    """Minimal ``problem`` dict accepted by run_experiment.

    NO top-level ``Pmin`` or ``Vmax`` — the exact shape produced by
    build_optimization_problem after the Pmax/Vmin commit.
    """
    return {
        "wn": None,                          # not used by runner
        "NP": NP,
        "VariablePipes": np.arange(NP),
        "FixedPipes": np.array([], dtype=int),
        "Constraints": {
            "Pmin": {"enabled": bool(use_pmin), "value": float(pmin)},
            "Pmax": {"enabled": bool(use_pmax), "value": float(pmax)},
            "Vmin": {"enabled": bool(use_vmin), "value": float(vmin)},
            "Vmax": {"enabled": bool(use_vmax), "value": float(vmax)},
        },
    }


def _make_fake_algo(*, cost: float = 1000.0, feasible: bool = True,
                    nfe: int = 100, n_sim: int = 100,
                    nfe_to_best: int = 50, progress_info: dict | None = None):
    """Return a stub with the same signature as run_ga / run_pso / ..."""
    def fake(problem, cfg, progress_callback=None, cancel_callback=None,
             meta=None):
        if progress_callback is not None and progress_info is not None:
            progress_callback(progress_info)
        if meta is not None:
            meta["nfe"] = nfe
            meta["n_sim"] = n_sim
            meta["nfe_to_best"] = nfe_to_best
            meta["restart_gens"] = []
            meta["epoch_bests"] = [cost]
            meta["extra"] = {}
        position = np.zeros(problem["NP"])
        convergence = np.array([cost, cost, cost])
        return cost, position, convergence, feasible
    return fake


def _run(problem, base_config, *, algorithms=("GA",), n_runs=1, seed=1,
         fake=None, cancel_callback=None, progress_callback=None):
    """Run run_experiment with all four algorithm functions patched."""
    fake = fake or _make_fake_algo()
    with patch("wds_optimizer.experiments.runner.run_ga", fake), \
         patch("wds_optimizer.experiments.runner.run_pso", fake), \
         patch("wds_optimizer.experiments.runner.run_de", fake), \
         patch("wds_optimizer.experiments.runner.run_sade", fake):
        return run_experiment(
            problem, base_config,
            number_of_runs=n_runs,
            algorithms=list(algorithms),
            experiment_seed=seed,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )


# ---------------------------------------------------------------------------
# THE regression: no top-level Pmin / Vmax keys
# ---------------------------------------------------------------------------

def test_no_legacy_pmin_vmax_keys_required():
    """run_experiment must not require problem["Pmin"] / ["Vmax"]."""
    problem = _problem()
    assert "Pmin" not in problem           # sanity: reproduces bug shape
    assert "Vmax" not in problem

    result = _run(problem, build_algorithm_config(NS=10, MaxGen=2))

    assert isinstance(result, ExperimentResult)
    assert result.runs["GA"][0].status == "Success"


def test_experiment_result_carries_constraint_values():
    """The four SI values come from problem["Constraints"]."""
    problem = _problem(pmin=25.0, pmax=180.0, vmin=0.3, vmax=3.0)
    cfg = build_algorithm_config(NS=10, MaxGen=2)

    result = _run(problem, cfg)

    assert result.Pmin == pytest.approx(25.0)
    assert result.Pmax == pytest.approx(180.0)
    assert result.Vmin == pytest.approx(0.3)
    assert result.Vmax == pytest.approx(3.0)


def test_experiment_result_carries_enabled_flags():
    """use_* flags mirror problem["Constraints"][*]["enabled"]."""
    problem = _problem(use_pmin=True, use_pmax=True,
                       use_vmin=False, use_vmax=False)
    result = _run(problem, build_algorithm_config(NS=10, MaxGen=2))

    assert result.use_pmin is True
    assert result.use_pmax is True
    assert result.use_vmin is False
    assert result.use_vmax is False


def test_defaults_match_legacy_behaviour():
    """App defaults (Pmin+Vmax on) preserve the pre-Pmax/Vmin shape."""
    problem = _problem()                       # defaults
    result = _run(problem, build_algorithm_config(NS=10, MaxGen=2))

    assert result.Pmin == pytest.approx(30.0)
    assert result.Vmax == pytest.approx(2.5)
    assert result.use_pmin is True
    assert result.use_vmax is True
    # Pmax/Vmin off but their values are still recorded:
    assert result.Pmax == pytest.approx(100.0)
    assert result.Vmin == pytest.approx(0.0)
    assert result.use_pmax is False
    assert result.use_vmin is False


# ---------------------------------------------------------------------------
# Structural guarantees
# ---------------------------------------------------------------------------

def test_all_four_algorithms_accepted():
    problem = _problem()
    result = _run(problem, build_algorithm_config(NS=10, MaxGen=2),
                  algorithms=("GA", "PSO", "DE", "SADE"), n_runs=1)
    assert result.algorithms == ["GA", "PSO", "DE", "SADE"]
    for algo in ("GA", "PSO", "DE", "SADE"):
        assert result.runs[algo][0].status == "Success"


def test_n_runs_creates_n_records_per_algorithm():
    problem = _problem()
    result = _run(problem, build_algorithm_config(NS=10, MaxGen=2),
                  algorithms=("GA", "SADE"), n_runs=4)
    assert len(result.runs["GA"]) == 4
    assert len(result.runs["SADE"]) == 4
    assert [r.run_index for r in result.runs["GA"]] == [1, 2, 3, 4]


def test_per_run_seeds_are_unique():
    problem = _problem()
    result = _run(problem, build_algorithm_config(NS=10, MaxGen=2),
                  algorithms=("GA",), n_runs=5)
    seeds = [r.seed for r in result.runs["GA"]]
    assert len(set(seeds)) == 5


def test_meta_propagates_to_run_record():
    """meta written by the algorithm lands on the RunRecord."""
    fake = _make_fake_algo(nfe=200, n_sim=180, nfe_to_best=120)
    result = _run(_problem(), build_algorithm_config(NS=10, MaxGen=2),
                  fake=fake)
    r = result.runs["GA"][0]
    assert r.nfe == 200
    assert r.n_sim == 180
    assert r.nfe_to_best == 120

def test_base_config_is_not_mutated():
    """run_experiment copies per-run config; caller's stays intact."""
    problem = _problem()
    cfg = build_algorithm_config(NS=10, MaxGen=2)
    cfg.GA.Pc = 0.42
    cfg.SADE.Fmin = 0.15
    _run(problem, cfg, algorithms=("GA", "SADE"), n_runs=2)
    assert cfg.GA.Pc == 0.42
    assert cfg.SADE.Fmin == 0.15


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def test_progress_callback_receives_run_context():
    """The wrapper adds RunIndex / NumberOfRuns / RunLabel per call."""
    captured = []
    fake = _make_fake_algo(progress_info={"Generation": 1, "BestCost": 500.0})
    _run(_problem(), build_algorithm_config(NS=10, MaxGen=2),
         n_runs=2, fake=fake,
         progress_callback=lambda info: captured.append(info))
    assert len(captured) == 2
    assert captured[0]["RunIndex"] == 1
    assert captured[0]["NumberOfRuns"] == 2
    assert captured[0]["RunLabel"] == "GA run 1/2"
    assert captured[1]["RunIndex"] == 2
    assert captured[1]["RunLabel"] == "GA run 2/2"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

def test_cancel_between_runs_marks_remaining_as_cancelled():
    """cancel_callback() True before run 2 → runs 2..n become Cancelled."""
    calls = [0]
    def cancel():
        calls[0] += 1
        return calls[0] > 1
    result = _run(_problem(), build_algorithm_config(NS=10, MaxGen=2),
                  n_runs=3, cancel_callback=cancel)
    statuses = [r.status for r in result.runs["GA"]]
    assert statuses[0] == "Success"
    assert statuses[1] == "Cancelled"
    assert statuses[2] == "Cancelled"
    assert result.cancelled is True


def test_cancel_before_first_run():
    result = _run(_problem(), build_algorithm_config(NS=10, MaxGen=2),
                  n_runs=2, cancel_callback=lambda: True)
    statuses = [r.status for r in result.runs["GA"]]
    assert all(s == "Cancelled" for s in statuses)
    assert result.cancelled is True


def test_cancel_during_run_keeps_partial_result():
    """GACancelled with a partial tuple → status "Partial", subsequent runs
    NotStarted (the loop breaks)."""
    def fake_partial(problem, cfg, progress_callback=None,
                     cancel_callback=None, meta=None):
        if meta is not None:
            meta["nfe"] = 50
            meta["n_sim"] = 50
            meta["nfe_to_best"] = 30
            meta["restart_gens"] = []
            meta["epoch_bests"] = [1000.0]
            meta["extra"] = {}
        partial = (1000.0, np.zeros(problem["NP"]),
                   np.array([1000.0]), True)
        raise GACancelled("test cancel", partial=partial)

    result = _run(_problem(), build_algorithm_config(NS=10, MaxGen=100),
                  n_runs=2, fake=fake_partial)
    assert result.runs["GA"][0].status == "Partial"
    assert result.runs["GA"][0].score == pytest.approx(1000.0)
    assert result.runs["GA"][0].feasible is True
    assert result.runs["GA"][1].status == "NotStarted"
    assert result.cancelled is True


def test_unexpected_error_marks_run_failed():
    """A generic Exception in the algorithm → status "Failed", no break."""
    def fake_failing(problem, cfg, progress_callback=None,
                     cancel_callback=None, meta=None):
        raise ValueError("boom")

    result = _run(_problem(), build_algorithm_config(NS=10, MaxGen=2),
                  n_runs=2, fake=fake_failing)
    assert result.runs["GA"][0].status == "Failed"
    assert "boom" in result.runs["GA"][0].error_message
    assert result.runs["GA"][1].status == "Failed"   # both fail, no break