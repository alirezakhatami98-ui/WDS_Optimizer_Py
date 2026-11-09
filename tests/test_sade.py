"""Regression tests for SADE (algorithms/sade.py).

The SADE implementation is the central algorithm of this project; the
chat that produced these fixes started from "SADE doesn't match the
reference paper".  The specific fixes that matter here are:

1.  **jrand enforcement** in the crossover operator (Eq. 3).  Without
    it, a trial can be an exact copy of its target with probability
    (1 - CR)^NVar, which is ~11% for the paper's NYTP case (NVar=21,
    CR=0.1).  An identical trial is scored "not worse", so the parent
    survives — and its F/CR never refresh, defeating the self-
    adaptation of Eq. 5.  In practice the search stalls in the first
    basin it finds.

2.  **Cv on feasible members only** (matching the reference MATLAB
    implementation).  Infeasible designs are typically cheaper (small
    pipes), so including them inflates the mean Cv and delays the
    paper's convergence test indefinitely.

3.  **Clip** out-of-bounds values rather than re-drawing them
    uniformly (matching the reference MATLAB implementation).

4.  **StagnationMax** safeguard so a run that never satisfies
    Cv < Tol still terminates at a reasonable generation count.

Design
------
All tests pre-populate ``config.Cache`` with every possible design, so
``evaluate_population`` returns immediately from cache and no WNTR /
EPANET / hydraulic simulation is involved.  Every possible integer
design (ND ** NP of them) is present for the small sizes used here.

Runtime: < 1 s for the entire file.

A fast "slow test" that runs SADE against a real benchmark network
(Hanoi or NYTP, converging to $6.08M or $38.64M respectively) would be
more end-to-end but requires test fixture files that may not be
present; it is deferred to a follow-up that ships those fixtures.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from wds_optimizer.algorithms.config import build_algorithm_config
from wds_optimizer.algorithms.sade import run_sade


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _problem(NP: int, ND: int) -> dict:
    """Minimal problem dict accepted by run_sade + evaluate_population.

    ``_solver`` is set to False so ``get_solver`` returns None without
    trying to instantiate an EPANET project (it would crash on wn=None).
    With every design pre-cached, the solver is never used anyway.
    """
    return {
        "wn": None,
        "_solver": False,
        "NP": NP,
        "VariablePipes": np.arange(NP),
        "FixedPipes": np.array([], dtype=int),
        "D": np.linspace(0.1, 0.3, ND),           # ND diameter options
        "InitialD": np.full(NP, 0.2),              # arbitrary
        "Cost": np.ones(ND),                       # unused when cached
        "L": np.ones(NP),                          # unused when cached
        "Constraints": {
            "Pmin": {"enabled": True, "value": 30.0},
            "Pmax": {"enabled": False, "value": 100.0},
            "Vmin": {"enabled": False, "value": 0.0},
            "Vmax": {"enabled": True, "value": 2.5},
        },
        "pipe_names": [f"P{i+1}" for i in range(NP)],
        "junction_names": [f"J{i+1}" for i in range(NP)],
        "Units": {"System": "SI", "PressureUnit": "m",
                  "VelocityUnit": "m/s", "LengthUnit": "m",
                  "DiameterUnit": "mm"},
    }


def _sade_cfg(*, NS: int, MaxGen: int, seed: int,
              Fmin=0.1, Fmax=0.9, CRmin=0.1, CRmax=0.9,
              use_conv=True, tol=1e-6, stagn=0):
    cfg = build_algorithm_config(NS=NS, MaxGen=MaxGen)
    cfg.SADE.Fmin = Fmin
    cfg.SADE.Fmax = Fmax
    cfg.SADE.CRmin = CRmin
    cfg.SADE.CRmax = CRmax
    cfg.SADE.UseConvergence = use_conv
    cfg.SADE.Tol = tol
    cfg.SADE.StagnationMax = stagn
    cfg.Seed = seed
    return cfg


def _all_designs(NP: int, ND: int):
    return itertools.product(range(ND), repeat=NP)


def _cache_with_cost(NP: int, ND: int, cost_fn):
    """Full cache: every design → (cost_fn(design), 0.0, True)."""
    return {
        tuple(design): (float(cost_fn(design)), 0.0, True)
        for design in _all_designs(NP, ND)
    }


# ---------------------------------------------------------------------------
# Convergence: the core promise of the algorithm
# ---------------------------------------------------------------------------

def test_sade_converges_to_global_minimum():
    """Classic sanity: convex landscape, find the global minimum.

    Cost = sum(indices).  Global min is [0, 0, 0, 0] with cost 0.
    Three seeds; every run must find the global min and be feasible.
    If jrand is removed (or any of the paper's mechanisms regresses),
    convergence slows enough that at least one seed is likely to miss
    the minimum within the (generous) 300-generation budget.
    """
    NP, ND = 4, 4                               # 4 pipes × 4 options
    problem = _problem(NP, ND)
    cache = _cache_with_cost(NP, ND, lambda d: sum(d))

    for seed in (1, 2, 3):
        cfg = _sade_cfg(NS=30, MaxGen=300, seed=seed)
        cfg.Cache = cache
        score, position, convergence, feasible = run_sade(problem, cfg)
        assert feasible is True, f"seed {seed}: returned an infeasible design"
        assert score == pytest.approx(0.0), (
            f"seed {seed}: SADE did not converge to the global minimum "
            f"(cost={score}); expected 0.0"
        )
        # Every pipe should be at index 0 → 0.1 m in this synthetic setup
        np.testing.assert_allclose(position, 0.1, atol=1e-9)


# ---------------------------------------------------------------------------
# Cv-based termination (Eq. 6)
# ---------------------------------------------------------------------------

def test_sade_stops_when_cv_below_tol():
    """A convergence-enabled run stops before MaxGen once Cv < Tol."""
    NP, ND = 3, 3
    problem = _problem(NP, ND)
    # Trivial landscape: every design has the same cost → Cv = 0 from
    # generation 1, well below any reasonable Tol.
    cache = {tuple(d): (100.0, 0.0, True)
             for d in _all_designs(NP, ND)}
    cfg = _sade_cfg(NS=20, MaxGen=1000, seed=5, use_conv=True, tol=1e-3)
    cfg.Cache = cache

    meta = {}
    score, position, convergence, feasible = run_sade(
        problem, cfg, meta=meta,
    )
    generations = meta["extra"]["generations"]
    assert 0 < generations < 1000, (
        f"SADE ran {generations} generations (MaxGen=1000); with "
        f"UseConvergence=True and Cv=0 from generation 1, it should "
        f"have stopped early."
    )
    assert meta["extra"]["converged"] is True
    assert meta["extra"]["stopped_by_stagnation"] is False


def test_sade_use_conv_false_runs_more_generations():
    """Sanity: disabling UseConvergence lets the algorithm run to the cap
    (or to the stagnation cap, if set)."""
    NP, ND = 3, 3
    problem = _problem(NP, ND)
    cache = {tuple(d): (100.0, 0.0, True)
             for d in _all_designs(NP, ND)}
    cfg = _sade_cfg(NS=10, MaxGen=40, seed=3, use_conv=False, stagn=0)
    cfg.Cache = cache

    meta = {}
    run_sade(problem, cfg, meta=meta)
    # Without convergence or stagnation, the loop runs to MaxGen.
    assert meta["extra"]["generations"] == 40
    assert meta["extra"]["converged"] is False
    assert meta["extra"]["stopped_by_stagnation"] is False


# ---------------------------------------------------------------------------
# StagnationMax safeguard
# ---------------------------------------------------------------------------

def test_sade_stagnation_safeguard_terminates():
    """With UseConvergence=False and StagnationMax=N, the run should
    terminate after ~N generations without improvement, not at MaxGen."""
    NP, ND = 3, 3
    problem = _problem(NP, ND)
    # Uniform costs → no improvement is ever possible → the stall
    # counter climbs 1 per generation.
    cache = {tuple(d): (100.0, 0.0, True)
             for d in _all_designs(NP, ND)}
    cfg = _sade_cfg(
        NS=10, MaxGen=1000, seed=1,
        use_conv=False, stagn=20,
    )
    cfg.Cache = cache

    meta = {}
    run_sade(problem, cfg, meta=meta)
    gens = meta["extra"]["generations"]
    assert 18 <= gens <= 25, (
        f"SADE ran {gens} generations; with StagnationMax=20 and no "
        f"possible improvement, it should have stopped after ~20-21 "
        f"generations (loop = initial + 20 stall gens)."
    )
    assert meta["extra"]["stopped_by_stagnation"] is True
    assert meta["extra"]["converged"] is False


# ---------------------------------------------------------------------------
# Constraint handling (Deb's tournament)
# ---------------------------------------------------------------------------

def test_sade_returns_feasible_when_cheapest_design_is_infeasible():
    """Deb's tournament should never return an infeasible design when a
    feasible one exists, even if the infeasible one is cheaper."""
    NP, ND = 3, 3
    problem = _problem(NP, ND)

    cache = {}
    for d in _all_designs(NP, ND):
        if all(x == 0 for x in d):
            cache[tuple(d)] = (1.0, 1.0, False)         # cheap, infeasible
        elif all(x == 1 for x in d):
            cache[tuple(d)] = (10.0, 0.0, True)         # feasible, best
        elif all(x == 2 for x in d):
            cache[tuple(d)] = (20.0, 0.0, True)         # feasible, worse
        else:
            cache[tuple(d)] = (100.0, 0.5, False)       # infeasible, mid

    cfg = _sade_cfg(NS=30, MaxGen=300, seed=7)
    cfg.Cache = cache

    score, position, convergence, feasible = run_sade(problem, cfg)
    assert feasible is True, (
        "SADE returned an infeasible design despite a feasible one "
        "existing — Deb's constraint tournament has regressed."
    )
    assert score == pytest.approx(10.0), (
        f"SADE should converge on the cheapest *feasible* design "
        f"(cost 10.0); got {score}.  The cheaper design (cost 1.0) is "
        f"infeasible and must not be returned."
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_same_result():
    """With a fixed seed and pre-populated cache, SADE is deterministic."""
    NP, ND = 3, 3
    problem = _problem(NP, ND)
    cache = _cache_with_cost(NP, ND, lambda d: sum(d))
    cfg = _sade_cfg(NS=20, MaxGen=200, seed=99)
    cfg.Cache = cache

    r1 = run_sade(problem, cfg)
    r2 = run_sade(problem, cfg)
    assert r1[0] == r2[0]                          # same score
    np.testing.assert_array_equal(r1[1], r2[1])    # same position
    np.testing.assert_array_equal(r1[2], r2[2])    # same convergence
    assert r1[3] == r2[3]                          # same feasible


# ---------------------------------------------------------------------------
# Position reconstruction
# ---------------------------------------------------------------------------

def test_returned_position_has_correct_shape():
    NP, ND = 5, 4
    problem = _problem(NP, ND)
    cache = _cache_with_cost(NP, ND, lambda d: sum(d))
    cfg = _sade_cfg(NS=15, MaxGen=50, seed=1)
    cfg.Cache = cache

    score, position, convergence, feasible = run_sade(problem, cfg)
    assert position.shape == (NP,)
    # Every returned diameter should be one of the ND options
    for i in range(NP):
        assert any(np.isclose(position[i], d) for d in problem["D"])