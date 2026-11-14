"""Tests for DE_suribabu_2010 (paper-faithful DE).

Covers two areas:

1. **Unit tests** on a tiny synthetic 3-pipe problem where every
   design has a known cost and feasibility.  No hydraulics.  Verify
   the return contract, the penalty behavior, determinism, and
   progress callback.

2. **Benchmark tests** (marked slow) that reproduce Suribabu (2010)
   on Hanoi and two-loop using the paper's high-setting runs
   (NS=100, MaxGen=1000).  Tolerances reflect the paper's reported
   variance (2% for Hanoi, 1% for two-loop).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from wds_optimizer.algorithms.config import DEConfig, build_algorithm_config
from wds_optimizer.algorithms.de_suribabu_2010 import run_de_suribabu_2010
from wds_optimizer.api import OptimizationRequest, run_optimization


BENCH = Path(__file__).resolve().parent.parent / "benchmarks"


# =====================================================================
#  Synthetic unit tests
# =====================================================================

def _synthetic_problem() -> dict:
    """3 pipes, 3 diameter options.  No hydraulics."""
    return {
        "wn": None,
        "_solver": False,
        "NP": 3,
        "VariablePipes": np.array([0, 1, 2]),
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
        "Units": {
            "System": "SI", "PressureUnit": "m",
            "VelocityUnit": "m/s", "LengthUnit": "m",
            "DiameterUnit": "mm",
        },
    }


def _synthetic_cache() -> dict:
    """All 27 designs, feasible iff sum of indices >= 2."""
    unit = [1.0, 10.0, 100.0]
    cache = {}
    for design in itertools.product(range(3), repeat=3):
        cost = sum(100.0 * unit[i] for i in design)
        feas = sum(design) >= 2
        viol = 0.0 if feas else 1.0
        cache[design] = (cost, viol, feas)
    return cache


def _make_config(*, seed=42, NS=20, MaxGen=200, F=0.8, CR=0.4):
    cfg = build_algorithm_config(NS=NS, MaxGen=MaxGen)
    cfg.DE.F = F
    cfg.DE.CR = CR
    cfg.Seed = seed
    return cfg


def test_de_suribabu_2010_returns_proper_tuple():
    problem = _synthetic_problem()
    cfg = _make_config()
    cfg.Cache = _synthetic_cache()
    result = run_de_suribabu_2010(problem, cfg)
    assert len(result) == 4
    score, pos, conv, feas = result
    assert isinstance(score, float)
    assert pos.shape == (3,)
    assert conv.shape == (200,)
    assert isinstance(feas, bool)


def test_de_suribabu_2010_finds_feasible_optimum():
    problem = _synthetic_problem()
    cfg = _make_config()
    cfg.Cache = _synthetic_cache()
    score, pos, conv, feas = run_de_suribabu_2010(problem, cfg)
    assert feas is True
    # Cheapest feasible: sum-of-indices >= 2, minimum cost = 2100
    #   [1,1,0], [1,0,1], [0,1,1]  →  1000 + 1000 + 100 = 2100
    assert score == pytest.approx(2100.0, rel=1e-6), (
        f"expected cost 2100, got {score}"
    )


def test_de_suribabu_2010_uses_penalty_when_no_feasible():
    """If no feasible design exists, returns max_cost (30000)."""
    problem = _synthetic_problem()
    cache = {}
    for design in itertools.product(range(3), repeat=3):
        cost = sum(100.0 * [1.0, 10.0, 100.0][i] for i in design)
        cache[design] = (cost, 1.0, False)
    cfg = _make_config()
    cfg.Cache = cache
    score, pos, conv, feas = run_de_suribabu_2010(problem, cfg)
    assert feas is False
    # max_cost = sum(L) * Cost[-1] = 300 * 100 = 30000
    assert score == pytest.approx(30000.0)


def test_de_suribabu_2010_deterministic():
    problem = _synthetic_problem()
    cfg = _make_config(seed=7)
    cfg.Cache = _synthetic_cache()
    r1 = run_de_suribabu_2010(problem, cfg)
    r2 = run_de_suribabu_2010(problem, cfg)
    assert r1[0] == r2[0]
    np.testing.assert_array_equal(r1[1], r2[1])
    np.testing.assert_array_equal(r1[2], r2[2])
    assert r1[3] == r2[3]


def test_de_suribabu_2010_progress_callback():
    events = []
    problem = _synthetic_problem()
    cfg = _make_config()
    cfg.Cache = _synthetic_cache()
    run_de_suribabu_2010(problem, cfg, progress_callback=events.append)
    assert len(events) >= 1
    e0 = events[0]
    for key in ("Algorithm", "Generation", "BestCost", "Feasible",
                "Convergence", "NFE"):
        assert key in e0, f"missing key {key!r}"
    assert e0["Algorithm"] == "DE_suribabu_2010"


def test_de_suribabu_2010_raises_on_bad_config():
    problem = _synthetic_problem()
    # NP < 4
    with pytest.raises(ValueError, match="at least 4"):
        run_de_suribabu_2010(
            problem, _make_config(NS=2),
        )
    # ND < 2
    bad = _synthetic_problem()
    bad["D"] = np.array([0.1])
    with pytest.raises(ValueError, match="at least 2"):
        run_de_suribabu_2010(bad, _make_config())


# =====================================================================
#  Benchmark verification (marked slow)
# =====================================================================

def _skip_if_missing(name: str) -> None:
    if not (BENCH / name / f"{name}.inp").exists():
        pytest.skip(
            f"benchmarks/{name}/ not present - benchmark files live in "
            f"the source repository, not in the pip package"
        )


def _hanoi_constraints():
    return {
        "Pmin": {"enabled": True,  "value": 30.0},
        "Pmax": {"enabled": False, "value": 100.0},
        "Vmin": {"enabled": False, "value": 0.0},
        "Vmax": {"enabled": False, "value": 2.5},
    }


@pytest.mark.slow
def test_de_suribabu_2010_hanoi_paper_settings():
    """Paper's high-setting run: NS=100, MaxGen=1000, F=0.8, CR=0.5.

    Per the paper (Table 6), this setting reaches $6.081M in every
    trial within the first few generations of the run.  Tolerance
    2% accounts for run-to-run variance with our 5-seed budget.
    """
    _skip_if_missing("hanoi")
    req = OptimizationRequest(
        inp_path=str(BENCH / "hanoi" / "hanoi.inp"),
        d_path=str(BENCH / "hanoi" / "D.txt"),
        cost_path=str(BENCH / "hanoi" / "Cost.txt"),
        diam_unit="in",
        algorithms=["DE_suribabu_2010"], n_runs=5, seed=42,
        ns=100, max_gen=1000,
        constraints=_hanoi_constraints(),
        de=DEConfig(F=0.8, CR=0.5),
    )
    result = run_optimization(req)
    assert result.best_feasible, "no feasible design produced"
    gap_pct = 100 * (result.best_score - 6.081e6) / 6.081e6
    assert gap_pct <= 2.0, (
        f"DE_suribabu_2010 on Hanoi: ${result.best_score:,.0f} is "
        f"{gap_pct:+.1f}% above paper best $6.081M (tolerance 2%)"
    )


@pytest.mark.slow
def test_de_suribabu_2010_twoloop_paper_settings():
    """Paper's setting: NS=100, MaxGen=500, F=0.8, CR=0.4.

    The paper reports the optimum is reached at the first generation
    of every trial with these settings.  Tolerance 1%.
    """
    _skip_if_missing("twoloop")
    req = OptimizationRequest(
        inp_path=str(BENCH / "twoloop" / "twoloop.inp"),
        d_path=str(BENCH / "twoloop" / "D.txt"),
        cost_path=str(BENCH / "twoloop" / "Cost.txt"),
        diam_unit="in",
        algorithms=["DE_suribabu_2010"], n_runs=5, seed=42,
        ns=100, max_gen=500,
        constraints=_hanoi_constraints(),
        de=DEConfig(F=0.8, CR=0.4),
    )
    result = run_optimization(req)
    assert result.best_feasible
    gap_pct = 100 * (result.best_score - 419_000) / 419_000
    assert gap_pct <= 1.0, (
        f"DE_suribabu_2010 on two-loop: ${result.best_score:,.0f} is "
        f"{gap_pct:+.1f}% above paper best $419,000 (tolerance 1%)"
    )