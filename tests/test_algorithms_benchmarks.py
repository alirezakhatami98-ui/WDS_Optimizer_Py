"""Benchmark verification tests.

Runs each algorithm on the standard WDS benchmarks (Hanoi, two-loop)
and asserts the result is within a documented tolerance of the
published optimum.

These tests are marked ``@pytest.mark.slow`` because they take
~10-15 minutes.  They are excluded by default; run them with:

    pytest tests/test_algorithms_benchmarks.py -v -m slow

Tolerances reflect current observed performance plus safety margin.
They are NOT tight equality checks.  The published optimum is from
many runs with tuned parameters; we use 3 fixed-seed runs.  Tight
tolerances would fail on normal run-to-run variance.

If a test here fails after a code change, the change regressed the
algorithm.  If it fails on a clean checkout, see docs/algorithms.md
for the algorithm's documented reference and known limitations.

Tolerance derivation (baseline run: 3 seeds, seed=42, see
scripts/baseline_benchmarks.py):

    Hanoi (paper best $6.081M, Suribabu 2010):
        GA   10.5% baseline → tolerance 20.0% (simple GA, wide variance)
        PSO   2.0% baseline → tolerance  5.0%
        DE    4.8% baseline → tolerance 10.0%
        SADE  1.8% baseline → tolerance  5.0%

    Two-loop (paper best $419,000, Alperovits & Shamir 1977):
        GA    0.95% baseline → tolerance 3.0%
        PSO   1.19% baseline → tolerance 3.0%
        DE    0.00% baseline → tolerance 2.0%
        SADE  0.00% baseline → tolerance 2.0%
"""
from pathlib import Path

import pytest

from wds_optimizer.api import OptimizationRequest, run_optimization


BENCH = Path(__file__).resolve().parent.parent / "benchmarks"

HANOI_PAPER_BEST   = 6.081e6      # Suribabu (2010)
TWOLOOP_PAPER_BEST = 419_000.0    # Alperovits & Shamir (1977)

HANOI_CASES = [
    # (algo, tolerance_pct, note)
    ("GA",   20.0, "Simple GA (Simpson et al. 1994 formulation)"),
    ("PSO",   5.0, "Suribabu & Neelakantan (2006) formulation"),
    ("DE",   10.0, "Suribabu (2010) formulation"),
    ("SADE",  5.0, "Zheng, Zecchin & Simpson (2013) formulation"),
]

TWOLOOP_CASES = [
    ("GA",    3.0, ""),
    ("PSO",   3.0, ""),
    ("DE",    2.0, ""),
    ("SADE",  2.0, ""),
]


def _skip_if_benchmark_missing(name: str) -> None:
    """Skip when benchmarks/ is absent (e.g. shallow clone or LFS)."""
    if not (BENCH / name / f"{name}.inp").exists():
        pytest.skip(
            f"benchmarks/{name}/ not present — benchmark files live in "
            f"the source repository, not in the pip package"
        )


def _hanoi_constraints():
    return {
        "Pmin": {"enabled": True,  "value": 30.0},
        "Pmax": {"enabled": False, "value": 100.0},
        "Vmin": {"enabled": False, "value": 0.0},
        "Vmax": {"enabled": False, "value": 2.5},
    }


def _twoloop_constraints():
    # Same as Hanoi: only Pmin is active in the reference formulation.
    return _hanoi_constraints()


@pytest.mark.slow
@pytest.mark.parametrize("algo,tolerance_pct,note", HANOI_CASES)
def test_hanoi(algo, tolerance_pct, note):
    _skip_if_benchmark_missing("hanoi")

    req = OptimizationRequest(
        inp_path=str(BENCH / "hanoi" / "hanoi.inp"),
        d_path=str(BENCH / "hanoi" / "D.txt"),
        cost_path=str(BENCH / "hanoi" / "Cost.txt"),
        diam_unit="in",
        algorithms=[algo], n_runs=3, seed=42,
        ns=200, max_gen=300,
        constraints=_hanoi_constraints(),
    )
    result = run_optimization(req)

    best = result.best_score
    assert best > 0, f"{algo}: no feasible design produced"

    gap_pct = 100 * (best - HANOI_PAPER_BEST) / HANOI_PAPER_BEST
    assert gap_pct <= tolerance_pct, (
        f"{algo} on Hanoi: ${best:,.0f} is {gap_pct:+.1f}% above "
        f"paper best ${HANOI_PAPER_BEST:,.0f} "
        f"(tolerance {tolerance_pct}%). "
        f"Reference: {note}. "
        f"See docs/algorithms.md for expected behaviour."
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo,tolerance_pct,note", TWOLOOP_CASES)
def test_twoloop(algo, tolerance_pct, note):
    _skip_if_benchmark_missing("twoloop")

    req = OptimizationRequest(
        inp_path=str(BENCH / "twoloop" / "twoloop.inp"),
        d_path=str(BENCH / "twoloop" / "D.txt"),
        cost_path=str(BENCH / "twoloop" / "Cost.txt"),
        diam_unit="in",
        algorithms=[algo], n_runs=3, seed=42,
        ns=100, max_gen=200,
        constraints=_twoloop_constraints(),
    )
    result = run_optimization(req)

    best = result.best_score
    assert best > 0, f"{algo}: no feasible design produced"

    gap_pct = 100 * (best - TWOLOOP_PAPER_BEST) / TWOLOOP_PAPER_BEST
    assert gap_pct <= tolerance_pct, (
        f"{algo} on two-loop: ${best:,.0f} is {gap_pct:+.1f}% above "
        f"paper best ${TWOLOOP_PAPER_BEST:,.0f} "
        f"(tolerance {tolerance_pct}%)."
    )