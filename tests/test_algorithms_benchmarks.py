"""Benchmark verification tests.

Runs each algorithm on the standard WDS benchmarks (Hanoi, two-loop)
and asserts the result is within a documented tolerance of the
published optimum.

These tests are marked ``@pytest.mark.slow`` because they take
~10-15 minutes.  They are excluded by default; run them with:

    pytest tests/test_algorithms_benchmarks.py -v -m slow

Reference choice
----------------
For each algorithm, the reference is the paper with the SAME CLASS of
algorithm, not the globally-best result:

- GA is a standard integer-coded GA with tournament/roulette selection,
  one-point crossover, uniform-reset mutation, and elitism.  It is
  closest in spirit to Simpson et al. (1994)'s Simple GA, which
  reports $6.112M on Hanoi.  It is NOT a faithful implementation of
  either Simpson (binary coding + bitwise mutation) or Dandy's
  Improved GA (Gray codes + adjacency mutation + variable power
  scaling).  Comparing GA to Suribabu (2010)'s $6.081M would be
  comparing across algorithm classes.
- PSO is standard global-best PSO with constant inertia.  Reference:
  $6.081M benchmark optimum.
- DE is standard DE/rand/1/bin.  Reference: $6.081M.
- SADE is a paper-faithful reproduction of Zheng et al. (2013); its
  reference is also $6.081M.

Tolerances reflect baseline observed performance (3 seeds, seed=42;
see scripts/baseline_benchmarks.py) plus a safety margin:

    Hanoi:
        GA    9.9% baseline (vs Simpson $6.112M) → tolerance 15.0%
        PSO   2.0% baseline (vs $6.081M)         → tolerance  5.0%
        DE    4.8% baseline (vs $6.081M)         → tolerance 10.0%
        SADE  1.8% baseline (vs $6.081M)         → tolerance  5.0%

    Two-loop (all compare to $419,000, Alperovits & Shamir 1977):
        GA    0.95% baseline → tolerance 3.0%
        PSO   1.19% baseline → tolerance 3.0%
        DE    0.00% baseline → tolerance 2.0%
        SADE  0.00% baseline → tolerance 2.0%

If a test here fails after a code change, the change regressed the
algorithm.  If it fails on a clean checkout, see docs/algorithms.md
for the algorithm's documented reference and known limitations.

Do NOT relax a tolerance without updating docs/algorithms.md.  A
failing verification test means one of:
  (a) a real regression (investigate),
  (b) the algorithm's expected performance drifted (update
      docs/algorithms.md and explain why),
  (c) the baseline changed (regenerate baseline_benchmarks.py).
"""
from pathlib import Path

import pytest

from wds_optimizer.api import OptimizationRequest, run_optimization


BENCH = Path(__file__).resolve().parent.parent / "benchmarks"

TWOLOOP_PAPER_BEST = 419_000.0    # Alperovits & Shamir (1977)


# Hanoi cases: (algo, reference_best, tolerance_pct, note)
HANOI_CASES = [
    ("GA",   6.112e6, 15.0,
     "reference is Simpson et al. (1994) Simple GA - same class of "
     "algorithm; see docs/algorithms.md"),
    ("PSO",  6.081e6,  5.0,
     "reference is Suribabu (2010); standard benchmark optimum"),
    ("DE",   6.081e6, 10.0,
     "reference is Suribabu (2010); standard benchmark optimum"),
    ("SADE", 6.081e6,  5.0,
     "reference is Suribabu (2010); SADE reproduces the benchmark "
     "optimum in Zheng et al. (2013)"),
]

# Two-loop cases: (algo, tolerance_pct)
TWOLOOP_CASES = [
    ("GA",   3.0),
    ("PSO",  3.0),
    ("DE",   2.0),
    ("SADE", 2.0),
]


def _skip_if_benchmark_missing(name: str) -> None:
    """Skip when benchmarks/ is absent (e.g. shallow clone or LFS)."""
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


def _twoloop_constraints():
    return _hanoi_constraints()


@pytest.mark.slow
@pytest.mark.parametrize(
    "algo,reference_best,tolerance_pct,note", HANOI_CASES,
)
def test_hanoi(algo, reference_best, tolerance_pct, note):
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

    gap_pct = 100 * (best - reference_best) / reference_best
    assert gap_pct <= tolerance_pct, (
        f"{algo} on Hanoi: ${best:,.0f} is {gap_pct:+.1f}% above "
        f"reference ${reference_best:,.0f} "
        f"(tolerance {tolerance_pct}%). {note}. "
        f"See docs/algorithms.md for expected behaviour."
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo,tolerance_pct", TWOLOOP_CASES)
def test_twoloop(algo, tolerance_pct):
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
        f"reference ${TWOLOOP_PAPER_BEST:,.0f} "
        f"(tolerance {tolerance_pct}%)."
    )