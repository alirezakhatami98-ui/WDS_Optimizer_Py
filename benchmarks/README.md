# Benchmarks

Reference networks and published best-known solutions, used to verify
each algorithm's correctness.

Each subfolder contains a standard `.inp` file, a candidate-diameter
file (`D.txt`), and a unit-cost file (`Cost.txt`), plus a README
citing the paper(s) that introduced the problem.

## Available benchmarks

| Network | Pipes | Junctions | Best-known cost | Reference |
|---------|-------|-----------|-----------------|-----------|
| `hanoi` | 34 | 32 | $6.081M | Fujiwara & Khang 1990 |
| `nytp`  | 21 | 16 | $38.64M | Dandy et al. 1996 |
| `anytown` | 37 | 19 | (varies) | Walski et al. 1987 |

Additional networks (Balerma, Fossolo, Pescara) may be added later.

## Why benchmark files live here, not in the package

The `.inp` files include data from academic publications whose
redistribution terms are not always clear.  Keeping them in the
source repository (not in the pip package) makes the provenance
explicit, respects the original authors, and keeps `pip install
wds-optimizer` small (a headless install never needs them).

The built-in `demo` benchmark (a 3-pipe synthetic network) is the
only one embedded in the package, for use by `wds-optimize benchmark
--network demo` in tutorials and tests.

## How to use

Point the CLI at any file with `--inp`, `--d`, `--cost`, and the
correct `--diam-unit` for that network (each README states it):

```bash
wds-optimize run \
    --inp benchmarks/hanoi/hanoi.inp \
    --d   benchmarks/hanoi/D.txt \
    --cost benchmarks/hanoi/Cost.txt \
    --diam-unit in \
    --algo SADE --pmin 30 --vmax 2.5
```

Or use the Python API:

```python
from wds_optimizer import OptimizationRequest, run_optimization
```

See each subfolder's README for the exact reproduction command and
the published cost that the run should reach.

## Verification status

`tests/test_algorithms_benchmarks.py` (marked `@pytest.mark.slow`)
runs each algorithm on these networks and asserts that the result is
within tolerance of the published optimum.  A failure there means a
real regression in the algorithm, not a flaky test — do not weaken
the tolerance without a paper citation that justifies it.