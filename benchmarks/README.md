# Benchmarks

Reference networks and published best-known solutions used to verify
each algorithm's correctness.

Each subfolder contains a standard `.inp` file, a candidate-diameter
file (`D.txt`), and a unit-cost file (`Cost.txt`), plus a README citing
the paper(s) that introduced the problem.

## Available benchmarks

| Network | Pipes | Junctions | Best-known cost | Constraint | Reference |
|---------|-------|-----------|-----------------|------------|-----------|
| `hanoi`   | 34 | 32 | $6.081M  | Pmin = 30 m | Fujiwara & Khang (1990); Simpson et al. (1994) |
| `nytp`    | 21 | 16 | $38.64M  | Head-based (see its README) | Dandy et al. (1996) |
| `twoloop` |  8 |  6 | $419,000 | Pmin = 30 m | Alperovits & Shamir (1977) |

All three networks are **single-objective, cost-minimization**
problems. None of them enforces a velocity constraint in the
literature; the optimal solutions naturally have high velocities in a
few pipes. Adding Vmax would render them infeasible.

## Why benchmark files live here, not in the package

The `.inp` data is redistributed from academic publications whose
terms are not always clear. Keeping it in the source repository (not
in `pip install wds-optimizer`) makes the provenance explicit and
respects the original authors. The built-in `demo` benchmark (a
3-pipe synthetic network) is the only one embedded in the package.

## Constraint settings for each network

**Always start from the constraint settings in each subfolder's
README.** In particular:

- **Hanoi and Two-loop:** Pmin = 30 m is the only active constraint.
  Disable Pmax, Vmin, Vmax.
- **NYTP:** Uses a head-based constraint that is not directly
  representable as a single Pmin value (see `nytp/README.md`). Uses
  a per-node equivalent; do not change without reading the README.

## Verification status

Tracked by `tests/test_algorithms_benchmarks.py` (marked
`@pytest.mark.slow`):

| Algorithm | Hanoi | NYTP | Two-loop |
|-----------|-------|------|----------|
| GA        | pending | pending | pending |
| PSO       | pending | pending | pending |
| DE        | pending | pending | pending |
| SADE      | pending | pending | pending |

A row becomes ✔ when the algorithm reproduces the published cost
within a documented tolerance.  "Pending" means the test exists but
the algorithm's current defaults have not been verified against the
paper yet — see docs/algorithms.md for the mapping.