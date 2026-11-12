# Hanoi network

The Hanoi network is one of the two canonical WDS design benchmarks
(the other being the New York Tunnels Problem).  A single reservoir
feeds a branched-then-looped layout of 34 pipes and 32 demand
junctions.

## Source

First introduced in:

> Fujiwara, O., and Khang, D. B. (1990). "A two-phase decomposition
> method for optimal design of looped water distribution networks."
> *Water Resources Research*, 26(4), 539-549.

Cost vector and diameter set standardized for GA benchmarking in:

> Simpson, A. R., Dandy, G. C., and Murphy, L. J. (1994). "Genetic
> algorithms compared to other techniques for pipe optimization."
> *Journal of Water Resources Planning and Management*, 120(4),
> 423-443.

The `.inp` file in this folder is sourced from the MINLP benchmark
dataset distributed with:

> Bragalli, C., D'Ambrosio, C., Lee, J., Lodi, A., and Toth, P.
> (2008). "Water network design by MINLP." IBM Research Report
> RC24495.

## Files

| File | Content | Unit |
|------|---------|------|
| `hanoi.inp` | EPANET network definition | SI (LPS) |
| `D.txt` | 6 candidate diameters | **inches** |
| `Cost.txt` | Unit cost per metre, same order as `D.txt` | USD/m |

**Note the diameter unit.** `D.txt` lists diameters in **inches**.
The CLI needs `--diam-unit in`; the API needs `diam_unit="in"`.
The `.inp` pipe diameters are already in metres (EPANET converts
from LPS).

Diameters (in → m):
12 → 0.3048 | 16 → 0.4064 | 20 → 0.5080 | 24 → 0.6096 | 30 → 0.7620 | 40 → 1.0160

## Published best-known cost

**$6.081 million** (feasible), reported in:

> Suribabu, C. R. (2010). "Differential evolution algorithm for
> optimal design of water distribution networks." *Journal of
> Hydroinformatics*, 12(1), 66-82.

Prior to Suribabu the best-known value was $6.112 million
(Simpson et al. 1994).

## Constraint settings

**Pmin = 30 m, and NOTHING ELSE.**

This is the standard formulation in the reference papers (Fujiwara &
Khang 1990; Simpson et al. 1994; Dandy et al. 1996; Suribabu 2010;
Zheng et al. 2013).  None of them enforces a velocity constraint.

**Do not enable Vmax.** The optimal Hanoi solution has a few pipes
whose velocities exceed 9 m/s; enforcing Vmax = 2.5 m/s renders the
problem infeasible (zero feasible designs after thousands of
evaluations).  The same is true for any other Vmax value below ~10
m/s.

| Constraint | Value |
|------------|-------|
| Pmin | **30 m** (enabled) |
| Pmax | disabled |
| Vmin | disabled |
| Vmax | **disabled** |

In the CLI:

```bash
--pmin 30
```

(no `--vmax`, no `--pmax`, no `--vmin`).

In the API:

```python
constraints={
    "Pmin": {"enabled": True,  "value": 30.0},
    "Pmax": {"enabled": False, "value": 100.0},
    "Vmin": {"enabled": False, "value": 0.0},
    "Vmax": {"enabled": False, "value": 2.5},
}
```

## Reproduction

```bash
wds-optimize run \
    --inp benchmarks/hanoi/hanoi.inp \
    --d   benchmarks/hanoi/D.txt \
    --cost benchmarks/hanoi/Cost.txt \
    --diam-unit in \
    --algo SADE --ns 200 --max-gen 500 --seed 1 \
    --pmin 30 \
    --output hanoi_sade.json
```

Expected: feasible design, cost ≤ $6.10 million.

## Verification status

Tracked by `tests/test_algorithms_benchmarks.py`:

- GA (Simpson 1994 implementation) — pending verification
- PSO (Suribabu 2006 implementation) — pending verification
- DE (Suribabu 2010 implementation) — pending verification
- SADE (Zheng 2013 implementation) — pending verification