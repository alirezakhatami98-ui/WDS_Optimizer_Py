# Two-loop network

The two-loop network is the smallest standard WDS design benchmark.
Six junctions are connected in two loops fed by a single reservoir
(8 pipes total).  Despite its size it is the most frequently used
test case in the WDS optimization literature because it exercises
topology (loops) with a well-established optimum.

## Source

First introduced in:

> Alperovits, E., and Shamir, U. (1977). "Design of optimal water
> distribution systems." *Water Resources Research*, 13(6), 885-900.

Reproduced and solved by GAs in:

> Simpson, A. R., Dandy, G. C., and Murphy, L. J. (1994). "Genetic
> algorithms compared to other techniques for pipe optimization."
> *Journal of Water Resources Planning and Management*, 120(4),
> 423-443.

Used as a benchmark in dozens of subsequent papers (ACO, PSO, DE,
SADE, harmony search, shuffled frog leaping, ...).

## Files

| File | Content | Unit |
|------|---------|------|
| `twoloop.inp` | EPANET network definition | SI (LPS) |
| `D.txt` | 14 candidate diameters | **inches** |
| `Cost.txt` | Unit cost per metre, same order as `D.txt` | USD/m |

**Diameter unit:** `D.txt` is in **inches** (`--diam-unit in`).
The `.inp` uses LPS and its pipe diameters are in mm.

Diameters (in → m):
1 → 0.0254 | 2 → 0.0508 | 3 → 0.0762 | 4 → 0.1016
6 → 0.1524 | 8 → 0.2032 | 10 → 0.2540 | 12 → 0.3048
14 → 0.3556 | 16 → 0.4064 | 18 → 0.4572 | 20 → 0.5080
22 → 0.5588 | 24 → 0.6096

## Published best-known cost

**$419,000**, reported in:

> Alperovits, E., and Shamir, U. (1977), *op. cit.*

This value has been reproduced by every credible metaheuristic in the
WDS literature; it is the single most-verified optimum in the field.

## Constraint settings

**Pmin = 30 m, and nothing else.**  Same as Hanoi: no velocity
constraint, no maximum pressure, no minimum velocity.

| Constraint | Value |
|------------|-------|
| Pmin | **30 m** (enabled) |
| Pmax | disabled |
| Vmin | disabled |
| Vmax | disabled |

Reservoir head = **210 m**; highest junction elevation = **165 m**;
available head = 45 m at the highest junction, of which 30 m is
reserved for pressure and 15 m can be lost to friction.  Feasible
solutions exist (the published optimum is one).

## Reproduction

```bash
wds-optimize run \
    --inp benchmarks/twoloop/twoloop.inp \
    --d   benchmarks/twoloop/D.txt \
    --cost benchmarks/twoloop/Cost.txt \
    --diam-unit in \
    --algo SADE --ns 100 --max-gen 300 --seed 1 \
    --pmin 30 \
    --output twoloop_sade.json
```

Expected: feasible design, cost ≤ $420,000.

## Verification status

Tracked by `tests/test_algorithms_benchmarks.py`:

- GA — pending
- PSO — pending
- DE — pending
- SADE — pending

Note: because two-loop is so small (8 pipes, 14 diameters = 14^8 ≈
1.5 billion possible designs), the true optimum can be found by
exhaustive search on a laptop in a few minutes.  If any algorithm
fails to reach $419,000, the algorithm has a bug — this benchmark is
far too easy to miss the optimum by accident.