# Hanoi network

The Hanoi network is one of the two canonical WDS design benchmarks
(the other being the New York Tunnels Problem).  It has a single
reservoir feeding a branched-then-looped layout of 34 pipes and 32
demand junctions.

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

The specific `.inp` file in this folder is sourced from the MINLP
benchmark dataset distributed with:

> Bragalli, C., D'Ambrosio, C., Lee, J., Lodi, A., and Toth, P.
> (2008). "Water network design by MINLP." IBM Research Report
> RC24495.  Also: *Optimization and Engineering* 13, 55-79 (2012).

The data in all three sources is identical; the citation set above
is provided so the file can be traced to any of them.

## Files

| File | Content | Unit |
|------|---------|------|
| `hanoi.inp` | EPANET network definition | SI (LPS) |
| `D.txt` | 6 candidate diameters, one per line | **inches** |
| `Cost.txt` | Unit cost per metre, same order as D.txt | USD/m |

**Note the diameter unit.** `D.txt` lists diameters in inches.  The
CLI needs `--diam-unit in`; the API needs `diam_unit="in"`.  The GUI
selects it from the sidebar.  The pipe diameters inside `hanoi.inp`
are already in metres (converted by EPANET from the LPS unit system),
so no conversion is needed there.

Diameters (inches → metres):
  12 → 0.3048 | 16 → 0.4064 | 20 → 0.5080
  24 → 0.6096 | 30 → 0.7620 | 40 → 1.0160

## Published best-known cost

**$6.081 million** (feasible), reported in:

> Suribabu, C. R. (2010). "Differential evolution algorithm for
> optimal design of water distribution networks." *Journal of
> Hydroinformatics*, 12(1), 66-82.

Prior to Suribabu, the best-known value was $6.112 million
(Simpson et al. 1994).  Most modern metaheuristics (GA, PSO, DE,
SADE, ACO) converge on $6.081M within a few thousand evaluations.

## Constraint values

- **Pmin = 30 m** at every junction.  **This is the only constraint
  used by the standard Hanoi benchmark.**
- **Pmax, Vmin, Vmax are all disabled.**

### Why Hanoi has no velocity constraint

This is a distinctive feature of the Hanoi benchmark that is often
missed.  The total demand is ≈ 5540 L/s.  The first pipe (from the
reservoir) carries the entire demand.  Even with the largest
available diameter (1016 mm), the velocity in pipe 1 is ≈ 6.8 m/s.
Any Vmax below this value makes the problem infeasible by
construction — no pipe size in D.txt can satisfy it.

All reference papers (Fujiwara & Khang 1990; Simpson et al. 1994;
Suribabu 2010; Zheng et al. 2013) therefore optimize Hanoi with
**Pmin only**.  A run that disables velocity is not a weaker test —
it is the correct test.

### Reproduction

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

Expected: feasible design, cost ≤ $6.10 million. Do not pass
--vmax — it will make the run infeasible regardless of algorithm.

Via the Python API:

```python
from wds_optimizer import OptimizationRequest, run_optimization

result = run_optimization(OptimizationRequest(
    inp_path="benchmarks/hanoi/hanoi.inp",
    d_path="benchmarks/hanoi/D.txt",
    cost_path="benchmarks/hanoi/Cost.txt",
    diam_unit="in",
    algorithms=["SADE"], n_runs=10, seed=1, ns=200, max_gen=500,
    constraints={
        "Pmin": {"enabled": True,  "value": 30.0},
        "Pmax": {"enabled": False, "value": 100.0},
        "Vmin": {"enabled": False, "value": 0.0},
        "Vmax": {"enabled": False,  "value": 2.5},
    },
))
print(f"{result.best_score:,.0f}  feasible={result.best_feasible}")
```

## Verification status

Tracked by `tests/test_algorithms_benchmarks.py`:

- GA (Simpson 1994 implementation) — *pending verification*
- PSO (Suribabu 2006 implementation) — *pending verification*
- DE (Suribabu 2010 implementation) — *pending verification*
- SADE (Zheng 2013 implementation) — *pending verification*