# New York Tunnels Problem (NYTP)

The New York Tunnels Problem (also called the "New York City Water
Supply Tunnels" problem) is the other canonical WDS design
benchmark.  A single reservoir feeds 20 demand nodes connected by
21 pipes; each pipe must be either kept at its current diameter or
replaced with a larger one from a discrete set.

## Source

Standardized for GA benchmarking in:

> Dandy, G. C., Simpson, A. R., and Murphy, L. J. (1996). "An
> improved genetic algorithm for pipe network optimization." *Water
> Resources Research*, 32(2), 449-457.

The dataset used in this folder is the version distributed by
Dandy's research group and reproduced in:

> Bragalli, C., D'Ambrosio, C., Lee, J., Lodi, A., and Toth, P.
> (2008). "Water network design by MINLP." IBM Research Report
> RC24495.

## Files

| File | Content | Unit |
|------|---------|------|
| `nytp.inp` | EPANET network definition | US customary (CFS) |
| `D.txt` | 15 candidate diameters (36-204 in) | inches |
| `Cost.txt` | Unit cost per foot, same order as `D.txt` | USD/ft |

**Diameter unit:** `D.txt` is in **inches** (`--diam-unit in`).
The `.inp` file already uses CFS for flow, so diameters in it are
already in inches.

**Note on the MINLP dataset's "0" diameter:** the original
`D.txt`/`Cost.txt` from Bragalli et al. (2008) prepend a `0` line
representing "pipe not installed" (topology optimization).  Since
our algorithms do diameter selection only (every pipe must exist),
that line is **omitted** here.  The 15 diameters from 36 to 204
inches match the Dandy et al. (1996) standard.

Diameters (in → m):
36 → 0.9144 | 48 → 1.2192 | 60 → 1.5240 | 72 → 1.8288
84 → 2.1336 | 96 → 2.4384 | 108 → 2.7432 | 120 → 3.0480
132 → 3.3528 | 144 → 3.6576 | 156 → 3.9624 | 168 → 4.2672
180 → 4.5720 | 192 → 4.8768 | 204 → 5.1816

## Published best-known cost

**$38.64 million** (feasible), reported in:

> Maier, H. R., Simpson, A. R., Zecchin, A. C., Foong, W. K., Phang,
> K. Y., Seah, H. Y., and Tan, C. L. (2003). "Ant colony optimization
> for the design of water distribution systems." *Journal of Water
> Resources Planning and Management*, 129(3), 200-209.

First found in Dandy et al. (1996) as $38.80M, then improved by
subsequent methods.  The $38.64M value has been reproduced by ACO,
GA, SADE, PSO, and most modern metaheuristics.

## Constraint settings

**The published literature does not enforce a uniform minimum
pressure on NYTP.**  What the papers actually enforce is a
**minimum HEAD of 255 ft** at every node:

- Reservoir head = **300 ft** (91.44 m)
- Junction elevations = **255-272.8 ft** (77.7-83.2 m)
- Highest junction (node 17) has elevation 272.8 ft
- Constraint: head ≥ 255 ft at every node

Translated to per-node pressure, this means:

| Node  | Elevation (ft) | Min pressure = 255 - elev |
|-------|----------------|---------------------------|
| Most  | 255            | 0 ft  (0 m)               |
| 16    | 260            | -5 ft (-1.52 m)           |
| 17    | 272.8          | -17.8 ft (-5.43 m)        |

Negative pressure at nodes 16 and 17 is *mathematically allowed* by
the head constraint in the original formulation.  Our tool enforces a
**uniform** Pmin, so the equivalent constraint must use the most
permissive value:

```python
"Pmin": {"enabled": True,  "value": -5.43},   # equivalent to head >= 255 ft
"Vmax": {"enabled": False, "value": 2.5},     # not enforced in the papers
```

**⚠️ Important:** using Pmin = 30 m (which would be the standard
value for Hanoi or two-loop) makes NYTP **infeasible** because the
reservoir head is too low.  This is why NYTP is unusual and requires
its own README.

If your run reproduces $38.64M with a different Pmin value (for
example Pmin = 0 m when using the head constraint differently in your
reference implementation), **use that value** and update this README.
The point of the benchmark is reproducibility against the published
optimum, not against this specific Pmin translation.

## Reproduction

```bash
wds-optimize run \
    --inp benchmarks/nytp/nytp.inp \
    --d   benchmarks/nytp/D.txt \
    --cost benchmarks/nytp/Cost.txt \
    --diam-unit in \
    --algo SADE --ns 100 --max-gen 500 --seed 1 \
    --pmin -5.43 \
    --output nytp_sade.json
```

Expected: feasible design, cost ≤ $38.7 million.

## Verification status

Tracked by `tests/test_algorithms_benchmarks.py`:

- GA — pending (NYTP is typically harder for GA; Dandy 1996
  reported $38.80M at 80% success rate)
- PSO — pending
- DE — pending
- SADE — pending