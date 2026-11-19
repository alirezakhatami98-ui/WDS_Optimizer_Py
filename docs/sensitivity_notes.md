# Sensitivity analysis — notes and interpretation

`sensitivity_analysis.md` (auto-generated) shows raw sweep results.
This document explains what those results **do and do not** mean, and
how to read them.

## Current data status (v0.7.1)

- **Plan**: `--quick` (twoloop only)
- **Runs per point**: 3
- **MaxGen**: 200, **NS**: 100, **seed**: 42
- **Sweeps**: 8 (DE×2, GA×3, PSO×2, SADE×1)

**Quick-mode data is indicative, not publication-grade.**  With 3
runs, the standard error of the mean is roughly `0.58 × std`.  When
std is small ($1–3K) the mean is reliable; when std is large
($10–30K), three runs cannot distinguish values at the 5% level.

## What the data shows

### DE — `de_cr = 0.9` is a real win

`de_cr = 0.9` produces mean $419,333 with std $577.  Compare the
default `0.3`: mean $420,333, std $1,528.  The std reduction
matters: at 0.9, *every* run was near the optimum.

`de_f = 0.3` is slightly better than `0.5`, but the difference
($333) is far below SEM (~$1K).  Not actionable.

**Why this might not transfer to hanoi**: larger networks benefit
more from exploration, which CR=0.3 provides by combining
smaller portions of the mutant with the target.  Verify with
`--full` before adopting 0.9 as the default.

### GA — noisy, but some parameters clearly matter

`ga_pm = 0.01` has std $34,530 — one run hit $419,000 but others
were near $465,000.  Avoid.  `ga_pm = 0.1` is much more stable
(mean $430,000, std $14,000).

`ga_pc` and `ga_sel` are borderline.  `ga_pc = 0.7` has the best
mean; `ga_pc = 0.6` has the best single result but terrible
variance.  `roulette` beats `tournament` by roughly 1 SEM — not
conclusive at n=3.

**Big picture**: GA's mean on twoloop ($428K best case) is worse
than DE's ($419K) and PSO's ($422K) even after tuning.  This is
expected: our GA is a Simple GA, not Dandy's Improved GA (see
algorithms.md).  Do not expect GA to match DE/SADE on small
benchmarks regardless of parameters.

### PSO — classical defaults win

`pso_w = 0.5` with `pso_c1 = 2.0` produces mean $422,333 (std
$2,887 on c1=2.0; std $0 on w=0.5).  These match the classical
Kennedy & Eberhart (1995) defaults.

Our defaults (`w=0.7, c1=1.5`) are slightly worse on twoloop, but
tuned for hanoi.  Users targeting small networks can override.

### SADE — `sade_stag` sweep is uninformative

All three values (100, 300, 500) produce identical results:
mean $422,333, std $2,887.  This is not a "100 is best" finding;
it is a **flat line** — the parameter had no measurable effect.

Why: `MaxGen = 200` in the quick plan.  SADE likely found its best
early (≈20–30 gens) and stagnated.  The safeguard triggers after
`stag` generations of no improvement.  With `stag=100`, trigger
needs gen ≥ 100 + (last improvement gen) ≈ 120–130, leaving only
~70–80 gens of "active stagnation" before MaxGen=200.  It may or
may not activate depending on the run.

The mean time for `stag=100` (4.8s) is *higher* than for
`stag=300/500` (2.3s), which is counterintuitive (shorter run
should be faster).  Most likely a slow-first-run outlier — the
JSON does not save per-run timings, so we cannot confirm.  For
v0.7.3 we may add per-run timing to the JSON to diagnose.

To measure `sade_stag` meaningfully, use `--full` (MaxGen=500) or
a harder benchmark (hanoi).

## What to do next

1. **Do not change defaults** based on this data.  The default
   parameters are tuned for hanoi-scale problems (larger, harder).
   Twoloop is a small benchmark; its optimum is easy to reach for
   most algorithms.

2. **Run `--full`** (10 runs, 500 gen, hanoi + twoloop) when
   compute time permits (~2–3 hours).  This will:
   - Reduce SEM by ~1.8× (10 vs 3 runs).
   - Differentiate twoloop parameters more sharply.
   - Give first hanoi results (paper best $6.081M).
   - Show whether `sade_stag` matters at all.

3. **Use the recommendations for per-experiment overrides**, not
   as new tool defaults.  E.g. if a user is optimizing a small
   twoloop-scale network, they can set `de_cr=0.9` or
   `pso_w=0.5, pso_c1=2.0` via the GUI's algorithm parameters.  The
   default remains the hanoi-tuned value.

## How to reproduce or extend

```bash
# Regenerate markdown from existing JSONs
python scripts/sensitivity_analysis.py --regenerate-only

# Re-run quick plan (~10–15 min)
python scripts/sensitivity_analysis.py --quick

# Run full plan overnight (~2–3 h)
python scripts/sensitivity_analysis.py --full
```

The JSONs in `docs/sensitivity_data/` are the source of truth.
Markdown is generated. Both are version-controlled, so future
algorithm changes can be diffed against this baseline.