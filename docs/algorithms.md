# Algorithms

Four optimization algorithms are implemented.  This document
describes **what each one actually is** in this codebase, **what it is
not**, and its verified performance on the standard benchmarks.

There are two categories:

| Algorithm | Category | Paper-faithful? |
|-----------|----------|------------------|
| **SADE**             | Paper-faithful reproduction | ✅ Yes — Zheng et al. (2013) |
| **DE_suribabu_2010** | Paper-faithful reproduction | ✅ Yes — Suribabu (2010) |
| **DE**               | Standard textbook algorithm | ⚠️ No — general DE formulation |
| **PSO**              | Standard textbook algorithm | ⚠️ No — general PSO formulation |
| **GA**               | Standard textbook algorithm | ⚠️ No — general GA formulation |

Users who need **strict paper reproducibility** should use **SADE**.
Users who need one of the others for comparison purposes have the
standard algorithms available, with their configuration made explicit
below.

---

## SADE — Self-Adaptive Differential Evolution

**Reference.** Zheng, F., Zecchin, A. C., and Simpson, A. R. (2013).
"A self-adaptive differential evolution algorithm applied to water
distribution system optimization." *Journal of Computing in Civil
Engineering*, 27(2), 148-158.

**Implementation status: paper-faithful.** Every mechanism from the
paper is implemented as described:

- F and CR encoded in the individual and self-adapted (Eq. 1, 5)
- DE/rand/1 mutation (Eq. 2)
- Binomial crossover with `jrand` (Eq. 3)
- Deb's constraint tournament with the paper's tie-breaking (Eq. 4)
- Coefficient-of-variation convergence on **feasible** members (Eq. 6)
- Stagnation safeguard (documented in `StagnationMax`; the paper
  relies purely on Cv < Tol, this is an addition for practicality)

**Verified performance.** See `tests/test_sade.py` and
`tests/test_algorithms_benchmarks.py`.

| Benchmark | Best (3 seeds) | Paper best | Gap |
|-----------|---------------|------------|-----|
| Hanoi     | $6.192M       | $6.081M    | +1.8% |
| Two-loop  | $419,000      | $419,000   | 0.0% |

---

## DE — Differential Evolution

**Reference (general algorithm).** Storn, R., and Price, K. (1995).
"Differential Evolution - A Simple and Efficient Adaptive Scheme for
Global Optimization over Continuous Space." Technical Report TR-95-012,
International Computer Science Institute, Berkeley, CA.

**This is NOT a faithful implementation of Suribabu (2010).**
Suribabu (2010) — *"Differential evolution algorithm for optimal
design of water distribution networks"* — was consulted for parameter
guidance (F ≈ 0.5, CR ≈ 0.3), but the exact formulation in that paper
(e.g. its boundary handling) has not been reproduced or verified
against.  Treat this module as **standard DE**, not as a
paper-reproduction of Suribabu 2010.

**What this implements.**

- DE/rand/1/bin (default), plus best/1/bin and current-to-best/1/bin
- F = 0.5, CR = 0.3 (configurable in GUI/CLI/API)
- Binomial crossover with `jrand` guarantee
- Bound handling: **re-draw uniform** for out-of-range genes
  (not clip, not boundary reflection)
- Deb's constraint tournament with ties accepted
- Restart on stagnation (not in any reference)

**Verified performance.**

| Benchmark | Best (3 seeds) | Paper best | Gap |
|-----------|---------------|------------|-----|
| Hanoi     | $6.374M       | $6.081M    | +4.8% |
| Two-loop  | $419,000      | $419,000   | 0.0% |

DE reaches the known optimum exactly on two-loop.  The +4.8% gap on
Hanoi is consistent with standard DE at the given budget (NS=200,
MaxGen=300, three seeds).

---

## DE_suribabu_2010 — Differential Evolution (paper-faithful)

**Reference.** Suribabu, C. R. (2010). "Differential evolution
algorithm for optimal design of water distribution networks."
*Journal of Hydroinformatics*, 12(1), 66-82.

**Implementation status: paper-faithful.**  Every mechanism from the
paper is implemented as described:

- **Mutation**: DE/rand/1 only (Eq. 8).  No strategy choice.
- **Crossover**: binomial, **no jrand** (Eq. 8).  The paper does not
  force at least one gene from the mutant, so with probability
  `(1 - CR)^NVar` a trial can be an exact copy of its target.
- **Constraint handling**: fixed penalty (Eq. 4 of the Application
  section).  Infeasible designs get
  `effective_cost = cost of all variable pipes at max diameter`
  (a constant).  Any feasible design beats any infeasible one.
- **Selection**: simple greedy (Eq. 9).  The trial replaces its
  target iff `effective_cost(trial) <= effective_cost(target)`.
  Ties go to the trial.  NOT Deb's constraint tournament.
- **Bound handling**: clip to `[0, ND-1]`, then round to the
  nearest integer for evaluation.  Not re-draw.
- **No restart-on-stagnation.**  Not in the paper.

**Recommended parameters** (from the paper's Table 2 and Table 6):

| Parameter | Value | Paper reference |
|-----------|-------|-----------------|
| F  | 0.8 | Table 2, trial 2 (two-loop) / Table 6, trial 2 (Hanoi) |
| CR | 0.4 | same |
| NS | 20  | Main runs in the paper |
| MaxGen | 500 | Paper's fixed generation budget |

The paper reports that larger populations (NS=100) with more
generations (MaxGen=1000) substantially improve the success rate
to reach the published optimum, at the cost of more function
evaluations.

**Verified performance.**  See
`tests/test_de_suribabu_2010.py::test_de_suribabu_2010_hanoi_paper_settings`
and
`tests/test_de_suribabu_2010.py::test_de_suribabu_2010_twoloop_paper_settings`.

| Benchmark | Best | Paper best | Gap |
|-----------|------|------------|-----|
| Hanoi     | (verified by test) | $6.081M | ≤ 2% |
| Two-loop  | (verified by test) | $419,000 | ≤ 1% |

Run-to-run variance is higher than the standard `DE` because the
paper's mechanisms (no jrand, fixed penalty) reduce diversity.  The
paper itself reports 40-70% success rates per trial depending on the
parameter combination; that is why verification uses paper's high-
setting run (NS=100, MaxGen=1000) rather than the main (NS=20) runs.

**When to use this instead of `DE`.**  When you specifically want to
reproduce Suribabu (2010), or when you want a DE variant whose
mechanisms match a published paper exactly for a research comparison.
For general-purpose DE, the standard `DE` in this project typically
performs slightly better because Deb's tournament handles constraints
more gracefully than fixed penalty.

---

## PSO — Particle Swarm Optimization

**Reference (general algorithm).** Kennedy, J., and Eberhart, R.
(1995). "Particle swarm optimization." *Proceedings of ICNN'95 -
International Conference on Neural Networks*, Perth, Australia.

**This is NOT a faithful implementation of Suribabu & Neelakantan
(2006).** Suribabu & Neelakantan (2006) — *"Design of water
distribution networks using particle swarm optimization"* — uses a
**decreasing** inertia weight (typically 0.9 → 0.4) and velocity
clamping.  This module uses **constant** `w = 0.7` and **no velocity
clamping**.  Treat this module as **standard global-best PSO**, not
as a paper-reproduction of Suribabu 2006.

**What this implements.**

- Global-best topology (one GBest for all particles)
- Standard velocity update:
  `V ← w·V + c1·r1·(PBest − X) + c2·r2·(GBest − X)`
- `w = 0.7` (constant), `c1 = c2 = 1.5`
- Continuous positions, rounded to diameter index at evaluation
- No velocity clamping
- No inertia decay
- Deb's constraint tournament with ties accepted
- Restart on stagnation (not in any reference)

**Verified performance.**

| Benchmark | Best (3 seeds) | Paper best | Gap |
|-----------|---------------|------------|-----|
| Hanoi     | $6.202M       | $6.081M    | +2.0% |
| Two-loop  | $424,000      | $419,000   | +1.2% |

---

## GA — Genetic Algorithm

**Reference (general algorithm).** Holland, J. H. (1975). *Adaptation
in Natural and Artificial Systems*, University of Michigan Press; and
Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization
and Machine Learning*, Addison-Wesley.

**This is NOT a faithful implementation of either Simpson et al.
(1994) or Dandy et al. (1996).**

- **Simpson, Dandy & Murphy (1994)** uses **binary coding** (4-bit
  substrings per gene), **bitwise complement mutation**, and
  `Pm ≈ 0.01`.  This module uses **integer coding** and **uniform
  reset mutation** with default `Pm = 0.03`.
- **Dandy, Simpson & Murphy (1996)** adds three improvements over
  Simpson 1994: **Gray codes**, **adjacency (creep) mutation**, and
  **variable power scaling** of the fitness function.  None of these
  three is implemented here.

Treat this module as **a standard integer-coded GA with elitism**.
It is a legitimate GA — it uses selection, crossover, mutation, and
elitism — but it is not a reproduction of any specific paper, and in
particular it should not be compared to Dandy's Improved GA results
without that caveat.

**What this implements.**

- **Integer encoding** — each gene is a direct index into `D.txt`
- Selection: **tournament** (default) or **roulette**, both available
  via `config.GA.Selection`
- Single-point crossover with probability `Pc = 0.8`
- **Uniform reset mutation** — with probability `Pm = 0.03` per gene,
  the gene is replaced by a uniform random index (not a bit-flip,
  not an adjacent-diameter step)
- Elitism of one individual
- No fitness scaling, no Gray codes, no adjacency mutation
- Restart on stagnation (not in any reference)

**Verified performance.**

| Benchmark | Best (3 seeds) | Simpson 1994 Simple GA | Gap vs Simpson |
|-----------|---------------|------------------------|----------------|
| Hanoi     | $6.718M       | $6.112M                | +9.9% |
| Two-loop  | $423,000      | —                      | — |

The +9.9% gap on Hanoi should be compared to **Simpson et al. (1994)'s
Simple GA result of $6.112M**, not to the $6.081M that Suribabu's DE
achieved later.  For context, Dandy et al. (1996) Table 7 reports that
their Simple GA gets $41–51M on NYTP, while their Improved GA gets
$38.8M — a gap of 6–30%.  A +10% gap on Hanoi is **within the expected
performance range of a simple GA** and is not evidence of a bug.

If you need the accuracy of an Improved GA, wait for v0.2 (see Roadmap
below) or use SADE.

---

## Restart on stagnation (all four algorithms)

All four algorithms implement an optional restart-on-stagnation
mechanism.  This is **not described in any of the reference papers**;
it is an extension of this project.  When the best-so-far design has
not improved for `Patience` generations, the population is
re-initialized (either randomly, or by perturbing the current best).
The all-time best is always preserved.

This mechanism is **off by default**.  It is exposed in the GUI
sidebar and in `RestartConfig` for CLI/API users.

---

## Roadmap: paper-faithful presets (v0.3)

**Completed:**
- ✅ **DE_suribabu_2010** — Differential Evolution (Suribabu 2010)
  in v0.3.0.

**Planned:**

- ⏳ **PSO_suribabu_2006** — PSO with decreasing inertia weight and
  velocity clamping (Suribabu & Neelakantan 2006), v0.3.1.
- ⏳ **GA_simpson_1994** — Simple GA with binary coding and bitwise
  mutation (Simpson et al. 1994), v0.3.2.
- ⏳ **GA_dandy_1996** — Improved GA with Gray codes, adjacency
  mutation, and variable power scaling (Dandy et al. 1996), v0.3.2.

Each preset is registered as a separate algorithm (e.g. the user
selects `DE_suribabu_2010` from the CLI/API/GUI dropdown) and is
verified against the paper's reported benchmark results.