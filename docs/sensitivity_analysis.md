# Sensitivity Analysis

_n_runs = 3, seed = 42, 8 parameter sweep(s)._


## `DE.de_cr`  (DE)  —  twoloop

| Value | Best | Mean | Std | Feasible % | Time (s) |
|-------|------|------|-----|------------|----------|
| 0.3 | $419,000 | $420,333 | $1,528 | 100% | 3.1 |
| 0.5 | $419,000 | $420,667 | $2,887 | 100% | 6.2 |
| 0.9 | $419,000 | $419,333 | $577 | 100% | 5.0 |
- Uninformative sweep: means $419,333–$420,667 (<1% spread; parameter has no measurable effect)

**Recommended:** _none — the sweep is uninformative_

## `DE.de_f`  (DE)  —  twoloop

| Value | Best | Mean | Std | Feasible % | Time (s) |
|-------|------|------|-----|------------|----------|
| 0.3 | $419,000 | $420,000 | $1,732 | 100% | 6.9 |
| 0.5 | $419,000 | $420,333 | $1,528 | 100% | 7.3 |
| 0.7 | $422,000 | $424,667 | $2,309 | 100% | 7.8 |
- Not significant: gap $333 between best and runner-up vs 2·SEM $2,667

**Recommended:** `DE.de_f = 0.3` (mean $420,000; lowest mean cost)

## `GA.ga_pc`  (GA)  —  twoloop

| Value | Best | Mean | Std | Feasible % | Time (s) |
|-------|------|------|-----|------------|----------|
| 0.6 | $419,000 | $447,333 | $24,542 | 100% | 4.0 |
| 0.7 | $423,000 | $428,333 | $8,386 | 100% | 4.1 |
| 0.8 | $424,000 | $436,000 | $11,136 | 100% | 4.9 |
| 0.9 | $438,000 | $444,333 | $6,506 | 100% | 4.5 |
- Not significant: gap $7,667 between best and runner-up vs 2·SEM $16,097
- High variance at value=0.6: CV = 5.5%

**Recommended:** `GA.ga_pc = 0.7` (mean $428,333; lowest mean cost)

## `GA.ga_pm`  (GA)  —  twoloop

| Value | Best | Mean | Std | Feasible % | Time (s) |
|-------|------|------|-----|------------|----------|
| 0.01 | $419,000 | $447,667 | $34,530 | 100% | 3.4 |
| 0.03 | $424,000 | $436,000 | $11,136 | 100% | 2.0 |
| 0.05 | $420,000 | $440,000 | $17,776 | 100% | 7.9 |
| 0.1 | $420,000 | $430,000 | $14,000 | 100% | 9.1 |
- Not significant: gap $6,000 between best and runner-up vs 2·SEM $20,656
- High variance at value=0.01: CV = 7.7%
- High variance at value=0.05: CV = 4.0%
- High variance at value=0.1: CV = 3.3%
- Timing anomaly at value=0.1: 9.1s vs median 5.6s

**Recommended:** `GA.ga_pm = 0.1` (mean $430,000; lowest mean cost)

## `GA.ga_sel`  (GA)  —  twoloop

| Value | Best | Mean | Std | Feasible % | Time (s) |
|-------|------|------|-----|------------|----------|
| tournament | $436,000 | $447,333 | $9,815 | 100% | 3.4 |
| roulette | $424,000 | $436,000 | $11,136 | 100% | 1.2 |
- Not significant: gap $11,333 between best and runner-up vs 2·SEM $17,140

**Recommended:** `GA.ga_sel = roulette` (mean $436,000; lowest mean cost)

## `PSO.pso_c1`  (PSO)  —  twoloop

| Value | Best | Mean | Std | Feasible % | Time (s) |
|-------|------|------|-----|------------|----------|
| 1 | $424,000 | $426,333 | $2,082 | 100% | 2.7 |
| 1.5 | $424,000 | $438,000 | $14,000 | 100% | 1.4 |
| 2 | $419,000 | $422,333 | $2,887 | 100% | 3.5 |
- Not significant: gap $4,000 between best and runner-up vs 2·SEM $4,110
- High variance at value=1.5: CV = 3.2%

**Recommended:** `PSO.pso_c1 = 2` (mean $422,333; lowest mean cost)

## `PSO.pso_w`  (PSO)  —  twoloop

| Value | Best | Mean | Std | Feasible % | Time (s) |
|-------|------|------|-----|------------|----------|
| 0.5 | $424,000 | $424,000 | $0 | 100% | 2.9 |
| 0.7 | $424,000 | $438,000 | $14,000 | 100% | 3.4 |
| 0.9 | $420,000 | $453,667 | $29,872 | 100% | 9.3 |
- Not significant: gap $14,000 between best and runner-up vs 2·SEM $16,166
- High variance at value=0.7: CV = 3.2%
- High variance at value=0.9: CV = 6.6%
- Timing anomaly at value=0.9: 9.3s vs median 3.4s

**Recommended:** `PSO.pso_w = 0.5` (mean $424,000; lowest mean cost)

## `SADE.sade_stag`  (SADE)  —  twoloop

| Value | Best | Mean | Std | Feasible % | Time (s) |
|-------|------|------|-----|------------|----------|
| 100 | $419,000 | $422,333 | $2,887 | 100% | 4.8 |
| 300 | $419,000 | $422,333 | $2,887 | 100% | 2.3 |
| 500 | $419,000 | $422,333 | $2,887 | 100% | 2.4 |
- Uninformative sweep: means $422,333–$422,333 (<1% spread; parameter has no measurable effect)
- Timing anomaly at value=100: 4.8s vs median 2.4s

**Recommended:** _none — the sweep is uninformative_