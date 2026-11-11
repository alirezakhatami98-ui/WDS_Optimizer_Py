# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

(nothing yet)

## [0.1.0] - 2026-11-11

First public release.  Single-objective optimization of water
distribution system designs, exposed through a GUI, a CLI and a Python
API, backed by 123 tests.

### Added

#### Algorithms
- Genetic Algorithm (GA) with tournament and roulette selection.
- Particle Swarm Optimization (PSO).
- Differential Evolution (DE) with three mutation strategies
  (`rand/1/bin`, `best/1/bin`, `current-to-best/1/bin`).
- Self-adaptive Differential Evolution (SADE), implementing
  Zheng, Zecchin & Simpson (2013), including jrand crossover,
  feasible-only Cv termination, per-individual F/CR adaptation,
  and a stagnation safeguard.
- Restart-on-stagnation (shared by all four algorithms).

#### Constraints
- Four independently-togglable constraints: Pmin, Pmax, Vmin, Vmax.
- SI values in the API and snapshot; display-unit conversion in GUI/CLI.
- Deb's constraint tournament (no penalty multiplier).
- Normalised per-family violation sums, bit-for-bit identical to the
  previous two-constraint implementation when only Pmin and Vmax are
  enabled.

#### Interfaces
- Python API (`wds_optimizer.api`):
  - `OptimizationRequest`, `OptimizationResult`, `run_optimization`.
  - `to_json()`, `to_csv()`, `to_excel()`, `to_report_zip()`
    export methods (all return bytes; no filesystem writes).
- CLI (`wds-optimize`):
  - `run` with full argument coverage (algorithms, population,
    generations, constraints, output formats).
  - `validate` to check inputs without running.
  - `info` for environment metadata (bug reports).
  - `benchmark` for built-in networks.
  - Graceful SIGINT handling with partial-result preservation.
  - Documented exit codes for scripting.
  - Diameter-unit sanity check (warns when D.txt values look
    inconsistent with `--diam-unit`).
- GUI (Streamlit):
  - Live progress panel with per-generation updates.
  - Run Inspector with per-run dropdown and ⭐ best-cost markers.
  - Network Map with Pmin/Pmax/Vmin/Vmax colouring.
  - Sweep tab for parameter studies.
  - Compare tab for A/B experiment comparison.
  - Statistical analysis (Mann-Whitney, Kruskal-Wallis, A12).

#### Packaging
- `wds-optimize` console script via `[project.scripts]`.
- Optional-dependency extras: `cli`, `report`, `gui`, `dev`, `all`.
  `pip install -e .` (base) is headless-safe.
- `__version__ = "0.1.0"` exported from `wds_optimizer`.

#### Reproducibility
- Snapshot schema v0.4.0 with embedded input files.
- Fully deterministic under fixed seed.
- Report ZIP containing REPORT.md, README.txt, all figures, all data
  CSVs, and the snapshot JSON.

### Fixed

- `KeyError: 'Pmin'` crash in `run_experiment` after the Pmax/Vmin
  refactor.  Fixed by populating `ExperimentResult.Pmin`/`Vmax` from
  `problem["Constraints"]`.
- Cross-network state leakage: `Hanoi -> Duhala -> Hanoi` returned
  absurd results on the third run.  Root causes (two independent
  bugs):
  - Cached `WaterNetworkModel` mutated in place by
    `run_hydraulic_simulation`.  Fixed by deep-copying on every load.
  - Upload paths collided when the same filename was uploaded for
    different networks (`A -> B -> A`).  Fixed by content-addressing
    uploads (`<stem>_<sha16><suffix>`).
- UnicodeEncodeError on Windows when CLI output was piped or captured.
  Fixed by keeping all `print()` strings ASCII-only.
- KeyboardInterrupt during the pre-run warning sleep produced a raw
  traceback.  Fixed with an explicit `except KeyboardInterrupt` in
  `cmd_run` and a safety net in `main`.
- `EXIT_OK` on a cancelled run.  Fixed: `cmd_run` now treats
  `experiment.cancelled` (set by any cancellation source) as
  `EXIT_CANCELLED`.
- `Pmin`/`Vmax` printed with em dashes in the per-algorithm summary
  (Windows crash).  Replaced with ASCII.

### Known limitations

- Single-objective only.  Multi-objective optimization is planned for
  v0.2.
- No plugin architecture.  Adding a new algorithm requires editing
  `algorithms/` and `runner.py`.
- The API does not accept `bytes` for input paths yet (file paths only).
  Needed for SaaS; planned for v0.2.
- Report PNGs use matplotlib and require the `[report]` extra.

### Tests

123 tests covering:
- `check_constraints` backward compatibility and edge cases (17).
- Upload handling and cross-network round-trips (12).
- `ExperimentResult` construction, cancel, progress (15).
- SADE convergence, termination, tournament (7).
- Snapshot backward/forward compatibility (6).
- CLI behaviour via subprocess (19).
- API request validation and end-to-end runs (22).
- Severity helper (12).
- Miscellaneous (13).

[Unreleased]: https://github.com/alirezakhatami98-ui/WDS_Optimizer_Py/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alirezakhatami98-ui/WDS_Optimizer_Py/releases/tag/v0.1.0