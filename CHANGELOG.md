# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Sensitivity batch backend (v0.8.1)
- `wds_optimizer/sensitivity/batch.py`:
  - `PlanRow`, `BatchPoint`, `BatchParamResult`, `BatchSweepResult`
  - `run_batch_sweep(problem, cfg, rows=..., n_runs=..., seed=...)`
    executes one experiment per (row, value).  Recommendation logic
    detects `lowest_mean`, `flat_line` (<1% spread), `high_variance`,
    and `no_successful_points`.
  - `validate_plan(rows)` fails fast on typos / mismatches (same
    lesson as v0.7.0's sensitivity_analysis.py validation).
- `wds_optimizer/reporting/sensitivity.py`:
  - `batch_report_markdown(batch, ...)` — full markdown report
  - `batch_report_csv(batch)` — flat CSV (utf-8-sig, Excel-friendly)
  - `load_batch_from_jsons(output_dir)` — reconstruct a batch from
    saved JSONs (for `--regenerate-only`); handles old v0.7.1 JSONs
    by recomputing recommendations.
- CLI: new `wds-optimize sensitivity` subcommand:
    --inp / --d / --cost / --diam-unit
    --algo (repeatable)
    --sweep "PARAM=V1,V2,..."  (repeatable)
    --n-runs, --seed, --output-dir, --markdown, --csv
  Reuses the existing constraint flags (`--pmin`, `--vmax`, ...).
- `scripts/sensitivity_analysis.py` refactored to use
  `run_batch_sweep` + `batch_report_markdown` + `load_batch_from_jsons`.
  CLI and JSON output are byte-compatible with v0.7.1; markdown
  format changes slightly (sections now say `## \`param\` (algorithm)`).

### Notes
- GUI unchanged.  v0.8.2 replaces the Sweep tab with a new
  Sensitivity dialog + tab that use the same backend.
- No regression in single-parameter sweeps: they are a one-row
  batch internally.

### Changed

#### Rename `sweep` → `sensitivity` (v0.8.0)
- Python package: `wds_optimizer.sweep` → `wds_optimizer.sensitivity`.
  The old package name still works but emits a `DeprecationWarning`
  and will be removed in v0.9.0.
- Public API (renamed):
  - `SWEEP_PARAMETERS` → `SENSITIVITY_PARAMETERS`
  - `build_sweep_parameters()` → `build_sensitivity_parameters()`
  - `parse_sweep_values()` → `parse_sensitivity_values()`
  - `run_parameter_sweep()` → `run_single_sweep()`
  - `SweepResult` → `SingleSweepResult`
  - `SweepPointResult` → `SingleSweepPoint`
  - `_apply_sweep_param()` → `_apply_sensitivity_param()`
  - Old names remain as aliases (`SWEEP_PARAMETERS = SENSITIVITY_PARAMETERS`,
    etc.) in the shim.
- Test file: `tests/test_sweep_dynamic.py` → `tests/test_sensitivity_dynamic.py`.
- Rationale: the user-facing feature is "sensitivity analysis".  Using
  "sweep" in code but "sensitivity" in UI creates two vocabularies for
  the same concept.  The rename aligns them.

### Deprecated
- `wds_optimizer.sweep` (whole package): will be removed in v0.9.0.
- All `SWEEP_*` and `*_sweep*` names inside the package.

### Notes
- **No user-visible behaviour change in v0.8.0.**  The GUI still shows
  "Parameter Sweep" in the sidebar and the Sweep tab.  v0.8.2 will
  replace both with the new Sensitivity dialog and tab.  This commit
  is internal-only.
- CLI: `wds-optimize sweep` (if it existed) still works via the shim.
  A new `wds-optimize sensitivity` subcommand will be added in v0.8.2.

### Added

#### Sensitivity analysis interpretation (v0.7.2)
- `docs/sensitivity_notes.md` — human-authored interpretation of the
  v0.7.1 quick-mode results: what is signal, what is noise, what is
  uninformative, and what to do next.
- `docs/parameters.md` — adds a "Twoloop (n=3)" column to each
  parameter table with observed values, and a "Methodology and
  caveats" section explaining the limits of 3-run quick-mode data.

### Notes
- No code changed in v0.7.2.  This is documentation only.
- Default parameters were **not** changed.  The twoloop data does
  not justify changing defaults (twoloop is small; hanoi-tuned
  defaults remain the right choice).
- Users can apply the twoloop-specific recommendations as
  per-experiment overrides via the GUI/API.
- `--full` (10 runs, hanoi + twoloop) is planned for a later
  release.

### Added

#### Sensitivity analysis pipeline (v0.7.0)
- `scripts/sensitivity_analysis.py` — runs predefined sweeps on the
  benchmarks and writes JSON + a generated Markdown report.
  Modes: `--quick` (twoloop only, ~5 min), `--full` (hanoi +
  twoloop, 10 runs, 500 gen, ~2-3 h), single-sweep
  (`--benchmark/--algo/--param/--values`), and `--regenerate-only`.
- `docs/parameters.md` — reference doc for every parameter: meaning,
  default, range, effect.  Complements `docs/algorithms.md` (which
  describes each algorithm).
- `docs/sensitivity_analysis.md` — auto-generated report;
  populated in v0.7.1 after running `--quick`.
- `docs/sensitivity_data/**/*.json` — raw sweep records, one file
  per (benchmark, algorithm, parameter).  Committed so future changes
  can be diffed against this baseline.
- `tests/test_sensitivity_pipeline.py` — 7 smoke tests
  (dry-run, mini-sweep, markdown regeneration, error handling).

### Notes
- v0.7.0 ships the *pipeline*.  The results are populated in v0.7.1
  after running `--quick`.  `--full` results will land in a later
  release.
- The script uses the existing Sweep backend (v0.6.0) unchanged.
  No new dependency, no backend change.
- Sensitivity analysis is an *authoring tool* for us.  Users run
  interactive sweeps through the GUI's Sweep tab; the script is what
  we use to write the parameter recommendations that ship with the
  tool.

### Added

#### Plugin system — GUI integration (v0.5.1)
- Plugins appear in the "Algorithms to run" dropdown with a `[P]`
  marker, alongside built-in algorithms.  No new concept, no
  separate panel — plugins are just algorithms.
- New "🔌 Plugins" expander in the sidebar (section 3):
  - 🔄 Reload — re-scan the plugins directory without restarting.
  - 📂 Open folder — opens the plugins directory in the file
    manager.
  - Load plugin from file — pick a `.py` file; it is copied into the
    plugins directory and loaded immediately.
  - List of loaded, failed, and skipped plugins with error details.
- First-use security warning: users must acknowledge that plugins
  are Python code with full system access before managing them.
- "View source code" expander inside each plugin's parameter panel,
  showing the plugin's `run()` source read-only.
- `reset_plugins()` is now public (was `_reset_for_tests`).

### Added

#### Plugin system — backend (v0.5.0)
- New `wds_optimizer/plugins.py` module: discovery, loading,
  validation, and registration of user-provided algorithms.
- Plugin directory: `~/.wds_optimizer/plugins/` (override with
  `WDS_OPTIMIZER_PLUGINS_DIR`; disable with
  `WDS_OPTIMIZER_DISABLE_PLUGINS=1`).
- Each plugin is a single `.py` file exporting `ALGORITHM_META`
  (same schema as built-ins + required `api_version: 1`) and
  `run(problem, config, *, progress_callback, cancel_callback, meta)`
  (same signature as built-ins).
- Plugin params must use `maps_to: None`; values land in
  `config.extra[short_id]`.
- Malformed or failing plugins never crash the app: import errors,
  missing exports, bad signatures, and META validation failures are
  all caught and reported in a `PluginLoadReport`.
- CLI: `wds-optimize plugins list` and `wds-optimize plugins load
  <path>`.  Every CLI invocation runs discovery automatically, so
  `--algo MyPlugin` works without extra steps.
- `plugins/sample_plugin.py` — annotated template.
- `docs/plugins.md` — full authoring guide with security warning.

### Notes

- **Security:** plugins are Python code with full system access.
  There is no sandbox in v0.5.  Only install plugins from sources
  you trust.  Set `WDS_OPTIMIZER_DISABLE_PLUGINS=1` to disable
  discovery entirely.
- GUI integration (sidebar expander, reload button, "View source",
  "Load from file") is planned for v0.5.1.
- A plugin that fails to load is skipped; other plugins and the
  four built-ins continue to work normally.

### Added

#### History replaces Scenarios (v0.4.1a)
- The "6. Scenarios" panel (v0.4.1) has been removed.  It introduced
  a second concept parallel to "experiment", which was confusing:
  it was not obvious whether "Run" created an experiment or updated
  a scenario, and there was no way to edit a scenario's config
  without loading it into the sidebar.
- Replaced by "6. History": an automatic list of past runs.  Each
  entry has a user-editable name, viewable config, and the ability
  to Load (into sidebar for edit/re-run), Rename, Save as JSON, and
  Delete.
- Optional "Run name" field above the Run button names the next run.
  Left empty, an auto-generated name is used (e.g.
  "GA+SADE / hanoi / ns=100 / seed=42").  Rename anytime via ✏️.
- History is session-local.  Use "💾 Save all" / "Load" to transfer
  across sessions or machines.
- No more "Duplicate" button — Load + edit + Run naturally creates
  a new entry.
- No more "Active scenario" concept — every run is recorded; the
  experiment you just ran is always the most recent History entry.

### Removed

- `src/wds_optimizer/scenario.py` and `tests/test_scenario.py`.
  Superseded by `history.py` (v0.4.1a).  The Scenario concept (pre-
  planning of multiple experiments + batch execution) is deferred;
  if it returns in v0.5, it will build on top of History rather than
  parallel to it.

### Added

#### Scenarios (v0.4.1)
- "6. Scenarios" panel in the sidebar.  Users can:
  - Save the current sidebar configuration as a named scenario
  - Update the active scenario from the current sidebar
  - Load a scenario into the sidebar (clears any current experiment)
  - Duplicate / delete scenarios
  - Save the whole workspace (all scenarios) as a single JSON
  - Load a workspace from a previously saved JSON
- When a run finishes with an active scenario, the result is recorded
  into that scenario (status becomes "Complete" or "Failed" on cancel).
- Scenario data model lives in wds_optimizer/scenario.py; it is pure
  data + I/O, no GUI imports, usable from CLI/API/tests/SaaS.
- Scenarios embed input files by default so they are portable.

### Fixed

- OptimizationRequest now deep-copies the `constraints` dict.
  Mutating the caller's dict after creating a request no longer
  affects the request.  Same class of bug as the shallow copy fixed
  in scenario.py (v0.4.0).

### Notes

- Loading a scenario into the sidebar populates the *configuration*
  only, not the scenario's experiment result.  Viewing results of
  completed scenarios across the whole workspace (batch mode) is
  planned for v0.4.2.
- Workspace files (wds_workspace_*.json) are separate from single-
  experiment snapshots (wds_experiment_*.json).  Use the existing
  "5. Save / Load" section for a single experiment; use "6. Scenarios"
  for a workspace.

### Added

#### Constraint-handling policy (v0.3.2)
- `constraint_policy` is now a first-class, end-to-end setting.
- Three built-in policies: `deb_prefers` (default), `fixed_penalty_prefers`,
  `simple_cost_only_prefers`.
- GUI: "Constraint handling" section in the sidebar with dropdown,
  description, and "View code and reference" expander.
- CLI: `--constraint-policy {deb_prefers,fixed_penalty_prefers,simple_cost_only_prefers}`.
- API: `OptimizationRequest.constraint_policy`.
- Snapshots record `constraint_policy` under `config` and `experiment`;
  old snapshots default to `deb_prefers`.
- Each algorithm reads `config.constraint_policy` at run start.

## [0.2.0] - 2026-11-13

Second public release.  Adds benchmarks, verification tests, and
honest per-algorithm documentation.  First release with published-
benchmark reproducibility verified by the test suite.

### Added

#### Benchmarks (`benchmarks/`)
- `hanoi/` (Fujiwara & Khang 1990; Simpson et al. 1994) —
  best-known cost $6.081M, Pmin = 30 m only
- `twoloop/` (Alperovits & Shamir 1977) — best-known cost $419,000,
  Pmin = 30 m only
- `nytp/README.md` — reserved for v0.3 (requires duplication
  formulation, documented)
- Per-benchmark READMEs with citation, unit notes, constraint
  settings, and reproduction commands
- `benchmarks/README.md` index with constraint guidance

#### Verification (`tests/test_algorithms_benchmarks.py`)
- Slow-marked tests (`@pytest.mark.slow`, excluded by default)
  that run each algorithm on hanoi and twoloop and assert the
  best-seed cost is within a documented tolerance of the correct
  reference (GA: Simpson $6.112M; PSO/DE/SADE: Suribabu $6.081M;
  two-loop: Alperovits & Shamir $419K).

#### Documentation
- `docs/algorithms.md` — for each algorithm: what it IS, what it
  is NOT, verified gaps, reference paper, and roadmap for
  paper-faithful presets in v0.3
- `scripts/baseline_benchmarks.py` — regenerates baseline table

#### Packaging
- `pytest` markers (`slow`) and `addopts = '-m "not slow"'` so
  CI stays fast; full verification runs on demand.

### Fixed

- `pyproject.toml` markers array was TOML-invalid (unclosed array,
  two adjacent strings without a comma); `pip install -e .` failed
  on CI.
- `benchmarks/hanoi/README.md` incorrectly listed Vmax = 2.5 m/s
  as active.  No reference paper on Hanoi enforces a velocity
  constraint; Vmax = 2.5 m/s makes the problem infeasible (the
  optimal design has pipes at ~9 m/s).
- `tests/test_algorithms_benchmarks.py` (initial draft) compared
  GA against Suribabu's DE result ($6.081M), which is unfair for a
  Simple GA.  Fixed to compare GA against Simpson 1994's Simple GA
  ($6.112M), which is the correct reference for this algorithm
  class.

### Notes

- v0.1.0 (initial release) remains valid but did not include
  benchmarks.  Use v0.2.0 for benchmark reproducibility work.
- Paper-faithful presets (Simpson 1994 GA, Dandy 1996 Improved GA,
  Suribabu 2006 PSO, Suribabu 2010 DE) are planned for v0.3.

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

[Unreleased]: https://github.com/alirezakhatami98-ui/WDS_Optimizer_Py/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/alirezakhatami98-ui/WDS_Optimizer_Py/releases/tag/v0.2.0
[0.1.0]: https://github.com/alirezakhatami98-ui/WDS_Optimizer_Py/releases/tag/v0.1.0