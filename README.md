# WDS Optimizer

Design optimization for water distribution systems. Implements the
classic genetic algorithm, particle swarm optimization, differential
evolution, and the self-adaptive DE of Zheng, Zecchin & Simpson (2013),
all wrapped around the WNTR/EPANET hydraulic engine.

Three interfaces, one engine:

- **GUI** — Streamlit app for interactive exploration
- **CLI** — `wds-optimize` for scripts, batch jobs and CI
- **Python API** — `from wds_optimizer import run_optimization`

Results are identical across the three because they all call the same
`run_optimization` function.

## Features

- Four algorithms: GA, PSO, DE, and SADE (paper-accurate).
- Four independently-togglable constraints: `Pmin`, `Pmax`, `Vmin`, `Vmax`.
- Deb's constraint tournament (no penalty multiplier to tune).
- Reproducibility: fixed seeds, full snapshots, deterministic reruns.
- Report generation: `REPORT.md` + static PNG figures + CSVs, packaged
  in a single ZIP.
- 123-test suite covering algorithm correctness, constraint handling,
  snapshot compatibility, and CLI behaviour.

## Installation

`wds-optimizer` supports three install profiles. Pick the one that
matches how you plan to use it.

### 1. CLI + Python API only (headless)

```bash
pip install -e .
```

Installs the core engine, `wnr`, and `pandas`. Enough for
`wds-optimize run ...` and `from wds_optimizer import run_optimization`.
No Streamlit, no matplotlib. Ideal for servers, CI, embedded use.

### 2. Add progress bars and reports

```bash
pip install -e ".[cli,report]"
```

Adds `tqdm` (CLI progress bar) and `matplotlib` (static figures in
`--report`). Still no Streamlit.

### 3. Full desktop experience (GUI + everything)

```bash
pip install -e ".[all]"
```

Installs Streamlit, Plotly and everything above. Use this if you want
the GUI.

### For development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Quickstart — CLI

```bash
# Check that the input files parse correctly
wds-optimize validate \
    --inp hanoi.inp --d D.txt --cost Cost.txt --diam-unit in

# Run SADE with Pmin=30 m, Vmax=2.5 m/s
wds-optimize run \
    --inp hanoi.inp --d D.txt --cost Cost.txt --diam-unit in \
    --algo SADE --ns 200 --max-gen 500 \
    --pmin 30 --vmax 2.5 \
    --output hanoi.json --report hanoi.zip
```

Exit codes: `0` feasible result, `1` no feasible design found,
`2` invalid input, `3` hydraulics failed, `4` cancelled, `10` unexpected.
All exit codes are documented in `wds-optimize run --help`.

`wds-optimize info` prints the installed versions of every relevant
package — include its output in any bug report.

## Quickstart — Python API

```python
from wds_optimizer import OptimizationRequest, run_optimization

req = OptimizationRequest(
    inp_path="hanoi.inp",
    d_path="D.txt",
    cost_path="Cost.txt",
    diam_unit="in",
    algorithms=["SADE"],
    n_runs=10,
    seed=42,
    ns=200,
    max_gen=500,
    constraints={
        "Pmin": {"enabled": True,  "value": 30.0},   # metres
        "Pmax": {"enabled": False, "value": 100.0},  # metres
        "Vmin": {"enabled": False, "value": 0.0},    # m/s
        "Vmax": {"enabled": True,  "value": 2.5},    # m/s
    },
)

result = run_optimization(req)

print(f"Best cost:  {result.best_score:,.0f}")
print(f"Feasible:   {result.best_feasible}")
print(f"Algorithm:  {result.best_algo} run {result.best_run_index}")

# Exports (all return bytes; no filesystem side effects)
open("hanoi.json", "wb").write(result.to_json())
open("hanoi.csv",  "wb").write(result.to_csv())
open("hanoi.xlsx", "wb").write(result.to_excel())
open("hanoi.zip",  "wb").write(result.to_report_zip())
```

The API always uses **SI units** (pressure in metres, velocity in m/s).
The CLI converts from the network's native units automatically; the API
does not, because its callers are expected to be code, not humans.

## Quickstart — GUI

```bash
pip install -e ".[gui]"
streamlit run src/wds_optimizer/app/streamlit_app.py
```

## What's implemented

| Feature | Status |
|---------|--------|
| Genetic Algorithm (GA) | ✅ |
| Particle Swarm Optimization (PSO) | ✅ |
| Differential Evolution (DE) | ✅ |
| Self-adaptive DE (SADE, Zheng et al. 2013) | ✅ |
| Four optional constraints (Pmin/Pmax/Vmin/Vmax) | ✅ |
| Deb's constraint tournament | ✅ |
| Restart on stagnation | ✅ |
| Snapshot save/load | ✅ |
| Reproducibility report (ZIP) | ✅ |
| CLI (`wds-optimize`) | ✅ |
| Python API | ✅ |
| Multi-objective optimization | 🔜 v0.2 |
| Plugin architecture | 🔜 v0.3 |

## Tests

```bash
pytest tests/ -v
```

123 tests. Runtime ~3 minutes on a laptop; the algorithm tests use
synthetic networks so they stay under 1 s each.

## Citing

If you use this tool in academic work, please cite:

- Zheng, F., Zecchin, A. C., & Simpson, A. R. (2013). *A self-adaptive
  differential evolution algorithm applied to water distribution system
  optimization*. Journal of Computing in Civil Engineering, 27(2),
  148-158.

And the underlying simulator:

- Klise, K. A., et al. *Water Network Tool for Resilience (WNTR)*, US
  Environmental Protection Agency.

## License

MIT for research and non-commercial use.  Commercial deployments may
require a separate license — please contact the author.

See [LICENSE](LICENSE) for the full text.

## Contributing

Bug reports and pull requests are welcome. For larger changes, please
open an issue first to discuss the design.