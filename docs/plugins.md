# Plugins

Add your own optimization algorithms to WDS Optimizer without
modifying the source code.

## Quick start

1. Copy `plugins/sample_plugin.py` (in the source repo) to your
   plugins directory:

   ```
   Windows:  %USERPROFILE%\.wds_optimizer\plugins\
   Linux:    ~/.wds_optimizer/plugins/
   macOS:    ~/.wds_optimizer/plugins/
   ```

   Override the location with `WDS_OPTIMIZER_PLUGINS_DIR=/some/path`.

2. Edit the file: change `ALGORITHM_META` and implement `run()`.

3. Verify it loaded:

   ```
   wds-optimize plugins list
   ```

4. Run it:

   ```
   wds-optimize run --inp net.inp --d D.txt --cost Cost.txt \
       --algo MyRandomSearch --pmin 30 --output out.json
   ```

In the GUI (v0.5.1), the algorithm appears in the "Algorithms to
run" dropdown with a `[P]` marker.

## Writing a plugin

A plugin is a single `.py` file with two exports:

```python
ALGORITHM_META = { ... }
def run(problem, config, *, progress_callback=None,
        cancel_callback=None, meta=None): ...
```

### ALGORITHM_META

Required keys:

| Key | Type | Description |
|-----|------|-------------|
| `short_id` | str | Unique id (used in `--algo`, GUI, API) |
| `display_name` | str | Human-readable name |
| `category` | str | Free-form category |
| `paper_faithful` | bool | True if this reproduces a paper |
| `description` | str | One-sentence summary |
| `references` | list[str] | Citations (`["-"]` if none) |
| `params` | list[dict] | Parameters shown in the GUI |
| `constraint_policy` | str | Must be a key of `CONSTRAINT_POLICIES` |
| `api_version` | int | Must be `1` |

Each param is a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `key` | str | Session-state key (e.g. `"my_iters"`) |
| `name` | str | Short name |
| `display_name` | str | Label shown in the GUI |
| `type` | str | `float` / `int` / `choice` / `bool` |
| `default` | same as type | Default value |
| `description` | str | Help text |
| `maps_to` | `None` | **Required** for plugins |
| `min`/`max`/`step` | number | For `float`/`int` |
| `choices` | list | For `choice` |

**Plugins must use `maps_to: None`.**  Values are stored in
`config.extra[short_id]` and read inside `run()`:

```python
my_iters = config.extra.get("MyAlgo", {}).get("my_iters", 100)
```

### run()

```python
def run(problem, config, *, progress_callback=None,
        cancel_callback=None, meta=None):
    ...
    return score, full_position_m, convergence, feasible
```

| Argument | Meaning |
|----------|---------|
| `problem` | Dict (see `optimization/problem.py`) |
| `config` | `AlgorithmConfig` |
| `progress_callback` | Called at intervals with a dict |
| `cancel_callback` | Return True to stop gracefully |
| `meta` | Fill with NFE counters if desired |

Returns:

- `score` (float) — best objective value
- `full_position_m` (ndarray) — full diameter vector in metres
- `convergence` (ndarray) — best cost per generation
- `feasible` (bool)

The return contract is identical to the built-in algorithms.  Copy
from `de.py` for a working template.

## Helpers available inside run()

```python
from wds_optimizer.optimization.evaluate import evaluate_population
from wds_optimizer.optimization.policies import get_policy
from wds_optimizer.optimization.constraints import check_constraints
```

These are the same primitives the built-in algorithms use.

## Error handling

If a plugin fails to load, the app **does not crash**.  Failures are
reported:

```
wds-optimize plugins list
```

prints loaded, failed, and skipped plugins with error messages.

Common failure reasons:

| Error | Fix |
|-------|-----|
| Missing `ALGORITHM_META` | Add the dict |
| Missing `run` | Add the function |
| `api_version` mismatch | Set to `1` |
| `short_id` collision | Rename it |
| `run()` signature wrong | Accept `(problem, config)` plus optional kwargs |
| `maps_to` not None | Change to `None` |

## Reloading

CLI: each invocation runs discovery automatically.

GUI (v0.5.1): "Reload plugins" button in the sidebar.

## Security

**Plugins are Python code with full system access.**  They can read
files, make network requests, and consume CPU/memory.  Only install
plugins from sources you trust.  There is no sandbox in v0.5.

To disable plugin loading in a sensitive environment:

```
set WDS_OPTIMIZER_DISABLE_PLUGINS=1    # Windows
export WDS_OPTIMIZER_DISABLE_PLUGINS=1 # Linux/macOS
```

## Sharing a plugin

Send the `.py` file.  The recipient copies it to their plugins
directory.  No packaging or versioning needed (v0.5).

## Roadmap

- v0.6: package plugins (folder with multiple files), sandbox
- v1.0: plugin versioning, marketplace