"""Plugin discovery, loading, and validation.

Discovers Python files in a user-configurable directory, imports
them, validates their ALGORITHM_META and run() signature, and
registers them with the algorithm registry.

Directory
---------
Default:  ~/.wds_optimizer/plugins/
Override: WDS_OPTIMIZER_PLUGINS_DIR environment variable
Disable:  WDS_OPTIMIZER_DISABLE_PLUGINS=1 (discovery is a no-op)

Plugin file
-----------
A single .py file at the top level of the plugins directory.
It must export:

    ALGORITHM_META : dict    — same schema as built-in algorithms,
                                plus required "api_version": 1
    run(...)       : callable — same signature as built-in algorithms

For plugin params, META entries MUST use "maps_to": None.  Values
land in ``config.extra[short_id]`` and are read inside run() as:

    my_lr = config.extra.get("MyAlgo", {}).get("my_lr", 0.1)

Security
--------
Plugins are arbitrary Python code with full system access.  Only
install plugins from sources you trust.  This module does NOT
sandbox plugins; sandboxing is future work (v0.6+).

A malformed or failing plugin never crashes the app: every step
(import, META validation, signature check, register) is wrapped in
try/except, and failures are reported in the PluginLoadReport.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .algorithms.registry import (
    ALGORITHM_REGISTRY,
    _validate_meta,
    register,
)


CURRENT_API_VERSION = 1
_REQUIRED_RUN_KWARGS = ("progress_callback", "cancel_callback", "meta")


@dataclass
class LoadResult:
    """Result of loading one plugin file."""
    status: str              # "loaded" | "skipped" | "failed"
    short_id: str = ""
    error: str = ""


@dataclass
class PluginLoadReport:
    """Summary of one discovery pass."""
    enabled: bool = True
    loaded: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Module-level state: short_ids registered from plugin files, and the
# last discovery report.  Kept private; helpers expose read-only views.
_PLUGIN_SHORT_IDS: set[str] = set()
_LAST_REPORT: Optional[PluginLoadReport] = None


# ---------------------------------------------------------------------
#  Directory and enable/disable
# ---------------------------------------------------------------------

def plugins_dir() -> Path:
    """Return the plugin directory (env override or default)."""
    override = os.environ.get("WDS_OPTIMIZER_PLUGINS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".wds_optimizer" / "plugins"


def plugins_enabled() -> bool:
    """False if discovery is disabled via env var."""
    return os.environ.get("WDS_OPTIMIZER_DISABLE_PLUGINS", "") != "1"


def get_plugin_short_ids() -> set[str]:
    """Return the set of short_ids loaded from plugin files."""
    return _PLUGIN_SHORT_IDS.copy()


def last_report() -> PluginLoadReport:
    """Return the most recent discovery report (empty if none yet)."""
    return _LAST_REPORT or PluginLoadReport()


# ---------------------------------------------------------------------
#  Import helper
# ---------------------------------------------------------------------

def _import_module_from_path(path: Path):
    """Import a .py file as a fresh module.  Safe against name collisions."""
    module_name = f"_wds_plugin_{path.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}: not a valid Python file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Do not leave a half-initialised module in sys.modules.
        sys.modules.pop(module_name, None)
        raise
    return module


# ---------------------------------------------------------------------
#  Validation helpers
# ---------------------------------------------------------------------

def _check_run_signature(run_callable) -> None:
    """Raise ValueError if run() cannot accept the runner's kwargs."""
    try:
        sig = inspect.signature(run_callable)
    except (ValueError, TypeError):
        # Cannot introspect (e.g. C extension).  Trust the user.
        return
    params = sig.parameters
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    positional = [
        p for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) < 2:
        raise ValueError(
            "run() must accept at least (problem, config) as the first "
            "two positional arguments"
        )
    if not has_var_kw:
        missing = [k for k in _REQUIRED_RUN_KWARGS if k not in params]
        if missing:
            raise ValueError(
                f"run() is missing keyword arguments {missing}; add them "
                f"or accept **kwargs"
            )


def _check_param_maps_to(meta: dict) -> None:
    """Plugins must not map params onto built-in family fields.

    A plugin that declared ``"maps_to": ("GA", "Pc")`` would overwrite
    a built-in algorithm's parameter.  That is not the plugin's job.
    Plugins use ``"maps_to": None`` so their values land in
    ``config.extra[short_id]``.
    """
    for p in meta["params"]:
        if p.get("maps_to") is not None:
            raise ValueError(
                f"param {p['key']!r} declares maps_to={p['maps_to']!r}; "
                f"plugins must use maps_to=None (values land in "
                f"config.extra[short_id])"
            )


# ---------------------------------------------------------------------
#  Load one file
# ---------------------------------------------------------------------

def load_plugin_from_file(path: str | Path, *, verbose: bool = False) -> LoadResult:
    """Load, validate, and register one plugin file.

    Never raises on file-level problems; returns a LoadResult with
    status "loaded" / "skipped" / "failed" and a human-readable error.
    """
    path = Path(path)
    name = path.name

    # ---- Import ----
    try:
        module = _import_module_from_path(path)
    except SyntaxError as e:
        msg = f"Syntax error at line {e.lineno}: {e.msg}"
        if verbose:
            print(f"[plugins] {name}: {msg}", file=sys.stderr)
        return LoadResult("failed", error=msg)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"[plugins] {name}: {msg}", file=sys.stderr)
        return LoadResult("failed", error=msg)

    # ---- Required exports ----
    if not hasattr(module, "ALGORITHM_META"):
        msg = "missing ALGORITHM_META dictionary"
        if verbose:
            print(f"[plugins] {name}: {msg}", file=sys.stderr)
        return LoadResult("failed", error=msg)
    meta = module.ALGORITHM_META

    # Extract short_id early so subsequent checks (api_version skip,
    # collision) can reference it without re-parsing meta.
    short_id = meta.get("short_id", "")
    if not isinstance(short_id, str) or not short_id:
        msg = "ALGORITHM_META['short_id'] must be a non-empty string"
        if verbose:
            print(f"[plugins] {name}: {msg}", file=sys.stderr)
        return LoadResult("failed", error=msg)

    if not hasattr(module, "run") or not callable(module.run):
        msg = "missing callable 'run' function"
        if verbose:
            print(f"[plugins] {name}: {msg}", file=sys.stderr)
        return LoadResult("failed", error=msg)

    # ---- api_version ----
    if "api_version" not in meta:
        msg = "ALGORITHM_META is missing 'api_version' (current: 1)"
        if verbose:
            print(f"[plugins] {name}: {msg}", file=sys.stderr)
        return LoadResult("failed", error=msg)
    if meta["api_version"] != CURRENT_API_VERSION:
        msg = (
            f"api_version={meta['api_version']} is not supported "
            f"(current: {CURRENT_API_VERSION})"
        )
        if verbose:
            print(f"[plugins] {name}: {msg}", file=sys.stderr)
        return LoadResult("skipped", short_id=short_id, error=msg)

    # ---- short_id collision ----
    if short_id in ALGORITHM_REGISTRY:
        if short_id in _PLUGIN_SHORT_IDS:
            msg = f"already loaded as {short_id!r}"
        else:
            msg = f"short_id {short_id!r} collides with an existing algorithm"
        if verbose:
            print(f"[plugins] {name}: {msg}", file=sys.stderr)
        return LoadResult("skipped", short_id=short_id, error=msg)

    # ---- META validation (uses registry's _validate_meta) ----
    try:
        _validate_meta(meta)
    except ValueError as e:
        msg = str(e)
        if verbose:
            print(f"[plugins] {name}: {msg}", file=sys.stderr)
        return LoadResult("failed", error=msg)

    # ---- Plugin-specific rules ----
    try:
        _check_param_maps_to(meta)
        _check_run_signature(module.run)
    except ValueError as e:
        msg = str(e)
        if verbose:
            print(f"[plugins] {name}: {msg}", file=sys.stderr)
        return LoadResult("failed", error=msg)

    # ---- Register ----
    try:
        register(short_id, run=module.run, meta=meta)
    except ValueError as e:
        msg = str(e)
        if verbose:
            print(f"[plugins] {name}: {msg}", file=sys.stderr)
        return LoadResult("failed", error=msg)

    _PLUGIN_SHORT_IDS.add(short_id)
    if verbose:
        print(f"[plugins] loaded {short_id!r} from {name}")
    return LoadResult("loaded", short_id=short_id)


# ---------------------------------------------------------------------
#  Discovery
# ---------------------------------------------------------------------

def discover_plugins(*, verbose: bool = False) -> PluginLoadReport:
    """Load every .py in the plugins directory.

    Safe to call multiple times: already-loaded plugins are skipped
    (their files remain registered).  Returns a report of what
    happened.  Stores the report for ``last_report()``.
    """
    global _LAST_REPORT
    report = PluginLoadReport()

    if not plugins_enabled():
        report.enabled = False
        if verbose:
            print("[plugins] discovery disabled via "
                  "WDS_OPTIMIZER_DISABLE_PLUGINS=1")
        _LAST_REPORT = report
        return report

    pdir = plugins_dir()
    try:
        pdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        report.warnings.append(
            f"could not create plugins directory {pdir}: {e}"
        )
        if verbose:
            print(f"[plugins] {report.warnings[-1]}", file=sys.stderr)
        _LAST_REPORT = report
        return report

    py_files = sorted(
        p for p in pdir.glob("*.py") if p.name != "__init__.py"
    )
    if not py_files:
        if verbose:
            print(f"[plugins] no plugins in {pdir}")
        _LAST_REPORT = report
        return report

    for py in py_files:
        result = load_plugin_from_file(py, verbose=verbose)
        if result.status == "loaded":
            report.loaded.append(result.short_id)
        elif result.status == "skipped":
            report.skipped.append((str(py), result.error))
        else:
            report.failed.append((str(py), result.error))

    _LAST_REPORT = report
    return report

def reset_plugins() -> None:
    """Unregister all plugins and clear the last discovery report.

    Used by the GUI's "Reload plugins" button: after reset, the next
    ``discover_plugins()`` call rescans the directory from scratch.
    Built-in algorithms (GA, PSO, DE, SADE) are never touched.

    The CLI does not need this — each invocation runs in a fresh
    process.
    """
    global _LAST_REPORT
    for sid in list(_PLUGIN_SHORT_IDS):
        ALGORITHM_REGISTRY.pop(sid, None)
    _PLUGIN_SHORT_IDS.clear()
    _LAST_REPORT = None

# ---------------------------------------------------------------------
#  Test-only reset
# ---------------------------------------------------------------------

def _reset_for_tests() -> None:
    """Alias for reset_plugins().  Kept for backward compatibility
    with test code written before reset_plugins was public."""
    reset_plugins()