"""Tests for plugin discovery, loading, and validation."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from wds_optimizer.algorithms.registry import (
    ALGORITHM_REGISTRY, list_algorithms,
)
from wds_optimizer import plugins as plugins_mod
from wds_optimizer.plugins import (
    LoadResult, PluginLoadReport,
    discover_plugins, get_plugin_short_ids,
    last_report, load_plugin_from_file, plugins_dir, plugins_enabled,
    _reset_for_tests,
)


# ---------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_plugins(tmp_path, monkeypatch):
    """Fresh plugins dir per test + automatic cleanup."""
    pdir = tmp_path / "plugins"
    pdir.mkdir()
    monkeypatch.setenv("WDS_OPTIMIZER_PLUGINS_DIR", str(pdir))
    monkeypatch.delenv("WDS_OPTIMIZER_DISABLE_PLUGINS", raising=False)
    _reset_for_tests()
    yield pdir
    _reset_for_tests()


VALID_PLUGIN = textwrap.dedent("""\
    import numpy as np

    ALGORITHM_META = {
        "short_id": "TestAlgo",
        "display_name": "Test Algorithm",
        "category": "Custom",
        "paper_faithful": False,
        "description": "A test plugin.",
        "references": ["-"],
        "params": [
            {"key": "my_p", "name": "p", "display_name": "P",
             "type": "float", "default": 0.5,
             "min": 0.0, "max": 1.0, "step": 0.1,
             "maps_to": None,
             "description": "test param"},
        ],
        "constraint_policy": "deb_prefers",
        "api_version": 1,
    }

    def run(problem, config, *, progress_callback=None,
            cancel_callback=None, meta=None):
        return 0.0, None, None, True
""")


def _write(pdir: Path, name: str, content: str) -> Path:
    p = pdir / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------
#  Directory and enable/disable
# ---------------------------------------------------------------------

def test_plugins_dir_uses_env_var(isolated_plugins):
    assert plugins_dir() == isolated_plugins.resolve()


def test_plugins_dir_default(tmp_path, monkeypatch):
    monkeypatch.delenv("WDS_OPTIMIZER_PLUGINS_DIR", raising=False)
    expected = Path.home() / ".wds_optimizer" / "plugins"
    assert plugins_dir() == expected


def test_plugins_enabled_default():
    assert plugins_enabled() is True


def test_plugins_disabled_via_env(monkeypatch):
    monkeypatch.setenv("WDS_OPTIMIZER_DISABLE_PLUGINS", "1")
    report = discover_plugins()
    assert report.enabled is False
    assert report.loaded == []


# ---------------------------------------------------------------------
#  Load one file
# ---------------------------------------------------------------------

def test_load_valid_plugin(isolated_plugins):
    p = _write(isolated_plugins, "test_algo.py", VALID_PLUGIN)
    result = load_plugin_from_file(p)
    assert result.status == "loaded"
    assert result.short_id == "TestAlgo"
    assert "TestAlgo" in list_algorithms()
    assert "TestAlgo" in get_plugin_short_ids()


def test_syntax_error_returns_failed(isolated_plugins):
    p = _write(isolated_plugins, "bad.py", "def broken(:\n  pass\n")
    result = load_plugin_from_file(p)
    assert result.status == "failed"
    assert "Syntax error" in result.error


def test_missing_meta_returns_failed(isolated_plugins):
    p = _write(isolated_plugins, "nom.py", "def run(): pass\n")
    result = load_plugin_from_file(p)
    assert result.status == "failed"
    assert "ALGORITHM_META" in result.error


def test_missing_run_returns_failed(isolated_plugins):
    src = VALID_PLUGIN.replace(
        "def run(problem, config, *, progress_callback=None,\n"
        "        cancel_callback=None, meta=None):\n"
        "    return 0.0, None, None, True",
        "",
    )
    p = _write(isolated_plugins, "norun.py", src)
    result = load_plugin_from_file(p)
    assert result.status == "failed"
    assert "run" in result.error


def test_missing_api_version_returns_failed(isolated_plugins):
    src = VALID_PLUGIN.replace('"api_version": 1,', "")
    p = _write(isolated_plugins, "noapi.py", src)
    result = load_plugin_from_file(p)
    assert result.status == "failed"
    assert "api_version" in result.error


def test_wrong_api_version_skipped(isolated_plugins):
    src = VALID_PLUGIN.replace('"api_version": 1,', '"api_version": 99,')
    p = _write(isolated_plugins, "v99.py", src)
    result = load_plugin_from_file(p)
    assert result.status == "skipped"
    assert "api_version=99" in result.error


def test_short_id_collision_skipped(isolated_plugins):
    src = VALID_PLUGIN.replace('"short_id": "TestAlgo",',
                                '"short_id": "GA",')
    p = _write(isolated_plugins, "collide.py", src)
    result = load_plugin_from_file(p)
    assert result.status == "skipped"
    assert "collides" in result.error


def test_maps_to_not_none_rejected(isolated_plugins):
    src = VALID_PLUGIN.replace('"maps_to": None,',
                                '"maps_to": ("GA", "Pc"),')
    p = _write(isolated_plugins, "badmap.py", src)
    result = load_plugin_from_file(p)
    assert result.status == "failed"
    assert "maps_to" in result.error


def test_bad_run_signature_rejected(isolated_plugins):
    src = VALID_PLUGIN.replace(
        "def run(problem, config, *, progress_callback=None,\n"
        "        cancel_callback=None, meta=None):\n"
        "    return 0.0, None, None, True",
        "def run(problem):\n    return 0.0, None, None, True",
    )
    p = _write(isolated_plugins, "badsig.py", src)
    result = load_plugin_from_file(p)
    assert result.status == "failed"
    assert "run()" in result.error


# ---------------------------------------------------------------------
#  Discovery
# ---------------------------------------------------------------------

def test_discover_empty_dir(isolated_plugins):
    report = discover_plugins()
    assert report.enabled is True
    assert report.loaded == []
    assert report.failed == []


def test_discover_loads_all(isolated_plugins):
    _write(isolated_plugins, "a.py", VALID_PLUGIN)
    src_b = VALID_PLUGIN.replace('"short_id": "TestAlgo",',
                                 '"short_id": "AlgoB",')
    _write(isolated_plugins, "b.py", src_b)

    report = discover_plugins(verbose=False)
    assert set(report.loaded) == {"TestAlgo", "AlgoB"}
    assert "TestAlgo" in get_plugin_short_ids()
    assert "AlgoB" in get_plugin_short_ids()


def test_discover_partial_failure(isolated_plugins):
    _write(isolated_plugins, "good.py", VALID_PLUGIN)
    _write(isolated_plugins, "bad.py", "def oops(:\n  pass")
    report = discover_plugins()
    assert len(report.loaded) == 1
    assert len(report.failed) == 1
    assert "Syntax" in report.failed[0][1]


def test_discover_idempotent(isolated_plugins):
    """Running discovery twice does not re-register (no duplicates)."""
    _write(isolated_plugins, "a.py", VALID_PLUGIN)
    discover_plugins()
    report2 = discover_plugins()
    # Second run should report skipped (already loaded)
    assert "TestAlgo" in report2.loaded or any(
        "TestAlgo" in s[1] for s in report2.skipped
    )
    # At most one registration in ALGORITHM_REGISTRY
    count = sum(1 for k in ALGORITHM_REGISTRY if k == "TestAlgo")
    assert count == 1


def test_last_report_after_discovery(isolated_plugins):
    _write(isolated_plugins, "a.py", VALID_PLUGIN)
    discover_plugins()
    r = last_report()
    assert "TestAlgo" in r.loaded


def test_reset_for_tests(isolated_plugins):
    _write(isolated_plugins, "a.py", VALID_PLUGIN)
    discover_plugins()
    assert "TestAlgo" in ALGORITHM_REGISTRY

    _reset_for_tests()
    assert "TestAlgo" not in ALGORITHM_REGISTRY
    assert "TestAlgo" not in get_plugin_short_ids()