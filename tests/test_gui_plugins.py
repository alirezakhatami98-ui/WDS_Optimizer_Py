"""Tests for the GUI-integration helpers of the plugin system.

Does not exercise Streamlit rendering.  Covers:
- _algo_display_label for built-in and plugin algorithms
- _discover_plugins_if_needed sets the flag and calls discover once
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from wds_optimizer.algorithms.registry import ALGORITHM_REGISTRY
from wds_optimizer import plugins as plugins_mod
from wds_optimizer.plugins import (
    _reset_for_tests, discover_plugins,
)


with patch("streamlit.set_page_config"):
    from wds_optimizer.app import streamlit_app
    from wds_optimizer.app.streamlit_app import (
        _algo_display_label, _discover_plugins_if_needed,
    )


VALID_PLUGIN = textwrap.dedent("""\
    ALGORITHM_META = {
        "short_id": "GuiTestAlgo",
        "display_name": "My Gui Test Algo",
        "category": "Custom",
        "paper_faithful": False,
        "description": "GUI test.",
        "references": ["-"],
        "params": [],
        "constraint_policy": "deb_prefers",
        "api_version": 1,
    }
    def run(problem, config, *, progress_callback=None,
            cancel_callback=None, meta=None):
        return 0.0, None, None, True
""")


class FakeSessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v


@pytest.fixture(autouse=True)
def isolated_plugins(tmp_path, monkeypatch):
    pdir = tmp_path / "plugins"
    pdir.mkdir()
    monkeypatch.setenv("WDS_OPTIMIZER_PLUGINS_DIR", str(pdir))
    monkeypatch.delenv("WDS_OPTIMIZER_DISABLE_PLUGINS", raising=False)
    _reset_for_tests()
    yield pdir
    _reset_for_tests()


# ---------------------------------------------------------------------
#  _algo_display_label
# ---------------------------------------------------------------------

def test_algo_label_builtin():
    for sid in ("GA", "PSO", "DE", "SADE"):
        label = _algo_display_label(sid)
        assert not label.startswith("[P]")
        assert sid in label or label == sid


def test_algo_label_plugin(isolated_plugins):
    p = isolated_plugins / "gui.py"
    p.write_text(VALID_PLUGIN, encoding="utf-8")
    discover_plugins()

    label = _algo_display_label("GuiTestAlgo")
    assert label.startswith("[P]")
    assert "My Gui Test Algo" in label


def test_algo_label_unknown_id():
    """Unknown id falls back to the id itself (defensive)."""
    assert _algo_display_label("NotARealAlgo") == "NotARealAlgo"


# ---------------------------------------------------------------------
#  _discover_plugins_if_needed
# ---------------------------------------------------------------------

def test_discover_runs_once(isolated_plugins, monkeypatch):
    fake = FakeSessionState()
    monkeypatch.setattr("streamlit.session_state", fake)

    call_count = {"n": 0}
    original = streamlit_app.discover_plugins

    def counting_discover(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        streamlit_app, "discover_plugins", counting_discover,
    )

    _discover_plugins_if_needed()
    assert fake["_plugins_discovered"] is True
    assert call_count["n"] == 1

    # Second call should skip
    _discover_plugins_if_needed()
    assert call_count["n"] == 1


def test_discover_survives_exception(isolated_plugins, monkeypatch):
    """If discover_plugins raises, the flag is still set and an error is stored."""
    fake = FakeSessionState()
    monkeypatch.setattr("streamlit.session_state", fake)

    def boom(*a, **k):
        raise RuntimeError("simulated discovery failure")

    monkeypatch.setattr(streamlit_app, "discover_plugins", boom)

    _discover_plugins_if_needed()
    assert fake["_plugins_discovered"] is True
    assert "simulated discovery failure" in fake["_plugins_error"]