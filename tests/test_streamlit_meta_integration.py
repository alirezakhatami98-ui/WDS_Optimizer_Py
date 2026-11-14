"""Tests for the META-driven GUI config builder.

Covers the two helpers that replaced the hard-coded param assignments
in streamlit_app.py:

- _apply_param_to_config: map a single META param onto the config.
- _build_base_cfg: assemble the full config from session_state + META.

Design
------
No Streamlit runtime.  st.session_state is replaced with a dict-like
FakeSessionState (same pattern used in test_prepare_problem_roundtrip)
via monkeypatch.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


with patch("streamlit.set_page_config"):
    from wds_optimizer.app import streamlit_app
    from wds_optimizer.app.streamlit_app import (
        _apply_param_to_config, _build_base_cfg,
    )

from wds_optimizer.algorithms.config import build_algorithm_config


class FakeSessionState(dict):
    """dict with attribute access, matching real st.session_state."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


@pytest.fixture
def fake_state(monkeypatch):
    fake = FakeSessionState(
        ns=50, max_gen=100,
        algo_input=["GA"],
    )
    monkeypatch.setattr("streamlit.session_state", fake)
    return fake


# ---------------------------------------------------------------------
#  _apply_param_to_config
# ---------------------------------------------------------------------

def test_apply_maps_to_family_field():
    cfg = build_algorithm_config(NS=10, MaxGen=10)
    p = {"key": "ga_pc", "maps_to": ("GA", "Pc")}
    _apply_param_to_config(cfg, "GA", p, 0.55)
    assert cfg.GA.Pc == 0.55


def test_apply_stores_in_extra_when_no_maps_to():
    cfg = build_algorithm_config(NS=10, MaxGen=10)
    p = {"key": "my_param"}          # no maps_to
    _apply_param_to_config(cfg, "MY_ALGO", p, 42)
    assert cfg.extra["MY_ALGO"]["my_param"] == 42


def test_apply_raises_on_bad_maps_to():
    cfg = build_algorithm_config(NS=10, MaxGen=10)
    p = {"key": "bogus", "maps_to": ("GA", "NonexistentField")}
    with pytest.raises(ValueError, match="does not exist"):
        _apply_param_to_config(cfg, "GA", p, 1.0)


# ---------------------------------------------------------------------
#  _build_base_cfg — end to end
# ---------------------------------------------------------------------

def test_build_base_cfg_populates_ga_params(fake_state):
    fake_state["ga_pc_input"] = 0.55
    fake_state["ga_pm_input"] = 0.02
    fake_state["ga_sel_input"] = "roulette"
    cfg = _build_base_cfg()
    assert cfg.GA.Pc == 0.55
    assert cfg.GA.Pm == 0.02
    assert cfg.GA.Selection == "roulette"


def test_build_base_cfg_populates_sade_params(fake_state):
    fake_state["algo_input"] = ["SADE"]
    fake_state["sade_fmin_input"] = 0.2
    fake_state["sade_fmax_input"] = 0.8
    fake_state["sade_crmin_input"] = 0.15
    fake_state["sade_crmax_input"] = 0.85
    fake_state["sade_conv_input"] = True
    fake_state["sade_tol_input"] = 1e-5
    fake_state["sade_stag_input"] = 500
    cfg = _build_base_cfg()
    assert cfg.SADE.Fmin == 0.2
    assert cfg.SADE.Fmax == 0.8
    assert cfg.SADE.CRmin == 0.15
    assert cfg.SADE.CRmax == 0.85
    assert cfg.SADE.UseConvergence is True
    assert cfg.SADE.Tol == 1e-5
    assert cfg.SADE.StagnationMax == 500


def test_build_base_cfg_validates_sade_fmin_le_fmax(fake_state):
    fake_state["algo_input"] = ["SADE"]
    fake_state["sade_fmin_input"] = 0.9
    fake_state["sade_fmax_input"] = 0.1
    fake_state["sade_crmin_input"] = 0.1
    fake_state["sade_crmax_input"] = 0.9
    fake_state["sade_conv_input"] = True
    fake_state["sade_tol_input"] = 1e-6
    fake_state["sade_stag_input"] = 300
    with pytest.raises(ValueError, match="SADE Fmin"):
        _build_base_cfg()


def test_build_base_cfg_restart_always_disabled(fake_state):
    """Restart is hard-disabled in the GUI (see CHANGELOG v0.2.1)."""
    fake_state["ga_pc_input"] = 0.8
    fake_state["ga_pm_input"] = 0.03
    fake_state["ga_sel_input"] = "tournament"
    cfg = _build_base_cfg()
    assert cfg.Restart.Enabled is False


def test_build_base_cfg_only_selected_algorithms(fake_state):
    """Only algorithms in ``algo_input`` have their params applied."""
    fake_state["algo_input"] = ["GA"]
    fake_state["ga_pc_input"] = 0.55
    # If SADE were wrongly included, this would override cfg.SADE.Fmin.
    fake_state["sade_fmin_input"] = 0.99
    cfg = _build_base_cfg()
    assert cfg.GA.Pc == 0.55
    # cfg.SADE.Fmin should remain the class default (0.1), not 0.99.
    assert cfg.SADE.Fmin == 0.1