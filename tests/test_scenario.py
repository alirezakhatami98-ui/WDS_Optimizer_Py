"""Tests for the Scenario data model and serialization."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wds_optimizer.scenario import (
    Scenario, Workspace,
    new_scenario, clone_scenario,
    save_scenario, load_scenario,
    save_workspace, load_workspace,
)


# ---------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------

def _constraints():
    return {
        "Pmin": {"enabled": True, "value": 30.0},
        "Pmax": {"enabled": False, "value": 100.0},
        "Vmin": {"enabled": False, "value": 0.0},
        "Vmax": {"enabled": True, "value": 2.5},
    }


def _algo_params():
    return {
        "GA": {"ga_pc": 0.8, "ga_pm": 0.03, "ga_sel": "tournament"},
    }


def _minimal_files(tmp_path):
    inp = tmp_path / "net.inp"
    inp.write_text("[TITLE]\nminimal\n[END]\n")
    d = tmp_path / "D.txt"
    d.write_text("150\n200\n")
    c = tmp_path / "Cost.txt"
    c.write_text("100\n150\n")
    return {"inp": str(inp), "d": str(d), "cost": str(c)}


def _make_scenario(tmp_path, name="test", **overrides):
    files = _minimal_files(tmp_path)
    kw = dict(
        inp_path=files["inp"], d_path=files["d"], cost_path=files["cost"],
        diam_unit="mm",
        algorithms=["GA"], n_runs=1, seed=42,
        ns=10, max_gen=5,
        constraints=_constraints(),
        algorithm_params=_algo_params(),
    )
    kw.update(overrides)
    return new_scenario(name, **kw)


# ---------------------------------------------------------------------
#  Scenario creation
# ---------------------------------------------------------------------

def test_new_scenario_has_uuid(tmp_path):
    sc = _make_scenario(tmp_path)
    assert len(sc.scenario_id) == 32  # UUID4 hex
    assert sc.name == "test"
    assert sc.status == "Draft"
    assert sc.experiment is None
    assert sc.created_at  # ISO timestamp


def test_new_scenario_deep_copies_dicts(tmp_path):
    """Mutating the input dicts after creation must not affect the scenario."""
    files = _minimal_files(tmp_path)
    constraints = _constraints()
    algo_params = _algo_params()

    sc = new_scenario(
        "test",
        inp_path=files["inp"], d_path=files["d"], cost_path=files["cost"],
        diam_unit="mm", algorithms=["GA"], n_runs=1, seed=42,
        ns=10, max_gen=5,
        constraints=constraints, algorithm_params=algo_params,
    )
    constraints["Pmin"]["value"] = 999
    algo_params["GA"]["ga_pc"] = 0.0

    assert sc.constraints["Pmin"]["value"] == 30.0
    assert sc.algorithm_params["GA"]["ga_pc"] == 0.8

def test_to_dict_returns_isolated_copy(tmp_path):
    """to_dict must not leak references to the scenario's internals."""
    from wds_optimizer.scenario import to_dict
    sc = _make_scenario(tmp_path)
    data = to_dict(sc, embed_inputs=False)
    data["scenario"]["constraints"]["Pmin"]["value"] = 999
    data["scenario"]["algorithm_params"]["GA"]["ga_pc"] = 0.0
    assert sc.constraints["Pmin"]["value"] == 30.0
    assert sc.algorithm_params["GA"]["ga_pc"] == 0.8

def test_clone_has_different_id_and_no_result(tmp_path):
    sc = _make_scenario(tmp_path)
    # Pretend it was already run
    sc.status = "Complete"
    sc.experiment = object()  # dummy

    clone = clone_scenario(sc, "clone")
    assert clone.scenario_id != sc.scenario_id
    assert clone.name == "clone"
    assert clone.status == "Draft"
    assert clone.experiment is None
    # Config was copied
    assert clone.algorithms == sc.algorithms
    assert clone.constraints == sc.constraints


# ---------------------------------------------------------------------
#  Round-trip
# ---------------------------------------------------------------------

def test_scenario_json_roundtrip(tmp_path):
    sc = _make_scenario(tmp_path, name="hanoi-ga")
    path = tmp_path / "scenario.json"
    save_scenario(sc, path)

    loaded = load_scenario(path)
    assert loaded.scenario_id == sc.scenario_id
    assert loaded.name == sc.name
    assert loaded.algorithms == sc.algorithms
    assert loaded.constraints == sc.constraints
    assert loaded.algorithm_params == sc.algorithm_params
    assert loaded.constraint_policy == "deb_prefers"


def test_scenario_embeds_inputs_by_default(tmp_path):
    sc = _make_scenario(tmp_path)
    path = tmp_path / "scenario.json"
    save_scenario(sc, path)

    data = json.loads(path.read_text())
    assert "inputs" in data
    assert data["inputs"]["inp"].startswith("[TITLE]")
    assert "150" in data["inputs"]["D_txt"]


def test_scenario_can_skip_embed_inputs(tmp_path):
    sc = _make_scenario(tmp_path)
    path = tmp_path / "scenario.json"
    save_scenario(sc, path, embed_inputs=False)
    data = json.loads(path.read_text())
    assert "inputs" not in data


def test_scenario_missing_input_files_embeds_empty(tmp_path):
    """Embedded inputs are '' when the source file is gone."""
    files = _minimal_files(tmp_path)
    sc = new_scenario(
        "t",
        inp_path=files["inp"], d_path="/nonexistent/D.txt",
        cost_path=files["cost"], diam_unit="mm",
        algorithms=["GA"], n_runs=1, seed=42, ns=10, max_gen=5,
        constraints=_constraints(), algorithm_params=_algo_params(),
    )
    path = tmp_path / "sc.json"
    save_scenario(sc, path)  # must not raise
    data = json.loads(path.read_text())
    assert data["inputs"]["D_txt"] == ""


def test_scenario_with_no_experiment_serializes(tmp_path):
    sc = _make_scenario(tmp_path)
    path = tmp_path / "sc.json"
    save_scenario(sc, path)
    data = json.loads(path.read_text())
    assert "experiment" not in data


# ---------------------------------------------------------------------
#  Workspace
# ---------------------------------------------------------------------

def test_workspace_add_and_active(tmp_path):
    ws = Workspace()
    sc1 = _make_scenario(tmp_path, "one")
    sc2 = _make_scenario(tmp_path, "two")

    ws.add(sc1)
    assert ws.active_scenario_id == sc1.scenario_id
    assert len(ws.list()) == 1

    ws.add(sc2)
    # Second add activates only if activate=True (default)
    assert ws.active_scenario_id == sc2.scenario_id

    ws.add(_make_scenario(tmp_path, "three"), activate=False)
    assert ws.active_scenario_id == sc2.scenario_id


def test_workspace_remove_active_falls_back(tmp_path):
    ws = Workspace()
    sc1 = _make_scenario(tmp_path, "one")
    sc2 = _make_scenario(tmp_path, "two")
    ws.add(sc1); ws.add(sc2)
    assert ws.active_scenario_id == sc2.scenario_id

    ws.remove(sc2.scenario_id)
    # Falls back to the remaining scenario
    assert ws.active_scenario_id == sc1.scenario_id


def test_workspace_remove_last_clears_active(tmp_path):
    ws = Workspace()
    sc = _make_scenario(tmp_path)
    ws.add(sc)
    ws.remove(sc.scenario_id)
    assert ws.active_scenario_id is None
    assert ws.active() is None


def test_workspace_roundtrip(tmp_path):
    ws = Workspace()
    s1 = _make_scenario(tmp_path, "a")
    s2 = _make_scenario(tmp_path, "b")
    ws.add(s1); ws.add(s2)

    path = tmp_path / "workspace.json"
    save_workspace(ws, path)

    loaded = load_workspace(path)
    assert len(loaded.list()) == 2
    assert loaded.active_scenario_id == ws.active_scenario_id
    names = {sc.name for sc in loaded.list()}
    assert names == {"a", "b"}


def test_workspace_empty_roundtrip(tmp_path):
    ws = Workspace()
    path = tmp_path / "ws.json"
    save_workspace(ws, path)
    loaded = load_workspace(path)
    assert loaded.list() == []
    assert loaded.active_scenario_id is None


def test_workspace_duplicate_id_raises(tmp_path):
    ws = Workspace()
    sc = _make_scenario(tmp_path)
    ws.add(sc)
    with pytest.raises(ValueError, match="already in workspace"):
        ws.add(sc)

def test_workspace_to_dict_structure(tmp_path):
    """workspace_to_dict returns a serializable dict with all scenarios."""
    from wds_optimizer.scenario import workspace_to_dict
    ws = Workspace()
    ws.add(_make_scenario(tmp_path, "a"))
    ws.add(_make_scenario(tmp_path, "b"))
    data = workspace_to_dict(ws)
    assert data["version"] == "0.4.0"
    assert data["active_scenario_id"] == ws.active_scenario_id
    assert len(data["scenarios"]) == 2
    names = {entry["scenario"]["name"] for entry in data["scenarios"]}
    assert names == {"a", "b"}

# ---------------------------------------------------------------------
#  GUI helper: _scenario_from_sidebar
# ---------------------------------------------------------------------

def test_scenario_from_sidebar_reads_widget_keys(tmp_path, monkeypatch):
    """_scenario_from_sidebar must read widget keys, not canonical keys.

    Regression for a bug where the helper read st.session_state.pmin
    (canonical, stale unless a run has synced it) instead of
    st.session_state.pmin_input (the widget value the user is looking
    at).
    """
    from unittest.mock import patch as _patch
    import streamlit as st

    with _patch("streamlit.set_page_config"):
        from wds_optimizer.app import streamlit_app
        from wds_optimizer.app.streamlit_app import _scenario_from_sidebar

    class FakeSession(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)
        def __setattr__(self, k, v):
            self[k] = v

    fake = FakeSession(
        inp_path=str(tmp_path / "net.inp"),
        d_path=str(tmp_path / "D.txt"),
        cost_path=str(tmp_path / "Cost.txt"),
        diam_unit="mm",
        network_units={
            "System": "SI", "PressureUnit": "m",
            "VelocityUnit": "m/s", "LengthUnit": "m",
            "DiameterUnit": "mm",
        },
        # Widget keys: what the user currently sees
        pmin_input=45.0,
        pmax_input=100.0,
        vmin_input=0.0,
        vmax_input=2.5,
        use_pmin_input=True,
        use_pmax_input=False,
        use_vmin_input=False,
        use_vmax_input=True,
        algo_input=["GA"],
        ns_input=100,
        max_gen_input=200,
        n_runs_input=3,
        exp_seed_input=7,
        constraint_policy_input="deb_prefers",
        fixed_pipes_input="",
        # Stale canonical keys — should be IGNORED by the helper
        pmin=30.0, pmax=100.0, vmin=0.0, vmax=2.5,
        use_pmin=True, use_pmax=False, use_vmin=False, use_vmax=True,
        ns=50, max_gen=500, n_runs=1, exp_seed=1,
        ga_pc_input=0.8, ga_pm_input=0.03, ga_sel_input="tournament",
        sade_fmin_input=0.1, sade_fmax_input=0.9,
        sade_crmin_input=0.1, sade_crmax_input=0.9,
        sade_conv_input=True, sade_tol_input=1e-6, sade_stag_input=300,
        pso_w_input=0.7, pso_c1_input=1.5, pso_c2_input=1.5,
        de_f_input=0.5, de_cr_input=0.3, de_strategy_input="rand/1/bin",
    )
    # Existing files must exist for from_sidebar to succeed
    (tmp_path / "net.inp").write_text("[TITLE]\nt\n[END]\n")
    (tmp_path / "D.txt").write_text("150\n")
    (tmp_path / "Cost.txt").write_text("100\n")

    monkeypatch.setattr(st, "session_state", fake)
    sc = _scenario_from_sidebar("test")

    # Values from widget keys (not canonical)
    assert sc.ns == 100            # not 50
    assert sc.max_gen == 200       # not 500
    assert sc.n_runs == 3          # not 1
    assert sc.seed == 7            # not 1
    assert sc.constraints["Pmin"]["value"] == 45.0   # not 30.0
    assert sc.algorithm_params["GA"]["ga_pc"] == 0.8