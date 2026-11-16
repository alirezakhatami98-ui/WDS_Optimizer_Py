"""Tests for the History data model and serialization."""
from __future__ import annotations

import json

import pytest

from wds_optimizer.history import (
    HistoryEntry,
    auto_name, history_to_dict, load_entry, load_history,
    new_entry, save_entry, save_history, to_dict,
)


def _constraints():
    return {
        "Pmin": {"enabled": True, "value": 30.0},
        "Pmax": {"enabled": False, "value": 100.0},
        "Vmin": {"enabled": False, "value": 0.0},
        "Vmax": {"enabled": True, "value": 2.5},
    }


def _algo_params():
    return {"GA": {"ga_pc": 0.8, "ga_pm": 0.03}}


def _minimal_files(tmp_path):
    inp = tmp_path / "net.inp"; inp.write_text("[TITLE]\nt\n[END]\n")
    d = tmp_path / "D.txt"; d.write_text("150\n")
    c = tmp_path / "Cost.txt"; c.write_text("100\n")
    return {"inp": str(inp), "d": str(d), "cost": str(c)}


def _make_entry(tmp_path, name="test", **overrides):
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
    return new_entry(name, **kw)


def test_new_entry_has_uuid_and_defaults(tmp_path):
    e = _make_entry(tmp_path)
    assert len(e.entry_id) == 32
    assert e.name == "test"
    assert e.status == "Complete"
    assert e.experiment is None


def test_new_entry_deep_copies(tmp_path):
    files = _minimal_files(tmp_path)
    constraints = _constraints()
    algo_params = _algo_params()
    e = new_entry(
        "t", inp_path=files["inp"], d_path=files["d"],
        cost_path=files["cost"], diam_unit="mm",
        algorithms=["GA"], n_runs=1, seed=42, ns=10, max_gen=5,
        constraints=constraints, algorithm_params=algo_params,
    )
    constraints["Pmin"]["value"] = 999
    algo_params["GA"]["ga_pc"] = 0.0
    assert e.constraints["Pmin"]["value"] == 30.0
    assert e.algorithm_params["GA"]["ga_pc"] == 0.8


def test_auto_name_format():
    n = auto_name(
        algorithms=["GA", "SADE"], inp_path="/path/to/hanoi.inp",
        ns=100, seed=42,
    )
    assert n == "GA+SADE / hanoi / ns=100 / seed=42"


def test_to_dict_isolates_from_entry(tmp_path):
    e = _make_entry(tmp_path)
    data = to_dict(e, embed_inputs=False)
    data["entry"]["constraints"]["Pmin"]["value"] = 999
    assert e.constraints["Pmin"]["value"] == 30.0


def test_entry_roundtrip(tmp_path):
    e = _make_entry(tmp_path, name="my run")
    path = tmp_path / "e.json"
    save_entry(e, path)
    loaded = load_entry(path)
    assert loaded.entry_id == e.entry_id
    assert loaded.name == "my run"
    assert loaded.constraints == e.constraints
    assert loaded.algorithm_params == e.algorithm_params


def test_entry_embeds_inputs(tmp_path):
    e = _make_entry(tmp_path)
    path = tmp_path / "e.json"
    save_entry(e, path)
    data = json.loads(path.read_text())
    assert data["inputs"]["inp"].startswith("[TITLE]")


def test_entry_skip_embed(tmp_path):
    e = _make_entry(tmp_path)
    path = tmp_path / "e.json"
    save_entry(e, path, embed_inputs=False)
    assert "inputs" not in json.loads(path.read_text())


def test_entry_missing_file_embeds_empty(tmp_path):
    files = _minimal_files(tmp_path)
    e = new_entry(
        "t", inp_path=files["inp"], d_path="/gone.txt",
        cost_path=files["cost"], diam_unit="mm",
        algorithms=["GA"], n_runs=1, seed=1, ns=10, max_gen=5,
        constraints=_constraints(), algorithm_params=_algo_params(),
    )
    path = tmp_path / "e.json"
    save_entry(e, path)          # must not raise
    assert json.loads(path.read_text())["inputs"]["D_txt"] == ""


def test_history_roundtrip(tmp_path):
    e1 = _make_entry(tmp_path, "first")
    e2 = _make_entry(tmp_path, "second")
    path = tmp_path / "hist.json"
    save_history([e1, e2], path)
    loaded = load_history(path)
    assert len(loaded) == 2
    assert {e.name for e in loaded} == {"first", "second"}


def test_history_to_dict_structure(tmp_path):
    e = _make_entry(tmp_path, "x")
    d = history_to_dict([e])
    assert d["version"] == "0.4.1a"
    assert len(d["entries"]) == 1
    assert d["entries"][0]["entry"]["name"] == "x"


def test_empty_history_roundtrip(tmp_path):
    path = tmp_path / "h.json"
    save_history([], path)
    assert load_history(path) == []

def test_load_history_accepts_single_entry(tmp_path):
    """Uploading a single-entry JSON via Load returns a one-item list."""
    e = _make_entry(tmp_path, "single")
    path = tmp_path / "entry.json"
    save_entry(e, path)
    loaded = load_history(path)
    assert len(loaded) == 1
    assert loaded[0].name == "single"