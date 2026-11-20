"""Tests for the dynamic sweep catalogue and parameter application."""
from __future__ import annotations

import copy
from unittest.mock import patch

import numpy as np
import pytest

from wds_optimizer.algorithms.config import build_algorithm_config
from wds_optimizer.algorithms.registry import ALGORITHM_REGISTRY
from wds_optimizer.sensitivity.config import (
    LEGACY_MAP, build_sensitivity_parameters, normalize_param_name,
    parse_sensitivity_values,
)
from wds_optimizer.sensitivity.runner import (
    SingleSweepResult, _apply_sensitivity_param, run_single_sweep,
)


# ---------------------------------------------------------------------
#  Catalogue
# ---------------------------------------------------------------------

def test_catalogue_includes_global_params():
    c = build_sensitivity_parameters()
    for key in ("NS", "MaxGen", "n_runs", "seed",
                "constraint_policy",
                "Pmin", "Pmax", "Vmin", "Vmax"):
        assert key in c, f"missing global param {key}"


def test_catalogue_includes_algorithm_params():
    c = build_sensitivity_parameters()
    for key in ("GA.ga_pc", "GA.ga_pm", "GA.ga_sel",
                "PSO.pso_w",
                "DE.de_f", "DE.de_cr",
                "SADE.sade_fmin", "SADE.sade_stag"):
        assert key in c, f"missing algorithm param {key}"


def test_catalogue_param_has_required_fields():
    c = build_sensitivity_parameters()
    for key, info in c.items():
        assert "label" in info, f"{key}: missing label"
        assert "kind" in info, f"{key}: missing kind"
        assert info["kind"] in ("int", "float", "choice", "bool"), (
            f"{key}: unknown kind {info['kind']!r}"
        )


def test_catalogue_reflects_new_plugins():
    """A plugin registered after import shows up in a fresh build."""
    from wds_optimizer.plugins import register as _plugins_register
    # Guard: register the plugin's META directly without touching disk
    fake_meta = {
        "short_id": "TestDynamicAlgo",
        "display_name": "Test Dynamic",
        "category": "Custom",
        "paper_faithful": False,
        "description": "test",
        "references": ["-"],
        "params": [
            {"key": "p1", "name": "p1", "display_name": "P1",
             "type": "float", "default": 0.5,
             "min": 0.0, "max": 1.0, "step": 0.1,
             "maps_to": None, "description": "x"},
        ],
        "constraint_policy": "deb_prefers",
        "api_version": 1,
    }
    _plugins_register("TestDynamicAlgo", run=lambda *a, **k: None,
                      meta=fake_meta)
    try:
        c = build_sensitivity_parameters()
        assert "TestDynamicAlgo.p1" in c
    finally:
        ALGORITHM_REGISTRY.pop("TestDynamicAlgo", None)


# ---------------------------------------------------------------------
#  Legacy aliases
# ---------------------------------------------------------------------

def test_legacy_names_map_to_namespaced():
    assert normalize_param_name("GA_Pc") == "GA.ga_pc"
    assert normalize_param_name("PSO_w") == "PSO.pso_w"
    assert normalize_param_name("DE_CR") == "DE.de_cr"


def test_legacy_unchanged_returns_same():
    assert normalize_param_name("NS") == "NS"
    assert normalize_param_name("GA.ga_pc") == "GA.ga_pc"


# ---------------------------------------------------------------------
#  Value parser
# ---------------------------------------------------------------------

def test_parse_sweep_values_numbers():
    assert parse_sensitivity_values("1, 2, 3") == [1, 2, 3]
    assert parse_sensitivity_values("0.5, 0.6, 0.7") == [0.5, 0.6, 0.7]
    assert parse_sensitivity_values("1, 0.5, 2") == [1, 0.5, 2]


def test_parse_sweep_values_choices():
    vals = parse_sensitivity_values("deb_prefers, fixed_penalty_prefers")
    assert vals == ["deb_prefers", "fixed_penalty_prefers"]


def test_parse_sweep_values_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_sensitivity_values("")

def test_parse_sweep_values_strips_single_quotes():
    """Users may type quoted values; parse them without quotes."""
    vals = parse_sensitivity_values("'deb_prefers', 'fixed_penalty_prefers'")
    assert vals == ["deb_prefers", "fixed_penalty_prefers"]


def test_parse_sweep_values_strips_double_quotes():
    vals = parse_sensitivity_values('"deb_prefers", "fixed_penalty_prefers"')
    assert vals == ["deb_prefers", "fixed_penalty_prefers"]


def test_parse_sweep_values_strips_only_matching_quotes():
    """A leading quote without a matching trailing one is kept."""
    vals = parse_sensitivity_values("'deb_prefers")
    # No pair, so the single leading quote stays.
    assert vals == ["'deb_prefers"]


def test_parse_sweep_values_mixed_quoted_and_unquoted():
    vals = parse_sensitivity_values("'a', b, \"c\"")
    assert vals == ["a", "b", "c"]

# ---------------------------------------------------------------------
#  Applying sweep params
# ---------------------------------------------------------------------

def _fake_problem():
    return {
        "wn": None,
        "_solver": False,
        "Constraints": {
            "Pmin": {"enabled": True,  "value": 30.0},
            "Pmax": {"enabled": False, "value": 100.0},
            "Vmin": {"enabled": False, "value": 0.0},
            "Vmax": {"enabled": True,  "value": 2.5},
        },
        "Units": {"System": "SI", "PressureUnit": "m",
                  "VelocityUnit": "m/s"},
    }


def test_apply_ns():
    cfg = build_algorithm_config(NS=100, MaxGen=500)
    problem = _fake_problem()
    _apply_sensitivity_param(cfg, problem, "NS", 200)
    assert cfg.NS == 200


def test_apply_max_gen():
    cfg = build_algorithm_config(NS=100, MaxGen=500)
    _apply_sensitivity_param(cfg, _fake_problem(), "MaxGen", 1000)
    assert cfg.MaxGen == 1000


def test_apply_constraint_policy():
    cfg = build_algorithm_config(NS=10, MaxGen=5)
    _apply_sensitivity_param(cfg, _fake_problem(),
                       "constraint_policy", "fixed_penalty_prefers")
    assert cfg.constraint_policy == "fixed_penalty_prefers"


def test_apply_pmin_enables_and_sets():
    cfg = build_algorithm_config(NS=10, MaxGen=5)
    problem = _fake_problem()
    problem["Constraints"]["Pmin"]["enabled"] = False
    _apply_sensitivity_param(cfg, problem, "Pmin", 25.0)
    assert problem["Constraints"]["Pmin"]["enabled"] is True
    assert problem["Constraints"]["Pmin"]["value"] == pytest.approx(25.0)


def test_apply_vmax_enables_and_sets():
    cfg = build_algorithm_config(NS=10, MaxGen=5)
    problem = _fake_problem()
    problem["Constraints"]["Vmax"]["enabled"] = False
    _apply_sensitivity_param(cfg, problem, "Vmax", 3.0)
    assert problem["Constraints"]["Vmax"]["enabled"] is True
    assert problem["Constraints"]["Vmax"]["value"] == pytest.approx(3.0)


def test_apply_pmin_in_us_units_converts_to_si():
    """Pmin entered as psi in a US network must convert to SI."""
    cfg = build_algorithm_config(NS=10, MaxGen=5)
    problem = _fake_problem()
    problem["Units"] = {
        "System": "US", "PressureUnit": "psi", "VelocityUnit": "ft/s",
    }
    _apply_sensitivity_param(cfg, problem, "Pmin", 30.0)  # 30 psi
    # 30 psi ≈ 21.09 m
    assert problem["Constraints"]["Pmin"]["value"] == pytest.approx(
        21.09, abs=0.1,
    )


def test_apply_ga_param():
    cfg = build_algorithm_config(NS=10, MaxGen=5)
    _apply_sensitivity_param(cfg, _fake_problem(), "GA.ga_pc", 0.65)
    assert cfg.GA.Pc == 0.65


def test_apply_legacy_ga_name():
    """Old snapshot with GA_Pc still works."""
    cfg = build_algorithm_config(NS=10, MaxGen=5)
    _apply_sensitivity_param(cfg, _fake_problem(), "GA_Pc", 0.65)
    assert cfg.GA.Pc == 0.65


def test_apply_plugin_param_uses_extra():
    """Plugin params land in cfg.extra."""
    from wds_optimizer.algorithms.registry import register
    fake_meta = {
        "short_id": "SweepTestAlgo",
        "display_name": "Sweep Test",
        "category": "Custom",
        "paper_faithful": False,
        "description": "x",
        "references": ["-"],
        "params": [
            {"key": "my_lr", "name": "lr", "display_name": "LR",
             "type": "float", "default": 0.1,
             "min": 0.0, "max": 1.0, "step": 0.01,
             "maps_to": None, "description": "x"},
        ],
        "constraint_policy": "deb_prefers",
        "api_version": 1,
    }
    register("SweepTestAlgo", run=lambda *a, **k: None, meta=fake_meta)
    try:
        cfg = build_algorithm_config(NS=10, MaxGen=5)
        _apply_sensitivity_param(cfg, _fake_problem(),
                           "SweepTestAlgo.my_lr", 0.25)
        assert cfg.extra["SweepTestAlgo"]["my_lr"] == 0.25
    finally:
        ALGORITHM_REGISTRY.pop("SweepTestAlgo", None)


# ---------------------------------------------------------------------
#  Runner end-to-end (synthetic problem, no hydraulics)
# ---------------------------------------------------------------------

def _synthetic_problem():
    return {
        "wn": None, "_solver": False,
        "NP": 3, "VariablePipes": np.array([0, 1, 2]),
        "FixedPipes": np.array([], dtype=int),
        "D": np.array([0.1, 0.2, 0.3]),
        "InitialD": np.array([0.2, 0.2, 0.2]),
        "Cost": np.array([1.0, 10.0, 100.0]),
        "L": np.array([100.0, 100.0, 100.0]),
        "Constraints": {
            "Pmin": {"enabled": True, "value": 30.0},
            "Pmax": {"enabled": False, "value": 100.0},
            "Vmin": {"enabled": False, "value": 0.0},
            "Vmax": {"enabled": False, "value": 2.5},
        },
        "pipe_names": ["P1", "P2", "P3"],
        "junction_names": ["J1", "J2", "J3"],
        "Units": {"System": "SI", "PressureUnit": "m",
                  "VelocityUnit": "m/s", "LengthUnit": "m",
                  "DiameterUnit": "mm"},
    }


def _synthetic_cache():
    """Feasible iff design indices sum to >= 2."""
    import itertools
    cache = {}
    unit = [1.0, 10.0, 100.0]
    for d in itertools.product(range(3), repeat=3):
        c = sum(100.0 * unit[i] for i in d)
        feas = sum(d) >= 2
        cache[d] = (c, 0.0 if feas else 1.0, feas)
    return cache


def test_runner_sweep_ga_pc():
    problem = _synthetic_problem()
    cfg = build_algorithm_config(NS=10, MaxGen=20)
    cfg.Seed = 42
    cfg.Cache = _synthetic_cache()

    result = run_single_sweep(
        problem, cfg,
        parameter_name="GA.ga_pc",
        values=[0.5, 0.8],
        algorithms=["GA"], n_runs=1, experiment_seed=42,
    )
    assert isinstance(result, SingleSweepResult)
    assert len(result.points) == 2
    for p in result.points:
        assert p.status == "Success"
        assert p.best_score == pytest.approx(2100.0)


def test_runner_sweep_pmin():
    problem = _synthetic_problem()
    cfg = build_algorithm_config(NS=10, MaxGen=20)
    cfg.Seed = 42
    cfg.Cache = _synthetic_cache()

    # Base problem had Pmin=30; sweeping to 50 should not change our
    # synthetic cache outcome (cache doesn't use Pmin), but the
    # constraint's value must change for each iteration.
    result = run_single_sweep(
        problem, cfg,
        parameter_name="Pmin",
        values=[30.0, 50.0],
        algorithms=["GA"], n_runs=1, experiment_seed=42,
    )
    assert len(result.points) == 2
    # Original problem's Constraints must be intact (not mutated)
    assert problem["Constraints"]["Pmin"]["value"] == 30.0


def test_runner_sweep_n_runs():
    problem = _synthetic_problem()
    cfg = build_algorithm_config(NS=10, MaxGen=10)
    cfg.Seed = 42
    cfg.Cache = _synthetic_cache()

    result = run_single_sweep(
        problem, cfg,
        parameter_name="n_runs",
        values=[1, 3],
        algorithms=["GA"], n_runs=1,  # ignored for n_runs sweep
        experiment_seed=42,
    )
    assert len(result.points) == 2
    # First point ran 1 run, second ran 3.
    assert len(result.points[0].experiment.runs["GA"]) == 1
    assert len(result.points[1].experiment.runs["GA"]) == 3


def test_runner_sweep_seed():
    problem = _synthetic_problem()
    cfg = build_algorithm_config(NS=10, MaxGen=10)
    cfg.Cache = _synthetic_cache()

    result = run_single_sweep(
        problem, cfg,
        parameter_name="seed",
        values=[1, 2],
        algorithms=["GA"], n_runs=1, experiment_seed=42,
    )
    assert len(result.points) == 2
    assert result.points[0].experiment.experiment_seed == 1
    assert result.points[1].experiment.experiment_seed == 2

def test_runner_sweep_constraint_policy_strings():
    """Sweep constraint_policy with string values; end-to-end.

    Regression for two bugs:
      1. Users typing quoted values ("'deb_prefers'") failed with
         "Unknown constraint policy".
      2. Sweep results rendering crashed on f"{value:g}" for strings.

    This test only covers (1); (2) is covered by the GUI display code.
    """
    problem = _synthetic_problem()
    cfg = build_algorithm_config(NS=10, MaxGen=5)
    cfg.Seed = 42
    cfg.Cache = _synthetic_cache()

    result = run_single_sweep(
        problem, cfg,
        parameter_name="constraint_policy",
        values=["deb_prefers", "fixed_penalty_prefers"],
        algorithms=["GA"], n_runs=1, experiment_seed=42,
    )
    assert len(result.points) == 2
    for p in result.points:
        assert p.status in ("Success", "AllInfeasible"), (
            f"point {p.value!r}: status={p.status}, error={p.error!r}"
        )
    # The base problem's Constraints must not be affected.
    assert problem["Constraints"]["Pmin"]["value"] == 30.0

def test_no_direct_g_format_on_sweep_value_in_source():
    """Guard: no f"{{...value:g}}" left in streamlit_app.py.

    Regression: three separate places used :g on sweep values, and
    each broke when the swept parameter was a string (constraint_policy,
    strategy).  This source-level test catches the pattern before it
    reaches the user.
    """
    from pathlib import Path
    path = (Path(__file__).resolve().parent.parent
            / "src" / "wds_optimizer" / "app" / "streamlit_app.py")
    src = path.read_text(encoding="utf-8")

    # Patterns that must NOT appear (they break on strings):
    forbidden = [
        "prog['ParameterValue']:g",
        'prog["ParameterValue"]:g',
        "p.value:g",
    ]
    for pat in forbidden:
        assert pat not in src, (
            f"Found forbidden format pattern {pat!r} in "
            f"streamlit_app.py — use _format_sweep_value() instead"
        )