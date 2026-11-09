"""Backward compatibility and round-trip tests for snapshots.

The user base for this tool may have .json snapshots written before the
Pmax/Vmin commit landed.  Such files lack ``Pmax``, ``Vmin`` and the
four ``use_*`` flags entirely.  Loading them must:

1. Not raise.
2. Apply the pre-Pmax/Vmin defaults (Pmin on, Vmax on, others off).
3. Preserve the rest of the state (runs, convergence, cost).

A second class of tests verifies the *forward* direction: a new
snapshot built from a modern ExperimentResult, serialized, and loaded
back gives the same ExperimentResult, with all eight constraint fields
intact.

Design
------
No Streamlit, no WNTR.  Only build_snapshot / load_snapshot /
restore_experiment are exercised, using small dicts built inline.
"""
from __future__ import annotations

import io
import json
from datetime import datetime

import numpy as np
import pytest

from wds_optimizer.experiments.runner import (
    ExperimentResult, RunRecord,
)
from wds_optimizer.persistence.snapshot import (
    build_snapshot, restore_experiment,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _minimal_experiment(*, with_constraint_fields: bool = True) -> ExperimentResult:
    """ExperimentResult with one successful run."""
    kwargs = dict(
        experiment_id="test_20260101_000000",
        timestamp=datetime(2026, 1, 1),
        number_of_runs=1,
        algorithms=["GA"],
        experiment_seed=42,
        NS=10, MaxGen=10,
        Pmin=30.0, Vmax=2.5,
        NP=3,
        variable_pipes=np.array([0, 1, 2]),
        fixed_pipes=np.array([], dtype=int),
    )
    if with_constraint_fields:
        kwargs.update(
            Pmax=100.0, Vmin=0.0,
            use_pmin=True, use_pmax=False,
            use_vmin=False, use_vmax=True,
        )
    exp = ExperimentResult(**kwargs)
    exp.runs["GA"] = [RunRecord(
        algorithm="GA", run_index=1, seed=1,
        score=100.0, position=np.array([0.2, 0.2, 0.2]),
        feasible=True, status="Success",
        execution_time=0.1, convergence=np.array([100.0, 100.0]),
        nfe=100, n_sim=100, nfe_to_best=50,
    )]
    return exp


def _problem_snapshot(*, with_pmax_vmin: bool = True) -> dict:
    snap = {
        "pipe_names": ["P1", "P2", "P3"],
        "junction_names": ["J1", "J2", "J3"],
        "L": np.array([100.0, 100.0, 100.0]),
        "initial_D": np.array([0.2, 0.2, 0.2]),
        "D_m": np.array([0.1, 0.2, 0.3]),
        "Din": np.array([4.0, 8.0, 12.0]),
        "Cost": np.array([1.0, 2.0, 3.0]),
        "VariablePipes": np.array([0, 1, 2]),
        "FixedPipes": np.array([], dtype=int),
        "Pmin": 30.0,
        "Vmax": 2.5,
        "Units": {"System": "SI", "PressureUnit": "m",
                  "VelocityUnit": "m/s", "LengthUnit": "m",
                  "DiameterUnit": "mm"},
    }
    if with_pmax_vmin:
        snap.update(
            Pmax=100.0, Vmin=0.0,
            use_pmin=True, use_pmax=False,
            use_vmin=False, use_vmax=True,
        )
    return snap


def _config():
    return {
        "ns": 10, "max_gen": 10, "n_runs": 1, "exp_seed": 42,
        "ga_pc": 0.8, "ga_pm": 0.03, "ga_sel": "tournament",
        "pso_w": 0.7, "pso_c1": 1.5, "pso_c2": 1.5,
        "de_f": 0.5, "de_cr": 0.3, "de_strategy": "rand/1/bin",
        "sade_fmin": 0.1, "sade_fmax": 0.9,
        "sade_crmin": 0.1, "sade_crmax": 0.9,
        "sade_conv": True, "sade_tol": 1e-6, "sade_stag": 300,
        "rs_enabled": False, "rs_patience": 30, "rs_max": 3,
        "rs_keep_best": False, "rs_mode": "perturb", "rs_perturb": 0.3,
    }

def _old_shape_snapshot_dict() -> dict:
    """A dict as written by the *previous* (pre-Pmax/Vmin) app version.

    This bypasses ``build_snapshot`` deliberately: the current
    ``build_snapshot`` fills in the new keys via ``.get(..., default)``
    (correct behaviour for *new* saves), so passing old-shape input
    through it produces a new-shape output — which does not exercise
    the backward-compat path in ``restore_experiment``.

    To test what actually happens when a user loads a file downloaded
    from the previous app version, we must construct the on-disk shape
    by hand: no ``Pmax``, no ``Vmin``, no ``use_*`` keys anywhere.

    The shape below matches what the pre-Pmax/Vmin ``build_snapshot``
    wrote; a snapshot written by that version and stored under
    ``experiment.json`` looks exactly like this.
    """
    return {
        "version": "0.3.0",
        "timestamp": "2025-01-01T00:00:00",
        "environment": {
            "diam_unit": "mm",
            "pmin_si": 30.0,
            "vmax_si": 2.5,
            "pmin": 30.0,
            "vmax": 2.5,
            # No pmax_si, vmin_si, use_pmin/pmax/vmin/vmax here
        },
        "config": _config(),
        "problem": {
            "pipe_names": ["P1", "P2", "P3"],
            "junction_names": ["J1", "J2", "J3"],
            "L": [100.0, 100.0, 100.0],
            "initial_D": [0.2, 0.2, 0.2],
            "D_m": [0.1, 0.2, 0.3],
            "Din": [4.0, 8.0, 12.0],
            "Cost": [1.0, 2.0, 3.0],
            "VariablePipes": [0, 1, 2],
            "FixedPipes": [],
            "Pmin": 30.0,
            "Vmax": 2.5,
            "Units": {
                "System": "SI", "PressureUnit": "m",
                "VelocityUnit": "m/s", "LengthUnit": "m",
                "DiameterUnit": "mm",
            },
            # No Pmax, Vmin, use_pmin/pmax/vmin/vmax here
        },
        "experiment": {
            "experiment_id": "test_20260101_000000",
            "number_of_runs": 1,
            "algorithms": ["GA"],
            "experiment_seed": 42,
            "NS": 10,
            "MaxGen": 10,
            "cancelled": False,
            "restart": {"enabled": False},
            "runs": {
                "GA": [{
                    "algorithm": "GA",
                    "run_index": 1,
                    "seed": 1,
                    "score": 100.0,
                    "position": [0.2, 0.2, 0.2],
                    "feasible": True,
                    "execution_time": 0.1,
                    "convergence": [100.0, 100.0],
                    "status": "Success",
                    "error_message": "",
                    "restart_gens": [],
                    "nfe": 100,
                    "n_sim": 100,
                    "nfe_to_best": 50,
                    "extra": {},
                    "epoch_bests": [],
                }],
            },
        },
        "best": {
            "algorithm": "GA",
            "run_index": 1,
            "position": [0.2, 0.2, 0.2],
        },
        "hydraulics": {
            "node_pressures": [40.0, 35.0, 45.0],
            "pipe_velocities": [1.0, 1.5, 2.0],
            "converged": True,
        },
    }


def _build_snapshot(exp, snap, **extra):
    return build_snapshot(
        problem_snapshot=snap,
        experiment=exp,
        best_algo="GA", best_run_index=1,
        best_position_m=np.array([0.2, 0.2, 0.2]),
        node_pressures=np.array([40.0, 35.0, 45.0]),
        pipe_velocities=np.array([1.0, 1.5, 2.0]),
        hydraulics_converged=True,
        config=_config(),
        **extra,
    )


def _roundtrip_dict(snapshot: dict) -> dict:
    """Serialize and deserialize the snapshot, returning the parsed dict.

    Uses ``build_snapshot`` output directly so we test the exact
    structure that ends up in experiment.json.
    """
    buf = io.StringIO()
    json.dump(snapshot, buf)
    buf.seek(0)
    return json.load(buf)


# ---------------------------------------------------------------------------
# Backward compat: old snapshots load with defaults
# ---------------------------------------------------------------------------

def test_old_snapshot_loads_with_legacy_defaults():
    """A pre-Pmax/Vmin snapshot dict restores with legacy defaults.

    We construct the on-disk shape by hand (via
    ``_old_shape_snapshot_dict``) rather than calling
    ``build_snapshot`` on old-shape input, because the current
    ``build_snapshot`` fills in the new keys via ``.get(..., default)``
    — which is correct for *new* saves but would defeat the purpose of
    this test.
    """
    old = _old_shape_snapshot_dict()

    # Prove the test really is old-shape — otherwise we are not testing
    # the backward-compat path at all.
    assert "use_pmin" not in old["problem"]
    assert "use_pmax" not in old["problem"]
    assert "Pmax" not in old["problem"]
    assert "Vmin" not in old["problem"]
    assert "use_pmin" not in old["environment"]
    assert "pmax_si" not in old["environment"]
    assert "vmin_si" not in old["environment"]

    restored = restore_experiment(old)
    # Pre-Pmax/Vmin behaviour: Pmin on, Vmax on, others off
    assert restored.use_pmin is True
    assert restored.use_vmax is True
    assert restored.use_pmax is False
    assert restored.use_vmin is False
    assert restored.Pmin == pytest.approx(30.0)
    assert restored.Vmax == pytest.approx(2.5)
    assert restored.Pmax == pytest.approx(100.0)
    assert restored.Vmin == pytest.approx(0.0)


def test_old_snapshot_preserves_run_records():
    """Backward compat must not corrupt existing run data."""
    old = _old_shape_snapshot_dict()

    restored = restore_experiment(old)
    assert len(restored.runs["GA"]) == 1
    r = restored.runs["GA"][0]
    assert r.status == "Success"
    assert r.score == pytest.approx(100.0)
    assert r.feasible is True


# ---------------------------------------------------------------------------
# Round-trip: new snapshots preserve all constraint fields
# ---------------------------------------------------------------------------

def test_new_snapshot_roundtrip_preserves_all_constraints():
    """A modern snapshot must round-trip all 8 constraint fields exactly."""
    exp = _minimal_experiment(with_constraint_fields=True)
    snap = _problem_snapshot(with_pmax_vmin=True)
    data = _roundtrip_dict(_build_snapshot(exp, snap))

    # Spot-check the dict we just saved
    assert data["problem"]["Pmin"] == pytest.approx(30.0)
    assert data["problem"]["Pmax"] == pytest.approx(100.0)
    assert data["problem"]["Vmin"] == pytest.approx(0.0)
    assert data["problem"]["Vmax"] == pytest.approx(2.5)
    assert data["problem"]["use_pmin"] is True
    assert data["problem"]["use_pmax"] is False
    assert data["problem"]["use_vmin"] is False
    assert data["problem"]["use_vmax"] is True

    # Restore and verify the same values came back
    restored = restore_experiment(data)
    assert restored.Pmin == pytest.approx(30.0)
    assert restored.Pmax == pytest.approx(100.0)
    assert restored.Vmin == pytest.approx(0.0)
    assert restored.Vmax == pytest.approx(2.5)
    assert restored.use_pmin is True
    assert restored.use_pmax is False
    assert restored.use_vmin is False
    assert restored.use_vmax is True


def test_new_snapshot_with_different_flags_roundtrips():
    """All four enabled → all four preserved across build/load."""
    exp = _minimal_experiment(with_constraint_fields=True)
    exp.use_pmin = True
    exp.use_pmax = True
    exp.use_vmin = True
    exp.use_vmax = True
    exp.Pmin = 25.0
    exp.Pmax = 80.0
    exp.Vmin = 0.2
    exp.Vmax = 3.0
    snap = _problem_snapshot(with_pmax_vmin=True)
    snap.update(Pmin=25.0, Pmax=80.0, Vmin=0.2, Vmax=3.0,
                use_pmin=True, use_pmax=True,
                use_vmin=True, use_vmax=True)
    data = _roundtrip_dict(_build_snapshot(exp, snap))

    restored = restore_experiment(data)
    assert restored.Pmin == pytest.approx(25.0)
    assert restored.Pmax == pytest.approx(80.0)
    assert restored.Vmin == pytest.approx(0.2)
    assert restored.Vmax == pytest.approx(3.0)
    assert restored.use_pmin is True
    assert restored.use_pmax is True
    assert restored.use_vmin is True
    assert restored.use_vmax is True


# ---------------------------------------------------------------------------
# JSON serialization safety
# ---------------------------------------------------------------------------

def test_snapshot_is_json_serializable():
    """build_snapshot output must serialize without special encoders."""
    exp = _minimal_experiment()
    snap = _problem_snapshot()
    data = _build_snapshot(exp, snap)
    # Should not raise
    text = json.dumps(data)
    assert "Pmax" in text and "Vmin" in text


def test_numpy_arrays_become_lists_in_snapshot():
    """All ndarrays are converted to lists (JSON-friendly)."""
    exp = _minimal_experiment()
    snap = _problem_snapshot()
    data = _build_snapshot(exp, snap)
    assert isinstance(data["problem"]["L"], list)
    assert isinstance(data["problem"]["D_m"], list)
    assert isinstance(data["hydraulics"]["node_pressures"], list)