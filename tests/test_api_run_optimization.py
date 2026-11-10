"""Tests for OptimizationRequest, OptimizationResult, run_optimization.

Commit 2 of the CLI/API feature.  Covers:

- Request validation (10 error paths).
- End-to-end run on a small synthetic network (real hydraulics, but
  fast: 3 pipes, NS=10, MaxGen=5).
- Every export method (JSON, CSV, Excel, report ZIP).
- progress_callback and cancel_callback wiring.

Runtime: ~3-5 s (dominated by the report-ZIP test, which generates
matplotlib PNGs).
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from wds_optimizer.api import (
    OptimizationRequest,
    OptimizationResult,
    run_optimization,
)


# ---------------------------------------------------------------------------
# Fixture: same synthetic 3-pipe network used in commit-1 tests
# ---------------------------------------------------------------------------

NETWORK_INP = """\
[TITLE]
api run test network

[JUNCTIONS]
J1  0  5
J2  0  5
J3  0  5

[RESERVOIRS]
R1  50

[PIPES]
P1  R1  J1  1000  200  100  0  Open
P2  J1  J2   800  200  100  0  Open
P3  J2  J3  1200  200  100  0  Open

[OPTIONS]
 Units      LPS
 Headloss   H-W

[END]
"""

D_TXT    = "150\n200\n250\n"
COST_TXT = "100\n150\n200\n"


@pytest.fixture
def files(tmp_path):
    inp = tmp_path / "net.inp";   inp.write_text(NETWORK_INP)
    d = tmp_path / "D.txt";       d.write_text(D_TXT)
    c = tmp_path / "Cost.txt";    c.write_text(COST_TXT)
    return {"inp": str(inp), "d": str(d), "cost": str(c)}


def _basic_request(files, **overrides) -> OptimizationRequest:
    kw = dict(
        inp_path=files["inp"], d_path=files["d"], cost_path=files["cost"],
        diam_unit="mm",
        algorithms=["GA"], n_runs=1, seed=42,
        ns=10, max_gen=5,
        constraints={
            "Pmin": {"enabled": True,  "value": 30.0},
            "Pmax": {"enabled": False, "value": 100.0},
            "Vmin": {"enabled": False, "value": 0.0},
            "Vmax": {"enabled": True,  "value": 2.5},
        },
    )
    kw.update(overrides)
    return OptimizationRequest(**kw)


# ---------------------------------------------------------------------------
# Request: defaults and coercion
# ---------------------------------------------------------------------------

def test_default_constraints_filled(files):
    req = OptimizationRequest(
        inp_path=files["inp"], d_path=files["d"], cost_path=files["cost"],
    )
    assert req.constraints["Pmin"]["enabled"] is True
    assert req.constraints["Vmax"]["enabled"] is True
    assert req.constraints["Pmax"]["enabled"] is False
    assert req.constraints["Vmin"]["enabled"] is False


def test_paths_coerced_to_str(files):
    req = _basic_request(files, inp_path=Path(files["inp"]))
    assert isinstance(req.inp_path, str)


# ---------------------------------------------------------------------------
# Request: validation errors
# ---------------------------------------------------------------------------

def test_missing_input_file_raises(files):
    req = _basic_request(files, inp_path="/nonexistent/inp.inp")
    with pytest.raises(ValueError, match="input file not found"):
        req.validate()


def test_bad_diam_unit_raises(files):
    req = _basic_request(files, diam_unit="ft")
    with pytest.raises(ValueError, match="diam_unit"):
        req.validate()


def test_empty_algorithms_raises(files):
    req = _basic_request(files, algorithms=[])
    with pytest.raises(ValueError, match="at least one algorithm"):
        req.validate()


def test_unknown_algorithm_raises(files):
    req = _basic_request(files, algorithms=["XYZ"])
    with pytest.raises(ValueError, match="unknown algorithm"):
        req.validate()


def test_n_runs_below_one_raises(files):
    req = _basic_request(files, n_runs=0)
    with pytest.raises(ValueError, match="n_runs"):
        req.validate()


def test_seed_negative_raises(files):
    req = _basic_request(files, seed=-1)
    with pytest.raises(ValueError, match="seed"):
        req.validate()


def test_ns_too_small_raises(files):
    req = _basic_request(files, ns=2)
    with pytest.raises(ValueError, match="population"):
        req.validate()


def test_max_gen_too_small_raises(files):
    req = _basic_request(files, max_gen=2)
    with pytest.raises(ValueError, match="max_gen"):
        req.validate()


def test_no_constraints_raises(files):
    req = _basic_request(files, constraints={
        "Pmin": {"enabled": False, "value": 30.0},
        "Pmax": {"enabled": False, "value": 100.0},
        "Vmin": {"enabled": False, "value": 0.0},
        "Vmax": {"enabled": False, "value": 2.5},
    })
    with pytest.raises(ValueError, match="at least one constraint"):
        req.validate()


def test_pmin_ge_pmax_raises(files):
    req = _basic_request(files, constraints={
        "Pmin": {"enabled": True, "value": 50.0},
        "Pmax": {"enabled": True, "value": 50.0},
        "Vmin": {"enabled": False, "value": 0.0},
        "Vmax": {"enabled": True, "value": 2.5},
    })
    with pytest.raises(ValueError, match="smaller than"):
        req.validate()


# ---------------------------------------------------------------------------
# run_optimization: end-to-end
# ---------------------------------------------------------------------------

def test_run_optimization_completes(files):
    result = run_optimization(_basic_request(files, algorithms=["GA"]))
    assert isinstance(result, OptimizationResult)
    assert result.experiment is not None
    assert result.best_score < float("inf")
    assert len(result.node_pressures_m) == 3
    assert len(result.pipe_velocities_ms) == 3
    assert result.request.algorithms == ("GA",)


def test_run_optimization_with_two_algorithms(files):
    result = run_optimization(
        _basic_request(files, algorithms=["GA", "DE"], n_runs=1)
    )
    assert "GA" in result.experiment.runs
    assert "DE" in result.experiment.runs


def test_best_run_has_valid_position(files):
    result = run_optimization(_basic_request(files))
    assert result.best_position_m.shape == (3,)
    # every diameter must be one of the option values
    for d in result.best_position_m:
        assert any(np.isclose(d, x) for x in [0.15, 0.2, 0.25])


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def test_to_json_is_loadable_and_complete(files):
    result = run_optimization(_basic_request(files))
    data = json.loads(result.to_json().decode("utf-8"))

    assert "experiment" in data
    assert "problem" in data
    assert "best" in data
    assert "hydraulics" in data
    assert "inputs" in data              # include_inputs=True default
    assert data["problem"]["use_pmin"] is True
    assert data["problem"]["use_vmax"] is True
    assert "Pmax" in data["problem"]
    assert "Vmin" in data["problem"]


def test_to_json_without_inputs(files):
    result = run_optimization(_basic_request(files))
    data = json.loads(result.to_json(include_inputs=False).decode("utf-8"))
    assert "inputs" not in data


def test_to_csv_has_header_and_at_least_one_row(files):
    result = run_optimization(_basic_request(files))
    text = result.to_csv().decode("utf-8-sig")
    lines = [l for l in text.strip().split("\n") if l]
    assert len(lines) >= 2
    assert "Algorithm" in lines[0]
    assert "GA" in lines[1]


def test_to_excel_loadable(files):
    import pandas as pd
    result = run_optimization(_basic_request(files))
    xlsx_bytes = result.to_excel()
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    assert "Summary" in xls.sheet_names
    assert "GA_runs" in xls.sheet_names


def test_to_report_zip_has_expected_structure(files):
    result = run_optimization(_basic_request(files))
    zip_bytes = result.to_report_zip()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "REPORT.md" in names
        assert "README.txt" in names
        assert "experiment.json" in names
        assert any(n.startswith("figures/") and n.endswith(".png")
                   for n in names)
        assert any(n.startswith("data/") for n in names)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def test_progress_callback_receives_generation_events(files):
    events = []
    # max_gen=5 is the validation floor (matches the GUI's min_value=5);
    # overriding it to 4 would trip ``OptimizationRequest.validate``.
    req = _basic_request(files, algorithms=["GA"], max_gen=5)
    run_optimization(req, progress_callback=events.append)
    assert len(events) >= 1
    # Each event has the fields the GUI's live panel uses
    e0 = events[0]
    for key in ("Algorithm", "Generation", "BestCost", "Feasible"):
        assert key in e0


def test_cancel_mid_run_returns_partial(files):
    """Cancel after a few progress events → Partial run kept."""
    n_seen = [0]

    def progress(_info):
        n_seen[0] += 1

    def cancel():
        return n_seen[0] >= 2

    req = _basic_request(files, algorithms=["GA"], max_gen=50)
    result = run_optimization(
        req, progress_callback=progress, cancel_callback=cancel,
    )
    # Some partial or full result should exist; algorithm must not
    # have completed all 50 generations
    assert result.experiment.cancelled is True
    assert result.best_score < float("inf")