"""Tests for wds_optimizer.api.build_problem and load_network_isolated.

This is commit 1 of the CLI/API feature.  The refactor moves
``_prepare_problem``'s logic out of streamlit_app.py into api.py, so
that CLI and embedded callers can build the same problem dict without
importing Streamlit.

The tests below pin:

1. ``build_problem`` produces the expected dict for a known input.
2. ``build_problem`` accepts both 1-based string and 0-based int
   ``fixed_pipes`` and produces the same result.
3. ``build_problem`` validates constraints (empty, Pmin>=Pmax,
   Vmin>=Vmax) with messages the user has seen before.
4. ``load_network_isolated`` returns a fresh copy each call.

The GUI's ``_prepare_problem`` behaviour is unchanged — its existing
tests in test_prepare_problem_roundtrip.py continue to pass.

No production behaviour changed by this commit other than the code
location of ``_prepare_problem``'s internals.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from wds_optimizer.api import (
    build_problem,
    load_network_isolated,
)


# ---------------------------------------------------------------------------
# Fixture: a minimal valid EPANET network (3-pipe line) + D/Cost files
# ---------------------------------------------------------------------------

NETWORK_INP = """\
[TITLE]
api test network

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
    inp  = tmp_path / "net.inp";   inp.write_text(NETWORK_INP)
    d    = tmp_path / "D.txt";     d.write_text(D_TXT)
    cost = tmp_path / "Cost.txt";  cost.write_text(COST_TXT)
    return {"inp": str(inp), "d": str(d), "cost": str(cost)}


def _constraints(*, pmin=True, pmax=False, vmin=False, vmax=True,
                 pmin_v=30.0, pmax_v=100.0, vmin_v=0.0, vmax_v=2.5):
    return {
        "Pmin": {"enabled": bool(pmin), "value": float(pmin_v)},
        "Pmax": {"enabled": bool(pmax), "value": float(pmax_v)},
        "Vmin": {"enabled": bool(vmin), "value": float(vmin_v)},
        "Vmax": {"enabled": bool(vmax), "value": float(vmax_v)},
    }


# ---------------------------------------------------------------------------
# build_problem: happy path
# ---------------------------------------------------------------------------

def test_build_problem_returns_expected_dict(files):
    problem, units, wn = build_problem(
        inp_path=files["inp"], d_path=files["d"], cost_path=files["cost"],
        diam_unit="mm", constraints=_constraints(),
    )
    assert problem["NP"] == 3
    assert problem["pipe_names"] == ["P1", "P2", "P3"]
    assert problem["junction_names"] == ["J1", "J2", "J3"]
    assert len(problem["D"]) == 3
    assert np.allclose(problem["D"], np.array([0.15, 0.20, 0.25]))
    assert np.allclose(problem["Cost"], np.array([100.0, 150.0, 200.0]))
    assert np.allclose(problem["L"], np.array([1000, 800, 1200]))
    assert np.allclose(problem["InitialD"], np.array([0.2, 0.2, 0.2]))
    assert problem["FixedPipes"].tolist() == []
    assert problem["VariablePipes"].tolist() == [0, 1, 2]
    assert units["System"] == "SI"
    assert wn is not None


def test_build_problem_with_fixed_pipes_as_string(files):
    problem, _, _ = build_problem(
        inp_path=files["inp"], d_path=files["d"], cost_path=files["cost"],
        diam_unit="mm", constraints=_constraints(),
        fixed_pipes="1, 3",
    )
    assert problem["FixedPipes"].tolist() == [0, 2]
    assert problem["VariablePipes"].tolist() == [1]


def test_build_problem_with_fixed_pipes_as_ints(files):
    problem, _, _ = build_problem(
        inp_path=files["inp"], d_path=files["d"], cost_path=files["cost"],
        diam_unit="mm", constraints=_constraints(),
        fixed_pipes=[0, 2],
    )
    assert problem["FixedPipes"].tolist() == [0, 2]
    assert problem["VariablePipes"].tolist() == [1]


def test_string_and_int_fixed_pipes_equivalent(files):
    p_str, _, _ = build_problem(
        inp_path=files["inp"], d_path=files["d"], cost_path=files["cost"],
        diam_unit="mm", constraints=_constraints(), fixed_pipes="1, 3",
    )
    p_int, _, _ = build_problem(
        inp_path=files["inp"], d_path=files["d"], cost_path=files["cost"],
        diam_unit="mm", constraints=_constraints(), fixed_pipes=[0, 2],
    )
    assert p_str["FixedPipes"].tolist() == p_int["FixedPipes"].tolist()
    assert (p_str["VariablePipes"].tolist()
            == p_int["VariablePipes"].tolist())


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------

def test_no_constraints_enabled_raises(files):
    with pytest.raises(ValueError, match="Enable at least one constraint"):
        build_problem(
            inp_path=files["inp"], d_path=files["d"],
            cost_path=files["cost"], diam_unit="mm",
            constraints=_constraints(
                pmin=False, pmax=False, vmin=False, vmax=False),
        )


def test_pmin_equal_pmax_raises(files):
    with pytest.raises(ValueError, match="must be smaller than"):
        build_problem(
            inp_path=files["inp"], d_path=files["d"],
            cost_path=files["cost"], diam_unit="mm",
            constraints=_constraints(pmin=True, pmax=True,
                                     pmin_v=50.0, pmax_v=50.0),
        )


def test_vmin_equal_vmax_raises(files):
    with pytest.raises(ValueError, match="must be smaller than"):
        build_problem(
            inp_path=files["inp"], d_path=files["d"],
            cost_path=files["cost"], diam_unit="mm",
            constraints=_constraints(vmin=True, vmax=True,
                                     vmin_v=2.0, vmax_v=2.0),
        )


def test_missing_constraint_key_raises(files):
    bad = _constraints()
    del bad["Pmax"]
    with pytest.raises(ValueError, match="missing key"):
        build_problem(
            inp_path=files["inp"], d_path=files["d"],
            cost_path=files["cost"], diam_unit="mm", constraints=bad,
        )


# ---------------------------------------------------------------------------
# Fixed-pipes validation
# ---------------------------------------------------------------------------

def test_fixed_pipe_string_out_of_range_raises(files):
    with pytest.raises(ValueError, match="between 1 and"):
        build_problem(
            inp_path=files["inp"], d_path=files["d"],
            cost_path=files["cost"], diam_unit="mm",
            constraints=_constraints(), fixed_pipes="5",
        )


def test_fixed_pipe_string_duplicate_raises(files):
    with pytest.raises(ValueError, match="must be unique"):
        build_problem(
            inp_path=files["inp"], d_path=files["d"],
            cost_path=files["cost"], diam_unit="mm",
            constraints=_constraints(), fixed_pipes="1, 1",
        )


def test_fixed_pipe_int_out_of_range_raises(files):
    with pytest.raises(ValueError, match="out of range"):
        build_problem(
            inp_path=files["inp"], d_path=files["d"],
            cost_path=files["cost"], diam_unit="mm",
            constraints=_constraints(), fixed_pipes=[10],
        )


# ---------------------------------------------------------------------------
# Isolation guarantee
# ---------------------------------------------------------------------------

def test_load_network_isolated_returns_fresh_copy(tmp_path):
    p = tmp_path / "net.inp"
    p.write_text(NETWORK_INP)

    wn1 = load_network_isolated(str(p))
    wn1.get_link("P1").diameter = 0.999

    wn2 = load_network_isolated(str(p))
    # The mutation on wn1 must not have leaked to wn2.
    assert wn2.get_link("P1").diameter != 0.999
    assert abs(wn2.get_link("P1").diameter - 0.2) < 1e-9