"""Tests for check_constraints (four-constraint version).

The critical test is ``test_backward_compat_identical_to_old_formula``:
with only Pmin and Vmax enabled (the previous defaults), the computed
``viol`` must be **numerically identical** to the pre-Pmax/Vmin
implementation.  Same operations, same order, same floats — so a run
with the old UI settings and the same seed produces the same cost
sequence as before, and the earlier experiment reports remain
directly comparable.

Edge cases covered
------------------
- Backward compat (Pmin + Vmax on) — exact numerics match the old code.
- Each constraint individually enabled (Pmin, Pmax, Vmin, Vmax).
- All four enabled — violations add linearly across families.
- All disabled — returns (0.0, True); no constraint is trivially met.
- Pmin == Pmax — permitted here (validated upstream in the UI).
- Vmin = 0 with non-negative velocities — no division by zero.
- Non-converged / None / non-finite inputs — return INVALID_VIOLATION.
- List inputs — accepted (NumPy coercion is internal).

Run with:
    pytest tests/test_constraints.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from wds_optimizer.optimization.constraints import (
    INVALID_VIOLATION,
    check_constraints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _constraints(*, pmin=False, pmax=False, vmin=False, vmax=False,
                 pmin_v=30.0, pmax_v=100.0, vmin_v=0.0, vmax_v=2.5):
    """Build a constraints dict; any family not passed ``True`` is off."""
    return {
        "Pmin": {"enabled": bool(pmin), "value": float(pmin_v)},
        "Pmax": {"enabled": bool(pmax), "value": float(pmax_v)},
        "Vmin": {"enabled": bool(vmin), "value": float(vmin_v)},
        "Vmax": {"enabled": bool(vmax), "value": float(vmax_v)},
    }


def _old_violation(Pj, Vpipes, Pmin, Vmax):
    """Pre-Pmax/Vmin reference implementation, inlined for comparison."""
    p_viol = np.maximum(0.0, (Pmin - np.asarray(Pj)) / Pmin)
    v_viol = np.maximum(0.0, (np.asarray(Vpipes) - Vmax) / Vmax)
    return float(p_viol.mean() + v_viol.mean())


# ---------------------------------------------------------------------------
# Backward compatibility (the critical test)
# ---------------------------------------------------------------------------

def test_backward_compat_identical_to_old_formula():
    """Pmin + Vmax on, others off → numerically identical to old code.

    Uses exact ``==`` (not isclose) deliberately: if this ever fails,
    the new check_constraints has diverged from what the project used
    to compute, and old results are no longer comparable.
    """
    Pj = np.array([10.0, 25.0, 35.0, 40.0, 22.5, 28.0])
    Vp = np.array([1.2, 2.5, 3.1, 0.8, 2.9, 2.2])
    Pmin, Vmax = 30.0, 2.5

    viol_new, feas_new = check_constraints(
        Pj, Vp, _constraints(pmin=True, vmax=True,
                             pmin_v=Pmin, vmax_v=Vmax),
    )
    viol_old = _old_violation(Pj, Vp, Pmin, Vmax)

    assert viol_new == viol_old, (
        f"backward compat broken: new={viol_new!r}, old={viol_old!r}"
    )
    assert feas_new == (viol_old == 0.0)


def test_backward_compat_when_feasible():
    Pj = np.array([35.0, 40.0, 50.0])
    Vp = np.array([1.0, 1.5, 2.0])
    viol, feas = check_constraints(
        Pj, Vp, _constraints(pmin=True, vmax=True),
    )
    assert viol == 0.0
    assert feas is True


def test_backward_compat_when_infeasible():
    Pj = np.array([10.0, 40.0, 50.0])
    Vp = np.array([3.0, 1.5, 2.0])
    viol, feas = check_constraints(
        Pj, Vp, _constraints(pmin=True, vmax=True),
    )
    assert viol > 0.0
    assert feas is False
    assert viol == _old_violation(Pj, Vp, 30.0, 2.5)


# ---------------------------------------------------------------------------
# Each constraint individually
# ---------------------------------------------------------------------------

def test_pmin_only():
    c = _constraints(pmin=True, pmin_v=30.0)
    Pj = np.array([10.0, 30.0, 60.0])
    Vp = np.full(3, 100.0)                     # ignored (Vmax off)
    viol, feas = check_constraints(Pj, Vp, c)
    # (30-10)/30 over 3 nodes → (20/30)/3
    assert np.isclose(viol, (20.0 / 30.0) / 3.0)
    assert feas is False


def test_pmax_only():
    c = _constraints(pmax=True, pmax_v=100.0)
    Pj = np.array([50.0, 100.0, 150.0])
    Vp = np.full(3, 0.1)
    viol, feas = check_constraints(Pj, Vp, c)
    # (150-100)/100 over 3 nodes → 0.5/3
    assert np.isclose(viol, 0.5 / 3.0)
    assert feas is False


def test_vmin_only():
    c = _constraints(vmin=True, vmin_v=1.0)
    Pj = np.full(3, 100.0)
    Vp = np.array([0.5, 1.0, 2.0])
    viol, feas = check_constraints(Pj, Vp, c)
    # (1-0.5)/1 over 3 pipes → 0.5/3
    assert np.isclose(viol, 0.5 / 3.0)
    assert feas is False


def test_vmax_only():
    c = _constraints(vmax=True, vmax_v=2.5)
    Pj = np.full(3, 100.0)
    Vp = np.array([1.0, 2.5, 5.0])
    viol, feas = check_constraints(Pj, Vp, c)
    # (5-2.5)/2.5 over 3 pipes → 1.0/3
    assert np.isclose(viol, 1.0 / 3.0)
    assert feas is False


# ---------------------------------------------------------------------------
# Multiple constraints
# ---------------------------------------------------------------------------

def test_pmin_pmax_sum_linearly():
    """Pmin and Pmax contributions add (not average)."""
    c = _constraints(pmin=True, pmin_v=30.0, pmax=True, pmax_v=100.0)
    Pj = np.array([10.0, 200.0, 50.0])
    Vp = np.zeros(3)
    viol, _ = check_constraints(Pj, Vp, c)
    expected = (20.0 / 30.0) / 3.0 + (100.0 / 100.0) / 3.0
    assert np.isclose(viol, expected)


def test_all_four_constraints_enabled():
    c = _constraints(pmin=True, pmin_v=30.0,
                     pmax=True, pmax_v=100.0,
                     vmin=True, vmin_v=1.0,
                     vmax=True, vmax_v=2.5)
    Pj = np.array([10.0, 50.0, 200.0])
    Vp = np.array([0.5, 1.5, 5.0])
    viol, feas = check_constraints(Pj, Vp, c)
    expected = ((20.0 / 30.0) + 1.0 + 0.5 + 1.0) / 3.0
    assert np.isclose(viol, expected)
    assert feas is False


def test_all_disabled():
    """No constraints → trivially feasible, zero violation."""
    c = _constraints()                          # all off
    Pj = np.array([-100.0, 0.0, 1000.0])
    Vp = np.array([-100.0, 0.0, 1000.0])
    viol, feas = check_constraints(Pj, Vp, c)
    assert viol == 0.0
    assert feas is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_pmin_equal_pmax_does_not_crash():
    """check_constraints does not validate; equal values are legal here.

    The UI refuses Pmin >= Pmax when both are enabled; this test just
    confirms the maths does not crash if it is ever called directly.
    """
    c = _constraints(pmin=True, pmin_v=50.0, pmax=True, pmax_v=50.0)
    Pj = np.full(3, 50.0)
    Vp = np.zeros(3)
    viol, feas = check_constraints(Pj, Vp, c)
    assert viol == 0.0
    assert feas is True


def test_vmin_zero_with_positive_velocities():
    """Vmin = 0 must not divide by zero (scale fallback kicks in)."""
    c = _constraints(vmin=True, vmin_v=0.0)
    Pj = np.zeros(3)
    Vp = np.array([0.5, 1.5, 2.5])
    viol, feas = check_constraints(Pj, Vp, c)
    assert np.isfinite(viol)
    assert viol == 0.0
    assert feas is True


def test_vmin_zero_with_zero_velocity():
    """V=0 satisfies Vmin=0 exactly."""
    c = _constraints(vmin=True, vmin_v=0.0)
    viol, _ = check_constraints(np.zeros(3), np.zeros(3), c)
    assert viol == 0.0


def test_non_converged_returns_invalid():
    c = _constraints(pmin=True, vmax=True)
    viol, feas = check_constraints(
        np.array([50.0, 50.0]), np.array([1.0, 1.0]), c, Converged=False,
    )
    assert viol == INVALID_VIOLATION
    assert feas is False


def test_none_inputs_return_invalid():
    c = _constraints(pmin=True, vmax=True)
    viol, feas = check_constraints(None, np.array([1.0]), c)
    assert viol == INVALID_VIOLATION
    assert feas is False


def test_non_finite_inputs_return_invalid():
    c = _constraints(pmin=True, vmax=True)
    viol, feas = check_constraints(
        np.array([50.0, np.nan]), np.array([1.0, 1.0]), c,
    )
    assert viol == INVALID_VIOLATION
    assert feas is False


def test_list_inputs_work():
    """check_constraints should accept lists, not only ndarrays."""
    c = _constraints(pmin=True, vmax=True)
    viol, feas = check_constraints([35.0, 40.0], [1.0, 1.5], c)
    assert viol == 0.0
    assert feas is True