"""Tests for constraint-handling policies."""
from __future__ import annotations

import pytest

from wds_optimizer.optimization.policies import (
    CONSTRAINT_POLICIES,
    POLICY_META,
    deb_prefers,
    fixed_penalty_prefers,
    get_policy,
    simple_cost_only_prefers,
)


# ---------------------------------------------------------------------
#  deb_prefers — the policy all four algorithms use
# ---------------------------------------------------------------------

def test_deb_feasible_beats_infeasible():
    assert deb_prefers(cost_a=1000, viol_a=0, cost_b=10, viol_b=0.1)
    assert not deb_prefers(cost_a=10, viol_a=0.1, cost_b=1000, viol_b=0)


def test_deb_two_feasible_lower_cost_wins():
    assert deb_prefers(500, 0, 1000, 0)
    assert not deb_prefers(1000, 0, 500, 0)


def test_deb_two_infeasible_lower_violation_wins():
    assert deb_prefers(100, 0.1, 100, 0.5)
    assert not deb_prefers(100, 0.5, 100, 0.1)


def test_deb_ties_return_false():
    """Ties are not 'strictly better' → False."""
    assert not deb_prefers(500, 0, 500, 0)
    assert not deb_prefers(500, 0.2, 500, 0.2)


# ---------------------------------------------------------------------
#  fixed_penalty_prefers
# ---------------------------------------------------------------------

def test_fixed_penalty_feasible_beats_infeasible():
    assert fixed_penalty_prefers(1_000_000, 0, 1, 0.1)
    assert not fixed_penalty_prefers(1, 0.1, 1_000_000, 0)


def test_fixed_penalty_cheaper_infeasible_wins():
    assert fixed_penalty_prefers(100, 0.5, 200, 0.5)
    assert not fixed_penalty_prefers(200, 0.5, 100, 0.5)


# ---------------------------------------------------------------------
#  simple_cost_only_prefers
# ---------------------------------------------------------------------

def test_simple_cost_only_ignores_violation():
    """Cheaper infeasible beats pricier feasible — by design."""
    assert simple_cost_only_prefers(10, 0.9, 1000, 0)
    assert not simple_cost_only_prefers(1000, 0, 10, 0.9)


# ---------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------

def test_all_policies_in_meta_and_registry():
    assert set(CONSTRAINT_POLICIES) == set(POLICY_META)


def test_get_policy_known():
    assert get_policy("deb_prefers") is deb_prefers


def test_get_policy_unknown_raises():
    with pytest.raises(KeyError, match="Unknown constraint policy"):
        get_policy("not_a_policy")


def test_policy_meta_has_required_keys():
    for name, m in POLICY_META.items():
        for key in ("display_name", "description", "reference", "code"):
            assert key in m, f"{name}: missing {key!r}"
        assert isinstance(m["code"], str) and m["code"], (
            f"{name}: 'code' must be a non-empty string for GUI display"
        )