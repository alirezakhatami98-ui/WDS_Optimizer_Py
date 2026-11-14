"""Constraint-handling policies — the "how do I compare two designs?" layer.

A constraint *policy* is a pure function

    prefers(cost_a, viol_a, cost_b, viol_b) -> bool

that returns True iff design A is strictly preferable to design B.
It is separate from ``check_constraints`` (which computes the
violation value) and from ``evaluate_population`` (which runs the
hydraulics and applies the policy indirectly through the algorithm's
selection step).

The library below makes constraint handling visible and swappable:

- Every policy has a ``POLICY_META`` entry with description, reference,
  and code.
- The GUI (v0.3.1) renders a dropdown of ``CONSTRAINT_POLICIES`` and,
  for each, shows the META — so a user selecting "Deb's rules" *sees*
  the exact comparison logic without writing any code.
- Plugin authors (v0.5) can either reference a built-in policy by
  name in their ``ALGORITHM_META`` or define a custom one inline in
  their plugin file and reference it by name.

This module contains no algorithm-specific logic.  It is a library
of "primitives" that any algorithm may import.
"""
from __future__ import annotations

from typing import Callable


# =====================================================================
#  Built-in policies
# =====================================================================

def deb_prefers(cost_a, viol_a, cost_b, viol_b) -> bool:
    """Deb's feasibility rules (Deb 2000).

    Returns True iff design A is *strictly* preferable to design B:

      * A feasible and B infeasible  → True
      * A infeasible and B feasible  → False
      * Both feasible                → True iff cost_a < cost_b
      * Both infeasible              → True iff viol_a < viol_b

    Ties (same cost, both feasible; same violation, both infeasible)
    return False — i.e., "A is not strictly better than B".  Callers
    that want to accept ties must test ``not deb_prefers(B, A)``.

    This is what all four algorithms in this project use.  It works
    with ``check_constraints``'s normalized violation value.
    """
    feasible_a = viol_a == 0
    feasible_b = viol_b == 0
    if feasible_a and feasible_b:
        return cost_a < cost_b
    if feasible_a != feasible_b:
        return feasible_a
    return viol_a < viol_b


def fixed_penalty_prefers(cost_a, viol_a, cost_b, viol_b) -> bool:
    """Fixed-penalty comparison.

    A design's effective cost is ``cost`` if feasible, else
    ``cost + penalty``.  Lower effective cost wins.

    Requires a penalty value; this function uses a module-level
    default of ``1e9`` (far above any WDS network cost) so any
    feasible design beats any infeasible one, while among infeasible
    designs the cheaper one is preferred.  Algorithms that want a
    different penalty (e.g. Suribabu 2010's max-max-cost rule) can
    define a custom policy inline.
    """
    _PENALTY = 1e9
    eff_a = cost_a if viol_a == 0 else cost_a + _PENALTY
    eff_b = cost_b if viol_b == 0 else cost_b + _PENALTY
    return eff_a < eff_b


def simple_cost_only_prefers(cost_a, viol_a, cost_b, viol_b) -> bool:
    """Ignore feasibility; compare raw cost only.

    Useful for *diagnostic* runs that want to see how a population
    drifts if constraints are ignored.  Not suitable for production
    design — the returned "best" design may be infeasible.
    """
    return cost_a < cost_b


# =====================================================================
#  Policy metadata (rendered by the GUI, inspectable by users)
# =====================================================================

POLICY_META = {
    "deb_prefers": {
        "display_name": "Deb's feasibility rules",
        "description": (
            "Feasible designs always beat infeasible ones.  Among two "
            "feasible designs, the cheaper wins.  Among two infeasible "
            "designs, the one with smaller total violation wins.  "
            "Ties go to the incumbent (no switch)."
        ),
        "reference": (
            "Deb, K. (2000). \"An efficient constraint handling method "
            "for genetic algorithms.\" Computer Methods in Applied "
            "Mechanics and Engineering, 186, 311-338."
        ),
        "code": (
            "def deb_prefers(cost_a, viol_a, cost_b, viol_b):\n"
            "    feasible_a = viol_a == 0\n"
            "    feasible_b = viol_b == 0\n"
            "    if feasible_a and feasible_b:\n"
            "        return cost_a < cost_b\n"
            "    if feasible_a != feasible_b:\n"
            "        return feasible_a\n"
            "    return viol_a < viol_b\n"
        ),
        "notes": (
            "Depends on a *normalised* violation value (check_constraints "
            "returns such).  If you change check_constraints to return "
            "raw units, this policy still works but comparing "
            "violations across designs becomes less meaningful."
        ),
    },
    "fixed_penalty_prefers": {
        "display_name": "Fixed penalty (feasible + cheap wins)",
        "description": (
            "Effective cost = actual cost, plus a large penalty "
            "(1e9) if the design is infeasible.  Lower effective cost "
            "wins.  Equivalent to 'feasible beats infeasible; among "
            "infeasible, cheaper is better'."
        ),
        "reference": (
            "Standard penalty-method formulation (e.g. Coello 2002)."
        ),
        "code": (
            "PENALTY = 1e9\n"
            "def fixed_penalty_prefers(cost_a, viol_a, cost_b, viol_b):\n"
            "    eff_a = cost_a if viol_a == 0 else cost_a + PENALTY\n"
            "    eff_b = cost_b if viol_b == 0 else cost_b + PENALTY\n"
            "    return eff_a < eff_b\n"
        ),
        "notes": (
            "Simple and deterministic, but the constant penalty can "
            "cause a population to drift toward cheaper (less feasible) "
            "designs during early generations.  Use seeded initial "
            "population if applying to a large benchmark."
        ),
    },
    "simple_cost_only_prefers": {
        "display_name": "Cost only (ignore constraints)",
        "description": (
            "Compare raw costs; ignore feasibility entirely.  "
            "Diagnostic use only — the returned 'best' design may be "
            "infeasible."
        ),
        "reference": "—",
        "code": (
            "def simple_cost_only_prefers(cost_a, viol_a, cost_b, viol_b):\n"
            "    return cost_a < cost_b\n"
        ),
        "notes": "Not for production use.",
    },
}


CONSTRAINT_POLICIES: dict[str, Callable] = {
    "deb_prefers": deb_prefers,
    "fixed_penalty_prefers": fixed_penalty_prefers,
    "simple_cost_only_prefers": simple_cost_only_prefers,
}


def get_policy(name: str) -> Callable:
    """Look up a constraint policy by its short name."""
    if name not in CONSTRAINT_POLICIES:
        raise KeyError(
            f"Unknown constraint policy {name!r}.  "
            f"Available: {sorted(CONSTRAINT_POLICIES)}"
        )
    return CONSTRAINT_POLICIES[name]