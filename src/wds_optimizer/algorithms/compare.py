"""Deb's feasibility rules – mirror of isBetterSolution.m."""


def is_better_solution(cost_a, viol_a, cost_b, viol_b):
    """Return True if A is strictly better than B."""
    feas_a = (viol_a == 0)
    feas_b = (viol_b == 0)

    if feas_a and feas_b:
        return cost_a < cost_b
    if feas_a != feas_b:
        return feas_a
    return viol_a < viol_b