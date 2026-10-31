"""Feasibility check identical to checkConstraints.m."""

import numpy as np

# Same value as the MATLAB code.
INVALID_VIOLATION = 1e8


def check_constraints(Pj, Vpipes, Pmin, Vmax, Converged=True):
    """Return (violation, feasible)."""
    if (not Converged) or (Pj is None) or (Vpipes is None):
        return INVALID_VIOLATION, False

    if not (np.all(np.isfinite(Pj)) and np.all(np.isfinite(Vpipes))):
        return INVALID_VIOLATION, False

    p_viol = np.maximum(0.0, (Pmin - np.asarray(Pj)) / Pmin)
    if Vmax is not None and Vmax > 0:
        v_viol = np.maximum(0.0, (np.asarray(Vpipes) - Vmax) / Vmax)
    else:
        v_viol = np.zeros_like(np.asarray(Vpipes))

    viol = float(p_viol.mean() + v_viol.mean())
    return viol, viol == 0.0