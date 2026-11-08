"""Feasibility check with four optional constraints.

Extends the original ``checkConstraints.m`` (which supported only
Pmin and Vmax) to four independent, individually togglable constraints:
Pmin, Pmax (pressure, SI metres) and Vmin, Vmax (velocity, SI m/s).

Backward compatibility: with only Pmin and Vmax enabled, the computed
violation is *numerically identical* to the previous implementation
(``p_viol.mean() + v_viol.mean()`` where each term is
``max(0, limit - value)/limit`` or ``max(0, value - limit)/limit``).
The user can therefore compare new results with old snapshots without
worrying about a change of scale.
"""

import numpy as np

# Same value as the MATLAB code.
INVALID_VIOLATION = 1e8

# Lower bound on the normalisation scale. Prevents division by zero for
# constraints such as Vmin=0 ("no reverse flow"). Such constraints are
# typically enabled together with a sensible bound, but a defensive
# floor costs nothing.
_MIN_SCALE = 1e-9


def _violation(values, limit, direction):
    """Return the per-element normalised violation of one constraint.

    ``direction = "min"`` → ``values < limit`` is a violation
    ``direction = "max"`` → ``values > limit`` is a violation

    The normaliser is ``abs(limit)`` when ``limit`` is non-zero; this
    reproduces the previous ``(Pmin - Pj) / Pmin`` and
    ``(Vpipes - Vmax) / Vmax`` exactly. When ``limit`` is 0 or near
    zero, the scale falls back to the max absolute value of the data,
    so normalised violations stay O(1) regardless of the network's
    physical scale.
    """
    arr = np.asarray(values, dtype=float)
    if direction == "min":
        raw = np.maximum(0.0, float(limit) - arr)
    else:
        raw = np.maximum(0.0, arr - float(limit))

    scale = abs(float(limit))
    if scale < _MIN_SCALE:
        scale = max(
            float(np.max(np.abs(arr))) if arr.size else 1.0,
            _MIN_SCALE,
        )
    return raw / scale


def check_constraints(Pj, Vpipes, constraints, Converged=True):
    """Return ``(violation, feasible)``.

    ``constraints`` is a dict::

        {
            "Pmin": {"enabled": bool, "value": float},   # SI, m
            "Pmax": {"enabled": bool, "value": float},   # SI, m
            "Vmin": {"enabled": bool, "value": float},   # SI, m/s
            "Vmax": {"enabled": bool, "value": float},   # SI, m/s
        }

    ``violation`` is the sum, over enabled families, of the *mean*
    normalised violation of that family — the same shape the previous
    implementation used to combine Pmin and Vmax. Summing across
    families (rather than averaging) means a design that violates two
    constraints is worse than one that violates only one.

    A design is feasible iff ``violation == 0``. If no constraint is
    enabled (validated upstream to be an error), this function returns
    ``(0.0, True)`` — every design is trivially feasible, which is the
    mathematically consistent outcome of "no constraint".
    """
    if (not Converged) or (Pj is None) or (Vpipes is None):
        return INVALID_VIOLATION, False

    if not (np.all(np.isfinite(Pj)) and np.all(np.isfinite(Vpipes))):
        return INVALID_VIOLATION, False

    viol = 0.0
    if constraints["Pmin"]["enabled"]:
        viol += float(_violation(Pj, constraints["Pmin"]["value"],
                                 "min").mean())
    if constraints["Pmax"]["enabled"]:
        viol += float(_violation(Pj, constraints["Pmax"]["value"],
                                 "max").mean())
    if constraints["Vmin"]["enabled"]:
        viol += float(_violation(Vpipes, constraints["Vmin"]["value"],
                                 "min").mean())
    if constraints["Vmax"]["enabled"]:
        viol += float(_violation(Vpipes, constraints["Vmax"]["value"],
                                 "max").mean())

    return viol, viol == 0.0