"""Normalised violation severity for visualisation.

Both the interactive network map (``app/network_map.py``) and the
static PNG in the reproducibility report (``reporting/figures.py``)
need to convert a physical value (pressure, velocity) into a scalar
in [0, 1] that indicates how far outside a constraint band it lies.

The logic was previously duplicated in each module, with subtly
different NaN handling.  This module is the single source of truth.
"""

from __future__ import annotations

import numpy as np


def two_sided_severity(values, low, high, *, min_scale: float = 1e-9):
    """Per-element severity in [0, 1] relative to a two-sided band.

    Parameters
    ----------
    values : array_like or scalar
        Physical values to score.  May contain ``NaN`` / ``inf``.
    low, high : float or None
        Band boundaries.  Either may be ``None`` to disable that side;
        passing both ``None`` makes the function return zeros for all
        finite inputs (equivalent to "no constraint").
    min_scale : float, default 1e-9
        Lower bound on the normaliser.  Prevents division by zero when
        the violated bound is exactly 0 (e.g. ``Vmin = 0``); a
        negative value against a bound of 0 yields maximum severity 1.

    Returns
    -------
    numpy.ndarray or float
        Same shape as ``values`` for array input; a Python ``float``
        for scalar input.  Finite inputs get 0.0 when inside the band,
        rising to (and capped at) 1.0 as they violate further outside.
        Non-finite inputs (``NaN``, ``±inf``) map to ``NaN`` so the
        caller can render them distinctly (typically gray) rather than
        as either an OK value or a violation — the value is *unknown*,
        not *wrong*.

    Notes
    -----
    For finite inputs against a non-zero bound, this reproduces the
    exact severity the pre-refactor ``network_map._two_sided_severity``
    returned, so archived figures that contain no NaN values remain
    bit-for-bit comparable.
    """
    arr = np.asarray(values, dtype=float)
    scalar_input = (arr.ndim == 0)
    arr = np.atleast_1d(arr)

    out = np.full(arr.shape, np.nan, dtype=float)
    finite = np.isfinite(arr)
    out[finite] = 0.0

    if low is not None:
        below = finite & (arr < float(low))
        if below.any():
            span = max(abs(float(low)), float(min_scale))
            out[below] = np.clip((float(low) - arr[below]) / span, 0.0, 1.0)

    if high is not None:
        above = finite & (arr > float(high))
        if above.any():
            span = max(abs(float(high)), float(min_scale))
            capped = np.clip((arr[above] - float(high)) / span, 0.0, 1.0)
            # An element can only be in one of the two branches because
            # low < high is assumed upstream.  Using maximum here makes
            # the function safe even if a caller passes low >= high.
            out[above] = np.maximum(out[above], capped)

    return float(out[0]) if scalar_input else out