"""Tests for the shared ``two_sided_severity`` helper.

The helper replaces two nearly-identical inline versions that lived in
``app/network_map.py`` and ``reporting/figures.py``.  Those two
disagreed on NaN handling:

- network_map.py returned 1.0 for NaN, so NaN pipes/junctions were
  rendered as severe violations (red).
- figures.py passed NaN through to matplotlib's colormap, whose
  default "bad" colour is transparent, so NaN pipes/junctions were
  effectively invisible.

Both were wrong — NaN means "no data".  The shared implementation
returns NaN for non-finite input, and both callers now render it as
neutral gray.

The last test in this file pins that for finite inputs the new
function is bit-for-bit identical to the old network_map.py version,
preserving archived reports whose data contain no NaN.
"""
from __future__ import annotations

import numpy as np
import pytest

from wds_optimizer.optimization.severity import two_sided_severity


# ---------------------------------------------------------------------------
# Basic severity: inside / below / above
# ---------------------------------------------------------------------------

def test_inside_band_returns_zero():
    assert two_sided_severity([50, 60], 40, 70).tolist() == [0.0, 0.0]


def test_below_low_returns_positive():
    # (40 - 30) / 40 = 0.25
    result = two_sided_severity([30], 40, 70)
    assert result[0] == pytest.approx(0.25)


def test_above_high_returns_positive():
    # (80 - 70) / 70 ≈ 0.1428
    result = two_sided_severity([80], 40, 70)
    assert result[0] == pytest.approx((80 - 70) / 70)


def test_severity_capped_at_one():
    assert two_sided_severity([1000], 40, 70)[0] == 1.0
    assert two_sided_severity([-1000], 40, 70)[0] == 1.0


# ---------------------------------------------------------------------------
# Optional sides
# ---------------------------------------------------------------------------

def test_low_only_ignores_high():
    assert two_sided_severity([100, 50], 30, None).tolist() == [0.0, 0.0]


def test_high_only_ignores_low():
    assert two_sided_severity([10, 50], None, 70).tolist() == [0.0, 0.0]


def test_neither_returns_zeros():
    assert two_sided_severity([1, 2, 3], None, None).tolist() == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Non-finite inputs
# ---------------------------------------------------------------------------

def test_nan_passthrough():
    """NaN and infinities must be preserved so callers render them
    distinctly (gray) rather than as OK or violation."""
    result = two_sided_severity([50, np.nan, 60, np.inf, -np.inf], 40, 70)
    assert result[0] == 0.0
    assert np.isnan(result[1])
    assert result[2] == 0.0
    assert np.isnan(result[3])
    assert np.isnan(result[4])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_low_uses_floor():
    """low=0 must not divide by zero.  Any negative value against a
    bound of 0 gets the maximum severity, since the natural normaliser
    is 0."""
    assert two_sided_severity([-1], 0, 10)[0] == 1.0
    assert two_sided_severity([1], 0, 10)[0] == 0.0


def test_scalar_input():
    result = two_sided_severity(50, 40, 70)
    assert float(result) == 0.0


def test_list_input_returns_ndarray():
    result = two_sided_severity([50, 60], 40, 70)
    assert isinstance(result, np.ndarray)


def test_array_input_preserves_shape():
    arr = np.array([[30, 50], [100, 60]])
    result = two_sided_severity(arr, 40, 70)
    assert result.shape == (2, 2)
    assert result[0, 0] > 0
    assert result[0, 1] == 0
    assert result[1, 0] > 0
    assert result[1, 1] == 0


# ---------------------------------------------------------------------------
# Backward compatibility with the pre-refactor inline version
# ---------------------------------------------------------------------------

def test_matches_previous_scalar_implementation_for_finite_inputs():
    """For finite inputs, the shared function must reproduce the
    previous network_map.py inline version bit-for-bit.  This keeps
    archived reports (whose data contain no NaN) visually identical."""
    def old_scalar(value, low, high):
        if not np.isfinite(value):
            return 1.0          # old behaviour — intentionally differs
        if low is not None and value < low:
            span = max(abs(low), 1e-9)
            return min(1.0, (low - value) / span)
        if high is not None and value > high:
            span = max(abs(high), 1e-9)
            return min(1.0, (value - high) / span)
        return 0.0

    for v in (10, 30, 40, 50, 70, 100):
        old = old_scalar(v, 40, 70)
        new = float(two_sided_severity(v, 40, 70))
        assert abs(old - new) < 1e-12, (
            f"v={v}: old={old!r}, new={new!r}"
        )