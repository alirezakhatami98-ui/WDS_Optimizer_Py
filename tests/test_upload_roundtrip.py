"""Regression tests for upload handling (``_read_uploaded``).

Covers the bug reported by the user:

    Upload Hanoi's D.txt  → run OK
    Upload Duhala's D.txt → run OK
    Upload Hanoi's D.txt  → run produces absurd costs and is
                            infeasible, with no error

Root cause: ``_read_uploaded`` used a fixed destination path
``<tmp>/<name>`` under ``@st.cache_data``.  The third call's
``(bytes, name)`` tuple matched the first call's cache key
(A → B → A), so Streamlit returned the *path* without re-writing,
but that path's file had been overwritten by network B's bytes.
The third run therefore loaded B's diameters against A's .inp.

Fix under test: content-addressed filenames (``<stem>_<sha16><suffix>``)
so that different contents never collide on disk.

Run with:
    pytest tests/test_upload_roundtrip.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st


# ---------------------------------------------------------------------------
# Import the function under test.
#
# ``streamlit_app.py`` calls ``st.set_page_config`` at import time, which
# raises outside a Streamlit run.  Patch it *only during import*; once the
# module is loaded, its other top-level statements have already run.
# ---------------------------------------------------------------------------
with patch("streamlit.set_page_config"):
    from wds_optimizer.app.streamlit_app import _read_uploaded


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_tmpdir(tmp_path, monkeypatch):
    """Redirect ``tempfile.gettempdir()`` to a per-test directory.

    ``_read_uploaded`` writes under ``<gettempdir()>/wds_optimizer_uploads``.
    Isolating it keeps the test deterministic, fast, and side-effect free.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    yield tmp_path


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    """Wipe ``@st.cache_data`` between tests.

    Without this, a cache entry from one test could mask a regression in
    the next (or, worse, be considered a pass when the fresh code path
    was never executed).
    """
    st.cache_data.clear()
    yield
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_same_content_same_path():
    """Same bytes + same name → same path (correct cache hit)."""
    path1 = _read_uploaded(b"identical bytes", "D.txt")
    path2 = _read_uploaded(b"identical bytes", "D.txt")
    assert path1 == path2
    assert Path(path1).read_bytes() == b"identical bytes"


def test_different_content_different_path():
    """Different bytes + same name → different paths (no overwrite)."""
    path_a = _read_uploaded(b"network A diameters", "D.txt")
    path_b = _read_uploaded(b"network B diameters", "D.txt")
    assert path_a != path_b
    assert Path(path_a).read_bytes() == b"network A diameters"
    assert Path(path_b).read_bytes() == b"network B diameters"


def test_filename_preserves_extension_and_stem():
    """The destination still ends in the original suffix."""
    p = _read_uploaded(b"payload", "weird.name.with.dots.txt")
    assert p.endswith(".txt")
    assert "weird.name.with.dots_" in Path(p).name


# ---------------------------------------------------------------------------
# The regression test — reproduces the exact A → B → A scenario
# ---------------------------------------------------------------------------

def test_a_b_a_content_roundtrip():
    """THE REGRESSION: A → B → A must yield A's *content* on call 3.

    In the pre-fix code the third call was a cache hit for A (A → B → A
    share the same key on calls 1 and 3) and returned the path without
    re-writing it.  But that path's file had been overwritten by B's
    bytes on call 2.  Result: the third A run read B's diameters.

    The three asserts below each pin down one property of the fix:
      1. Same content → same path      (caching is preserved)
      2. Different content → different path  (no collision)
      3. The bytes at *each* path match its own input (content integrity)
    """
    a_bytes = b"network A: 254,254,406,406,508"
    b_bytes = b"network B: 100,150,200"

    path_a1 = _read_uploaded(a_bytes, "D.txt")
    path_b  = _read_uploaded(b_bytes, "D.txt")
    path_a2 = _read_uploaded(a_bytes, "D.txt")   # cache hit in old code

    # 1. Caching still works
    assert path_a1 == path_a2
    # 2. No disk collision
    assert path_a1 != path_b
    # 3. THE critical assertion: content matches the *input*
    assert Path(path_a2).read_bytes() == a_bytes
    assert Path(path_b).read_bytes() == b_bytes


# ---------------------------------------------------------------------------
# Integration-light: the real multi-file sequence
# ---------------------------------------------------------------------------

def test_multi_file_upload_sequence_a_b_a():
    """Upload three same-named files for A, then B, then A again.

    This is the user's scenario end-to-end at the file level.  Before the
    fix, the *second* A cycle read B's D.txt and Cost.txt from the shared
    paths while the .inp was A's (because .inp filenames for Hanoi and
    Duhala differed on disk, so no collision was noticed).
    """
    files_a = {
        "net.inp":   b"[JUNCTIONS]\nJ1 0 0\n",
        "D.txt":     b"254\n254\n406\n",
        "Cost.txt":  b"100\n100\n120\n",
    }
    files_b = {
        "net.inp":   b"[JUNCTIONS]\nJ2 0 0\nJ3 0 0\n",
        "D.txt":     b"100\n150\n",
        "Cost.txt":  b"50\n50\n",
    }

    written = {}
    for label, files in (("A", files_a), ("B", files_b), ("A2", files_a)):
        written[label] = {
            name: _read_uploaded(data, name) for name, data in files.items()
        }

    # Content integrity at every destination
    for label, files in (("A", files_a), ("B", files_b), ("A2", files_a)):
        for name, expected in files.items():
            got = Path(written[label][name]).read_bytes()
            assert got == expected, (
                f"content mismatch for {label}/{name}: "
                f"expected {expected!r}, got {got!r}"
            )

    # Caching: A and A2 must resolve to the same paths
    for name in files_a:
        assert written["A"][name] == written["A2"][name]

    # Isolation: A and B must never share a path
    for name in files_a:
        assert written["A"][name] != written["B"][name]


# ---------------------------------------------------------------------------
# Regression guard against the *old* implementation
# ---------------------------------------------------------------------------

def test_old_fixed_path_implementation_would_fail_this_regression(tmp_path):
    """Meta-test: prove the old code would fail ``test_a_b_a_content_roundtrip``.

    This test does not exercise ``_read_uploaded`` at all.  It re-implements
    the *pre-fix* logic inline and asserts that it exhibits the bug — so
    that if the fix is ever reverted (e.g. someone changes
    ``_read_uploaded`` back to a fixed path), ``test_a_b_a_content_roundtrip``
    will fail again rather than silently pass.
    """
    # Pre-fix logic, verbatim
    cache: dict[tuple[bytes, str], str] = {}
    fixed_dir = tmp_path / "uploads"
    fixed_dir.mkdir()

    def old_read_uploaded(data: bytes, name: str) -> str:
        key = (data, name)
        p = fixed_dir / name
        if key not in cache:
            p.write_bytes(data)     # only on cache miss
            cache[key] = str(p)
        return cache[key]

    a = b"A-diameters"
    b = b"B-diameters"
    path_a1 = old_read_uploaded(a, "D.txt")
    path_b  = old_read_uploaded(b, "D.txt")
    path_a2 = old_read_uploaded(a, "D.txt")   # cache hit

    # The bug: same path, wrong content
    assert path_a1 == path_a2
    assert path_a1 == path_b        # all three share one file
    assert Path(path_a2).read_bytes() == b, (
        "sanity: the old implementation should exhibit the bug "
        "(if this fails, the test harness is broken, not the fix)"
    )