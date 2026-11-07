"""Content-addressed upload handling for the Streamlit app.

Kept in its own module so tests can import ``read_uploaded`` without
pulling in the full ``streamlit_app`` (which imports wntr, plotly,
pandas, scipy, and every algorithm module).  The reduction in import
time is ~2-3 s per test process.

The cache decorator is Streamlit's, so cache invalidation follows the
same rules as before (function source hash + args).  Moving the
function to a new module changes the function's module-qualified name
and therefore invalidates any old cache entries that still pointed at
shared paths — a desirable side effect of this refactor.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st


@st.cache_data(show_spinner=False)
def read_uploaded(uploaded_bytes: bytes, name: str) -> str:
    """Save uploaded bytes to a content-addressed temp file; return path.

    The filename embeds a SHA-256 hash of the content so:

    * Same bytes → same path (safe under @st.cache_data — a cache hit
      returns a path whose file still holds the right content).
    * Different bytes → different path (uploads of same-named files for
      different networks never overwrite each other).

    Without this, uploading ``D.txt`` for network A, then for network B,
    then for A again hit the cache on the third call (A → B → A) and
    returned a path whose file had been overwritten by B; the third A
    run then read B's diameters/costs against A's .inp, yielding
    infeasible non-converging results with no error.

    See ``tests/test_upload_roundtrip.py`` for the regression test.
    """
    tmpdir = Path(tempfile.gettempdir()) / "wds_optimizer_uploads"
    tmpdir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(uploaded_bytes).hexdigest()[:16]
    stem = Path(name).stem
    suffix = Path(name).suffix
    p = tmpdir / f"{stem}_{digest}{suffix}"
    p.write_bytes(uploaded_bytes)   # only runs on cache miss
    return str(p)