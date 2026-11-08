"""End-to-end regression tests for cross-network ``_prepare_problem``.

Companion to ``test_upload_roundtrip.py``.  That test verified the upload
helper in isolation; this one runs the *full* data path that the user hit:

    read_uploaded(bytes, "D.txt")     # content-addressed path
        → st.session_state.d_path
        → _prepare_problem()
        → load_optimization_data(d_path, cost_path, ...)
        → problem["D_m"], problem["Din"], problem["Cost"]

and confirms that the third cycle in A → B → A produces the same problem
dict as the first.

Also unit-tests the ``copy.deepcopy`` fix on ``_load_network`` (a separate
leak: ``run_hydraulic_simulation`` mutates a WNTR model in place, so the
cached model must not be handed out directly).

Network fixtures are minimal valid EPANET .inp files (3 pipes for A,
2 pipes for B) generated in-test; no external files are read.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import streamlit as st


# ---------------------------------------------------------------------------
# streamlit_app.py calls st.set_page_config() at import time; patch it.
# Once imported, the module's other top-level statements have already run.
# ---------------------------------------------------------------------------
with patch("streamlit.set_page_config"):
    from wds_optimizer.app import streamlit_app

from wds_optimizer.app.uploads import read_uploaded


# ---------------------------------------------------------------------------
# Minimal, valid EPANET networks (Units LPS → diameters in mm on disk)
# ---------------------------------------------------------------------------

NETWORK_A_INP = """\
[TITLE]
test network A

[JUNCTIONS]
J1  0  5
J2  0  5
J3  0  5

[RESERVOIRS]
R1  50

[PIPES]
P1  R1  J1  1000  200  100  0  Open
P2  J1  J2   800  200  100  0  Open
P3  J2  J3  1200  200  100  0  Open

[OPTIONS]
 Units      LPS
 Headloss   H-W

[END]
"""

NETWORK_B_INP = """\
[TITLE]
test network B

[JUNCTIONS]
K1  0  5
K2  0  5

[RESERVOIRS]
S1  40

[PIPES]
Q1  S1  K1   500  150  100  0  Open
Q2  K1  K2   700  150  100  0  Open

[OPTIONS]
 Units      LPS
 Headloss   H-W

[END]
"""

A_D    = "200\n250\n300\n"
A_COST = "100\n150\n200\n"
B_D    = "150\n250\n"
B_COST = "80\n180\n"


# ---------------------------------------------------------------------------
# Fake st.session_state with both dict-style and attribute-style access.
# Real st.session_state behaves this way; a plain dict lacks the attributes.
# ---------------------------------------------------------------------------

class FakeSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    yield tmp_path


@pytest.fixture(autouse=True)
def clear_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def fake_session_state(monkeypatch):
    """Replace st.session_state with an in-memory dict-like object.

    Both st.session_state.d_path (attribute) and st.session_state.get(...)
    (dict) must work — _prepare_problem uses both styles.

    Seeds the four constraint values and their enabled flags with the
    app defaults (Pmin=30 m on, Pmax=100 m off, Vmin=0 m/s off,
    Vmax=2.5 m/s on).  _prepare_problem reads ``st.session_state.pmin``,
    ``.pmax``, ``.vmin``, ``.vmax`` and ``.use_pmin`` / ``.use_pmax`` /
    ``.use_vmin`` / ``.use_vmax``; adding new constraints later requires
    updating this fixture too.
    """
    fake = FakeSessionState(
        inp_path=None, d_path=None, cost_path=None,
        diam_unit="mm",
        # values (display units, matching the unit system in the test .inp)
        pmin=30.0, pmax=100.0, vmin=0.0, vmax=2.5,
        # enabled flags (defaults: Pmin on, Vmax on, Pmax & Vmin off)
        use_pmin=True, use_pmax=False, use_vmin=False, use_vmax=True,
        fixed_pipes="",
    )
    monkeypatch.setattr(st, "session_state", fake)
    return fake


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upload_cycle(fake_ss, inp_str, d_str, cost_str):
    """Simulate one "upload three files" cycle; call _prepare_problem."""
    fake_ss.inp_path  = read_uploaded(inp_str.encode(),  "net.inp")
    fake_ss.d_path    = read_uploaded(d_str.encode(),    "D.txt")
    fake_ss.cost_path = read_uploaded(cost_str.encode(), "Cost.txt")
    return streamlit_app._prepare_problem()


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

def test_a_b_a_problem_dict_identical(fake_session_state):
    """THE REGRESSION: _prepare_problem() must return identical problem
    dicts on A cycle 1 and A cycle 3."""
    p_a1, _, _ = _upload_cycle(fake_session_state, NETWORK_A_INP, A_D, A_COST)
    p_b,  _, _ = _upload_cycle(fake_session_state, NETWORK_B_INP, B_D, B_COST)
    p_a2, _, _ = _upload_cycle(fake_session_state, NETWORK_A_INP, A_D, A_COST)

    # Networks A and B genuinely differ
    assert len(p_a1["D"]) == 3
    assert len(p_b["D"])  == 2

    # THE critical assertion
    for key in ("D", "Din", "Cost", "InitialD", "L"):
        assert np.array_equal(p_a1[key], p_a2[key]), (
            f"field {key!r} differs between A cycle 1 and A cycle 3: "
            f"first={p_a1[key]!r}, second={p_a2[key]!r}"
        )
    assert p_a1["pipe_names"]     == p_a2["pipe_names"]
    assert p_a1["junction_names"] == p_a2["junction_names"]


def test_a_b_a_cycle3_reads_a_diameters(fake_session_state):
    """A cycle 3 must read A.inp's and A_D.txt's values, not B's leftovers."""
    p_a1, _, _ = _upload_cycle(fake_session_state, NETWORK_A_INP, A_D, A_COST)
    _upload_cycle(fake_session_state, NETWORK_B_INP, B_D, B_COST)
    p_a2, _, _ = _upload_cycle(fake_session_state, NETWORK_A_INP, A_D, A_COST)

    # Diameter options from A_D.txt with diam_unit="mm"
    expected_D = np.array([0.2, 0.25, 0.3])
    assert np.allclose(p_a1["D"], expected_D)
    assert np.allclose(p_a2["D"], expected_D), (
        f"A cycle 3 diameter options should be {expected_D}, "
        f"got {p_a2['D']}. If this fails, D.txt was overwritten by B."
    )

    # Initial diameters from A.inp (200 mm on every pipe)
    expected_initial = np.array([0.2, 0.2, 0.2])
    assert np.allclose(p_a1["InitialD"], expected_initial)
    assert np.allclose(p_a2["InitialD"], expected_initial), (
        f"A cycle 3 InitialD should be 0.2 m for all 3 pipes, "
        f"got {p_a2['InitialD']}. Two possibilities: "
        f"(a) upload path returned stale bytes, or "
        f"(b) _load_network returned a cached WNTR model already mutated."
    )


def test_a_b_a_upload_paths_content_addressed(fake_session_state):
    """Paths are stable across same-content uploads but distinct across
    different content (the content-addressing invariant)."""
    _upload_cycle(fake_session_state, NETWORK_A_INP, A_D, A_COST)
    a1 = (fake_session_state.inp_path, fake_session_state.d_path,
          fake_session_state.cost_path)
    _upload_cycle(fake_session_state, NETWORK_B_INP, B_D, B_COST)
    b = (fake_session_state.inp_path, fake_session_state.d_path,
         fake_session_state.cost_path)
    _upload_cycle(fake_session_state, NETWORK_A_INP, A_D, A_COST)
    a2 = (fake_session_state.inp_path, fake_session_state.d_path,
          fake_session_state.cost_path)

    assert a1 == a2                       # same bytes → same path
    for pa, pb in zip(a1, b):             # different bytes → different path
        assert pa != pb, f"path collision: A={pa}, B={pb}"


def test_abab_alternating_cycles(fake_session_state):
    """Stress test: A → B → A → B. All A cycles identical, all B identical."""
    a_problems, b_problems = [], []
    for _ in range(2):
        a_problems.append(_upload_cycle(
            fake_session_state, NETWORK_A_INP, A_D, A_COST)[0])
        b_problems.append(_upload_cycle(
            fake_session_state, NETWORK_B_INP, B_D, B_COST)[0])

    for key in ("D", "Din", "Cost", "InitialD"):
        assert np.array_equal(a_problems[0][key], a_problems[1][key]), \
            f"A-cycle field {key!r} drifted between cycles"
        assert np.array_equal(b_problems[0][key], b_problems[1][key]), \
            f"B-cycle field {key!r} drifted between cycles"


# ---------------------------------------------------------------------------
# Unit test for the deepcopy fix on _load_network
# ---------------------------------------------------------------------------

def test_load_network_returns_isolated_copy(isolated_tmpdir):
    """_load_network must deepcopy: mutating its return value must not
    affect a subsequent call (which would poison the cached model)."""
    p = isolated_tmpdir / "net_a.inp"
    p.write_text(NETWORK_A_INP, encoding="utf-8")

    wn1 = streamlit_app._load_network(str(p))
    wn1.get_link("P1").diameter = 9.99

    wn2 = streamlit_app._load_network(str(p))
    assert wn2.get_link("P1").diameter != 9.99, (
        "deepcopy fix regressed: mutations to one returned model leak "
        "into the cached one."
    )
    assert abs(wn2.get_link("P1").diameter - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# Meta: prove the pre-fix logic would fail test_a_b_a_problem_dict_identical
# ---------------------------------------------------------------------------

def test_old_fixed_path_logic_would_fail_regression(tmp_path):
    """Re-implement pre-fix upload logic inline; show A→B→A yields B bytes.

    This test does not exercise any production code. Its purpose is to
    ensure that the meta-regression it describes (cache-hit without
    rewrite on a shared path) truly reproduces the original bug — so a
    future revert of the fix would be caught by
    test_a_b_a_problem_dict_identical above, not silently pass.
    """
    cache: dict[tuple[bytes, str], str] = {}
    fixed_dir = tmp_path / "uploads"
    fixed_dir.mkdir()

    def old_read(data: bytes, name: str) -> str:
        key = (data, name)
        p = fixed_dir / name
        if key not in cache:
            p.write_bytes(data)
            cache[key] = str(p)
        return cache[key]

    a_path1 = old_read(A_D.encode(), "D.txt")
    _       = old_read(B_D.encode(), "D.txt")     # overwrites fixed path
    a_path2 = old_read(A_D.encode(), "D.txt")     # cache hit; no rewrite

    assert a_path1 == a_path2, "cache hit must return same path"
    assert Path(a_path2).read_text() == B_D, (
        "sanity: the pre-fix logic must exhibit the bug (wrong content at "
        "the cached path). If this fails, the test harness is broken, not "
        "the fix."
    )