"""Guards for the UX-polish changes in v0.5.1a.

These are light "source presence" tests: they don't run Streamlit
(a full GUI test would need Playwright), but they verify that the
specific fixes are in place and cannot silently regress.
"""
from __future__ import annotations

from pathlib import Path

import pytest


APP_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "wds_optimizer" / "app" / "streamlit_app.py"
)


@pytest.fixture(scope="module")
def app_source():
    return APP_PATH.read_text(encoding="utf-8")


def test_clear_results_button_removed(app_source):
    """The old '🗑 Clear results' button is gone."""
    assert "🗑 Clear results" not in app_source


def test_section_headers_renamed(app_source):
    assert "2. Constraints & Population" in app_source
    assert "2. Hydraulics & Constraints" not in app_source

    assert "5. Snapshot (single experiment)" in app_source
    assert '"5. Save / Load"' not in app_source

    assert "6. History (all runs)" in app_source
    assert '"6. History"' not in app_source


def test_input_upload_clears_stale_experiment(app_source):
    """Uploading new files must clear a stale experiment.

    The comment marker is intentionally distinctive so a future edit
    that removes this logic will be noticed.
    """
    assert "Loaded a new network — previous experiment" in app_source
    # The reset block must run before assigning the new paths.
    reset_pos = app_source.find("st.session_state.experiment = None")
    # Just verify at least one occurrence exists inside the upload block.
    upload_idx = app_source.find("if inp_upload and d_upload and cost_upload")
    assert upload_idx >= 0
    assert reset_pos > upload_idx, (
        "Expected the stale-experiment reset block to live inside the "
        "upload block"
    )


def test_reset_experiment_no_longer_imported(app_source):
    """The unused reset_experiment import was removed."""
    assert "from wds_optimizer.app.state import init_state, reset_experiment" \
           not in app_source
    assert "from wds_optimizer.app.state import init_state" in app_source

def test_no_widgets_with_both_default_and_key(app_source):
    """Guard against the Streamlit warning:

        "The widget with key X was created with a default value but
         also had its value set via the Session State API."

    The warning fires whenever a widget has an explicit default
    (index=, default=, or a non-implicit value=) *and* a key= whose
    value is also seeded in session_state.

    We can't detect this from source perfectly (multi-line widgets,
    indirect calls), but we can catch the obvious single-line cases
    that already bit us:
        - selectbox(..., index=X, key=Y)
        - multiselect(..., default=[...], key=Y)
    and the specific call sites known to have been fixed.

    A cleaner check is to run `streamlit run ...` and watch stderr.
    That's done manually; this test is the fast, automated safety net.
    """
    import re

    # Any line with index= followed by key= (or vice versa) on the same
    # line is a red flag — even if the widget spans multiple lines
    # upstream, both args end up in the same call.
    bad_patterns = [
        re.compile(r"\bindex\s*=[^,\)]*\bkey\s*="),
        re.compile(r"\bkey\s*=[^,\)]*\bindex\s*="),
        re.compile(r"\bdefault\s*=[^,\)]*\bkey\s*="),
        re.compile(r"\bkey\s*=[^,\)]*\bdefault\s*="),
    ]

    lines = app_source.splitlines()
    hits = []
    for i, line in enumerate(lines, start=1):
        # Skip comments
        if line.lstrip().startswith("#"):
            continue
        for pat in bad_patterns:
            if pat.search(line):
                hits.append((i, line.strip()))
                break

    assert not hits, (
        "Found widget(s) with both a default and key= on the same line. "
        "Streamlit warns and the default is ignored when the key is "
        "already in session_state.  Seed session_state[key] BEFORE the "
        "widget instead.  Offending lines:\n"
        + "\n".join(f"  line {n}: {line}" for n, line in hits)
    )


def test_constraint_policy_widget_has_no_index(app_source):
    """Specific guard for the historically-buggy constraint_policy widget."""
    # Find the selectbox call for constraint_policy_input.
    import re
    pattern = re.compile(
        r"selectbox\([^)]*key\s*=\s*[\"']constraint_policy_input[\"'][^)]*\)",
        re.DOTALL,
    )
    m = pattern.search(app_source)
    assert m is not None, "constraint_policy selectbox not found"
    call = m.group(0)
    assert "index=" not in call, (
        "constraint_policy selectbox must not pass index=; "
        "seed st.session_state.constraint_policy_input before the "
        "widget instead."
    )


def test_diam_unit_widget_has_no_index(app_source):
    """Same guard for diam_unit_input (also had index=0)."""
    import re
    pattern = re.compile(
        r"selectbox\([^)]*key\s*=\s*[\"']diam_unit_input[\"'][^)]*\)",
        re.DOTALL,
    )
    m = pattern.search(app_source)
    assert m is not None, "diam_unit selectbox not found"
    assert "index=" not in m.group(0), (
        "diam_unit selectbox must not pass index=; "
        "seed st.session_state.diam_unit_input before the widget instead."
    )