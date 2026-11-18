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