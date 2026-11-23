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
    # Section numbers shift with each release; check the descriptive
    # part of each header, not the number.
    assert "Constraints & Population" in app_source
    assert "Hydraulics & Constraints" not in app_source

    assert "Sensitivity Analysis" in app_source    # new section
    assert "Snapshot (single experiment)" in app_source
    assert '"5. Save / Load"' not in app_source

    assert "History (all runs)" in app_source
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

def test_sensitivity_button_uses_job_button(app_source):
    """The Sensitivity Analysis button uses _job_button so it becomes
    a Cancel button while a sensitivity job runs."""
    assert '_job_button(\n            "sensitivity"' in app_source or \
           '_job_button("sensitivity"' in app_source


def test_sensitivity_dialog_is_one_shot(app_source):
    """sensitivity_dialog_open must be consumed before the dialog is
    rendered; otherwise it re-opens on any rerun (algorithm change,
    etc.)."""
    # Locate the "if sensitivity_dialog_open" block and verify the
    # first statement inside it sets the flag to False.
    import re
    m = re.search(
        r'if st\.session_state\.get\("sensitivity_dialog_open"\):\s*\n'
        r'\s+st\.session_state\.sensitivity_dialog_open = False\s*\n'
        r'\s+_sensitivity_dialog\(\)',
        app_source,
    )
    assert m is not None, (
        "sensitivity_dialog_open must be set to False BEFORE "
        "_sensitivity_dialog() is called (one-shot trigger)."
    )

def test_sensitivity_parameter_defaults_to_its_own_plan_row(app_source):
    """Each parameter widget must initialize from its own plan row."""
    assert 'st.session_state[par_key] = row["param"]' in app_source
    assert 'st.session_state.get(par_key, catalog_keys[0])' not in app_source


def test_sensitivity_parameter_options_exclude_other_rows(app_source):
    """A parameter selected in one row must not be offered in another."""
    assert "used_by_other_rows = {" in app_source
    assert 'other["id"] != rid' in app_source
    assert "row_catalog_keys = [" in app_source
    assert "options=row_catalog_keys" in app_source


def test_sensitivity_add_uses_only_available_parameters(app_source):
    """Add must use an unused parameter and disable when none remain."""
    assert "selected_params = {" in app_source
    assert "available_params = [" in app_source
    assert "disabled=not available_params" in app_source
    assert "default_param = available_params[0]" in app_source


def test_sensitivity_plan_edits_use_callbacks(app_source):
    """Add, Reset and Delete must use callbacks so only the dialog
    fragment reruns automatically."""
    import re

    dialog_match = re.search(
        r'@st\.dialog\("Sensitivity Analysis".*?'
        r"(?=^def _start_sensitivity_batch)",
        app_source,
        re.DOTALL | re.MULTILINE,
    )
    assert dialog_match is not None
    dialog = dialog_match.group(0)

    # Each plan-edit operation must use a callback.
    assert "on_click=_delete_sensitivity_row" in dialog
    assert "on_click=_add_sensitivity_parameter" in dialog
    assert "on_click=_reset_sensitivity_plan" in dialog

    # Plan edits must not force a full-app rerun or reopen the dialog.
    assert "st.session_state.sensitivity_dialog_open = True" not in dialog

    # The two remaining full-app reruns belong to Run and Cancel.
    assert dialog.count("st.rerun()") == 2

def test_sensitivity_syncs_sidebar_before_building_problem(app_source):
    """Sensitivity must use the current sidebar values, not stale
    canonical session-state values."""
    import re

    match = re.search(
        r"^def _start_sensitivity_batch\(.*?"
        r"(?=^def _finalize_sensitivity)",
        app_source,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None
    source = match.group(0)

    sync_pos = source.find(
        "_sync_sidebar_state(with_n_runs=False)"
    )
    prepare_pos = source.find("_prepare_problem()")

    assert sync_pos >= 0
    assert prepare_pos >= 0
    assert sync_pos < prepare_pos


def test_sensitivity_builds_config_for_selected_algorithms(app_source):
    """The sensitivity dialog algorithm selection must drive the
    base algorithm configuration."""
    assert "def _build_base_cfg(selected_algos=None):" in app_source
    assert "base_cfg = _build_base_cfg(selected_algos=algos)" in app_source
    assert "list(selected_algos)" in app_source

def test_latest_sensitivity_batch_is_selected_after_finalize(app_source):
    """The newly completed sensitivity batch must become the
    selected batch in the Sensitivity tab."""
    import re

    match = re.search(
        r"^def _finalize_sensitivity\(.*?"
        r"(?=^def _create_sensitivity_history_entries)",
        app_source,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None
    source = match.group(0)

    append_pos = source.find("batches.append(record)")
    save_pos = source.find(
        "st.session_state.sensitivity_batches = batches"
    )
    select_pos = source.find(
        "st.session_state.sensitivity_batch_selector = len(batches) - 1"
    )

    assert append_pos >= 0
    assert save_pos > append_pos
    assert select_pos > save_pos

def test_sensitivity_batch_selector_uses_session_state_only(app_source):
    """The batch selector must not combine index with Session State."""
    import re

    match = re.search(
        r"^def _render_sensitivity_tab\(.*?"
        r"(?=^def |\Z)",
        app_source,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None
    source = match.group(0)

    assert 'key="sensitivity_batch_selector"' in source
    assert "index=len(batches) - 1" not in source
    assert "st.session_state.sensitivity_batch_selector = len(batches) - 1" in source

def test_sensitivity_choice_parameters_use_multiselect(app_source):
    import re

    assert re.search(
        r"st\.multiselect\s*\(\s*[\"']Values[\"']",
        app_source,
    )
    assert 'choice_key = f"sen_choices_{rid}"' in app_source


def test_sensitivity_numeric_parameters_keep_text_input(app_source):
    assert 'row["values"] = st.text_input(' in app_source


def test_sensitivity_parameter_change_resets_defaults(app_source):
    assert "def _on_sensitivity_parameter_change(row_id):" in app_source
    assert 'row["param"] = new_param' in app_source
    assert 'row["values"] = ", ".join(map(str, defaults))' in app_source
    assert "on_change=_on_sensitivity_parameter_change" in app_source


def test_sensitivity_add_uses_catalog_defaults(app_source):
    assert 'info = catalog[default_param]' in app_source
    assert 'default_values = info["default_values"]' in app_source
    assert 'default_vals = "0.5, 0.7, 0.9"' not in app_source


def test_sensitivity_default_plan_uses_catalog(app_source):
    import re

    match = re.search(
        r"^def _make_default_plan\(.*?(?=^def |\\Z)",
        app_source,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None

    source = match.group(0)
    assert "build_sensitivity_parameters()" in source
    assert 'info["default_values"]' in source