"""Subprocess tests for the wds-optimize CLI.

Every test runs the CLI as a real subprocess (``python -m wds_optimizer.cli``)
so the console-script plumbing, exit codes, and stdout/stderr routing
are all exercised end-to-end.

Design
------
- No mocking.  The tests run the real CLI, which runs the real
  run_optimization, which runs the real algorithms — on a tiny 3-pipe
  synthetic network (NS=10, MaxGen=5) that completes in ~2-3 s per run.
- ``--quiet`` suppresses tqdm/print progress so the captured stdout is
  clean and parseable.
- ``python -m wds_optimizer.cli`` is used instead of the ``wds-optimize``
  console script so the tests pass even when the package has not been
  ``pip install``-ed (e.g. fresh checkout + pytest).

Runtime: ~20-25 s total.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

NETWORK_INP = """\
[TITLE]
cli test network

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

D_TXT    = "150\n200\n250\n"
COST_TXT = "100\n150\n200\n"


@pytest.fixture
def files(tmp_path):
    inp = tmp_path / "net.inp";   inp.write_text(NETWORK_INP)
    d = tmp_path / "D.txt";       d.write_text(D_TXT)
    c = tmp_path / "Cost.txt";    c.write_text(COST_TXT)
    return {"inp": str(inp), "d": str(d), "cost": str(c)}


def _run_cli(args, *, timeout: int = 120):
    """Run ``python -m wds_optimizer.cli args...`` in a subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "wds_optimizer.cli"] + list(args),
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# --help / --version / info
# ---------------------------------------------------------------------------

def test_help_lists_subcommands():
    code, out, _ = _run_cli(["--help"])
    assert code == 0
    for sub in ("run", "validate", "info", "benchmark"):
        assert sub in out


def test_version_prints_something():
    code, out, _ = _run_cli(["--version"])
    assert code == 0
    assert out.strip().startswith("wds-optimize")


def test_info_prints_environment():
    code, out, _ = _run_cli(["info"])
    assert code == 0
    assert "Python:" in out
    assert "wntr:" in out
    assert "wds-optimizer:" in out


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_validate_ok(files):
    code, out, err = _run_cli([
        "validate",
        "--inp", files["inp"], "--d", files["d"], "--cost", files["cost"],
    ])
    assert code == 0, err
    assert "Network:" in out
    assert "Pipes:" in out
    assert "Diameter options: 3" in out
    assert "Cost entries:     3" in out


def test_validate_missing_file_errors():
    code, _, err = _run_cli([
        "validate",
        "--inp", "/does/not/exist.inp",
        "--d", "/does/not/exist.txt",
        "--cost", "/does/not/exist.txt",
    ])
    assert code == 2
    assert "Error" in err


# ---------------------------------------------------------------------------
# run: error paths
# ---------------------------------------------------------------------------

def test_run_without_constraints_errors(files):
    code, _, err = _run_cli([
        "run", "--inp", files["inp"], "--d", files["d"],
        "--cost", files["cost"], "--quiet",
    ])
    assert code == 2
    assert "at least one constraint" in err


def test_run_pmin_ge_pmax_errors(files):
    code, _, err = _run_cli([
        "run", "--inp", files["inp"], "--d", files["d"],
        "--cost", files["cost"],
        "--pmin", "100", "--pmax", "50",
        "--quiet",
    ])
    assert code == 2
    assert "smaller than" in err


def test_run_missing_file_errors(files):
    code, _, err = _run_cli([
        "run", "--inp", "/nope.inp", "--d", files["d"],
        "--cost", files["cost"], "--pmin", "30", "--quiet",
    ])
    assert code == 2
    assert "not found" in err or "Error" in err


# ---------------------------------------------------------------------------
# run: end-to-end happy path
# ---------------------------------------------------------------------------

def test_run_writes_json_and_returns_ok(files, tmp_path):
    out_path = tmp_path / "out.json"
    code, out, err = _run_cli([
        "run",
        "--inp", files["inp"], "--d", files["d"], "--cost", files["cost"],
        "--algo", "GA",
        "--ns", "10", "--max-gen", "5", "--n-runs", "1",
        "--pmin", "30", "--vmax", "2.5",
        "--output", str(out_path),
        "--quiet",
    ])
    # 0 (feasible) or 1 (infeasible) are both acceptable for a tiny run
    assert code in (0, 1), f"exit={code}, stderr={err}"
    assert out_path.exists()

    data = json.loads(out_path.read_text())
    assert "experiment" in data
    assert "problem" in data
    assert "best" in data
    assert "hydraulics" in data


def test_run_writes_csv(files, tmp_path):
    csv_path = tmp_path / "runs.csv"
    code, _, err = _run_cli([
        "run",
        "--inp", files["inp"], "--d", files["d"], "--cost", files["cost"],
        "--ns", "10", "--max-gen", "5",
        "--pmin", "30",
        "--csv", str(csv_path),
        "--quiet",
    ])
    assert code in (0, 1), err
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8-sig")
    assert "Algorithm" in text
    assert "GA" in text


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def test_benchmark_list():
    code, out, _ = _run_cli(["benchmark", "--list"])
    assert code == 0
    assert "demo" in out
    assert "Available benchmarks" in out


def test_benchmark_demo_runs(files, tmp_path):
    out_path = tmp_path / "demo.json"
    code, out, err = _run_cli([
        "benchmark",
        "--network", "demo",
        "--ns", "10", "--max-gen", "5",
        "--output", str(out_path),
        "--quiet",
    ])
    assert code in (0, 1), f"stderr={err}"
    assert out_path.exists()


def test_benchmark_unknown_name_errors():
    code, _, err = _run_cli(["benchmark", "--network", "unknown-net"])
    assert code == 2
    assert "unknown benchmark" in err

# ---------------------------------------------------------------------------
# Diameter-unit sanity check (commit 3.5)
# ---------------------------------------------------------------------------

# Three values, matching the 3-pipe test network.  Using 6 values would
# trip ``load_optimization_data``'s count consistency check before the
# sanity warning is even evaluated.
D_TXT_INCHES_FOR_SI = "8\n10\n12\n"   # max < 50: looks like inches


def test_validate_warns_when_mm_values_look_like_inches(files, tmp_path):
    """SI network + --diam-unit mm + values 8-18 → warning."""
    d_in = tmp_path / "D_small.txt"
    d_in.write_text(D_TXT_INCHES_FOR_SI)
    code, out, _ = _run_cli([
        "validate", "--inp", files["inp"], "--d", str(d_in),
        "--cost", files["cost"],
    ])
    assert code == 0
    assert "unusually small" in out
    assert "--diam-unit in" in out


def test_validate_no_warning_for_normal_mm(files):
    """SI network + --diam-unit mm + values 150-250 → no warning."""
    code, out, _ = _run_cli([
        "validate", "--inp", files["inp"], "--d", files["d"],
        "--cost", files["cost"],
    ])
    assert code == 0
    assert "unusually small" not in out
    assert "values 150" in out         # shows the actual value range


def test_run_warns_on_suspicious_diam_unit(files, tmp_path):
    """run warns to stderr but proceeds; exit code still 0 or 1."""
    d_in = tmp_path / "D_small.txt"
    d_in.write_text(D_TXT_INCHES_FOR_SI)
    code, _, err = _run_cli([
        "run",
        "--inp", files["inp"], "--d", str(d_in),
        "--cost", files["cost"],
        "--ns", "10", "--max-gen", "5", "--pmin", "30",
        "--quiet",
    ])
    assert code in (0, 1)
    assert "unusually small" in err
    assert "--diam-unit in" in err

# ---------------------------------------------------------------------------
# CLI output must remain ASCII (Windows portability guard)
# ---------------------------------------------------------------------------

def test_cli_source_has_no_non_ascii_outside_comments():
    """Guard against UnicodeEncodeError on Windows.

    Windows consoles with cp1252 / cp1256 codepages raise
    UnicodeEncodeError when a non-ASCII character reaches stdout or
    stderr through a pipe (which is what happens when the CLI is
    invoked by ``subprocess.run(capture_output=True)``, piped to a
    file, or run in CI).

    This test strips ``#`` comments and fails if any non-ASCII bytes
    remain in ``cli.py``'s source.  Docstrings are also caught — if a
    contributor ever needs non-ASCII prose in a docstring, they should
    move it to a comment.
    """
    import re as _re
    src_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "wds_optimizer" / "cli.py"
    )
    text = src_path.read_text(encoding="utf-8")
    # Strip line comments.  (Docstrings are left in — a bit conservative,
    # but the CLI currently has no non-ASCII docstrings, so it stays true.)
    no_comments = "\n".join(
        line.split("#", 1)[0] for line in text.splitlines()
    )
    leaked = sorted({
        (ch, f"U+{ord(ch):04X}")
        for ch in no_comments
        if ord(ch) > 127
    })
    if leaked:
        pytest.fail(
            "cli.py contains non-ASCII characters that may reach "
            "stdout/stderr via print().  Windows consoles with non-UTF-8 "
            "codepages raise UnicodeEncodeError on these characters.\n"
            f"Found: {leaked}\n"
            "Use ASCII equivalents: '[!]' for warnings, '->' for arrows, "
            "'-' for dashes."
        )