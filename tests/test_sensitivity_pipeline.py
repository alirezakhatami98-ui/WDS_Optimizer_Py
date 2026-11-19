"""Smoke tests for scripts/sensitivity_analysis.py.

Only tests the pipeline mechanics, not the numerical results.
Numerical results live in docs/sensitivity_analysis.md and are
documentation, not assertions.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "sensitivity_analysis.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "_sens_script", SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sens_script"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _hanoi_present() -> bool:
    return (
        Path(__file__).resolve().parent.parent
        / "benchmarks" / "hanoi" / "hanoi.inp"
    ).exists()


# ---------------------------------------------------------------------
#  Dry run
# ---------------------------------------------------------------------

def test_dry_run_quick_prints_plan(script, tmp_path, capsys):
    code = script.main([
        "--quick", "--dry-run", "--output-dir", str(tmp_path),
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "Dry-run:" in captured.out
    assert "twoloop" in captured.out
    # No files written
    assert not any(tmp_path.rglob("*.json"))


def test_dry_run_no_plan_errors(script):
    """Without a plan source, main() returns non-zero."""
    code = script.main(["--dry-run"])
    assert code != 0


# ---------------------------------------------------------------------
#  Mini-sweep (real, small)
# ---------------------------------------------------------------------

def test_mini_sweep_writes_json(script, tmp_path):
    if not _hanoi_present():
        pytest.skip("benchmarks/hanoi not present")

    code = script.main([
        "--benchmark", "hanoi",
        "--algo", "GA",
        "--param", "GA.ga_pc",
        "--values", "0.6,0.9",
        "--runs", "1",
        "--max-gen", "5",
        "--output-dir", str(tmp_path),
        "--markdown", str(tmp_path / "sens.md"),
    ])
    assert code == 0

    jsons = list(tmp_path.rglob("*.json"))
    assert len(jsons) == 1

    data = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert data["benchmark"] == "hanoi"
    assert data["algorithm"] == "GA"
    assert data["parameter"] == "GA.ga_pc"
    assert data["values"] == [0.6, 0.9]
    assert data["n_runs"] == 1
    assert data["experiment_seed"] == 42
    assert len(data["points"]) == 2
    for p in data["points"]:
        assert "value" in p
        assert "status" in p
        assert "best_score" in p


# ---------------------------------------------------------------------
#  Markdown regeneration from fixtures
# ---------------------------------------------------------------------

def test_regenerate_only_creates_markdown(script, tmp_path):
    # Hand-crafted JSON fixture
    jp = tmp_path / "hanoi" / "GA" / "GA_ga_pc.json"
    jp.parent.mkdir(parents=True)
    jp.write_text(json.dumps({
        "benchmark": "hanoi",
        "algorithm": "GA",
        "parameter": "GA.ga_pc",
        "values": [0.6, 0.9],
        "n_runs": 1,
        "experiment_seed": 42,
        "timestamp": "2026-01-01T00:00:00",
        "points": [
            {"value": 0.6, "status": "Success", "best_score": 1000.0,
             "mean_score": 1000.0, "std_score": 0.0, "feasible_pct": 100.0,
             "mean_time_s": 1.0, "error": ""},
            {"value": 0.9, "status": "Success", "best_score": 900.0,
             "mean_score": 900.0, "std_score": 0.0, "feasible_pct": 100.0,
             "mean_time_s": 1.0, "error": ""},
        ],
    }, indent=2), encoding="utf-8")

    md = tmp_path / "sens.md"
    code = script.main([
        "--regenerate-only",
        "--output-dir", str(tmp_path),
        "--markdown", str(md),
    ])
    assert code == 0
    assert md.exists()

    text = md.read_text(encoding="utf-8")
    assert "# Sensitivity Analysis" in text
    assert "hanoi" in text
    assert "GA" in text
    assert "GA.ga_pc" in text
    assert "Recommended" in text
    # 0.9 has the lower mean cost → recommended
    assert "`GA.ga_pc = 0.9`" in text


def test_regenerate_with_no_data(script, tmp_path):
    md = tmp_path / "empty.md"
    code = script.main([
        "--regenerate-only",
        "--output-dir", str(tmp_path / "empty_dir"),
        "--markdown", str(md),
    ])
    assert code == 0
    assert md.exists()
    assert "No sensitivity data" in md.read_text(encoding="utf-8")


# ---------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------

def test_unknown_param_errors(script, tmp_path):
    if not _hanoi_present():
        pytest.skip("benchmarks/hanoi not present")
    code = script.main([
        "--benchmark", "hanoi",
        "--algo", "GA",
        "--param", "GA.not_a_param",
        "--values", "1,2",
        "--runs", "1",
        "--max-gen", "5",
        "--output-dir", str(tmp_path),
        "--markdown", str(tmp_path / "sens.md"),
    ])
    assert code != 0


def test_param_mode_without_benchmark_errors(script):
    code = script.main([
        "--param", "GA.ga_pc",
        "--values", "0.6,0.9",
    ])
    assert code != 0