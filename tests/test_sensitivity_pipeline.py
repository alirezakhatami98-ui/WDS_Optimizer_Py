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

def test_unknown_param_errors(script, tmp_path, capsys):
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
    err = capsys.readouterr().err
    # v0.8.0 renamed "sweep parameter" -> "parameter" throughout.
    assert "Unknown parameter" in err
    # Nothing was written
    assert not any(tmp_path.rglob("*.json"))


def test_unknown_algo_errors(script, tmp_path, capsys):
    if not _hanoi_present():
        pytest.skip("benchmarks/hanoi not present")
    code = script.main([
        "--benchmark", "hanoi",
        "--algo", "NOT_AN_ALGO",
        "--param", "GA.ga_pc",
        "--values", "0.6,0.9",
        "--runs", "1",
        "--max-gen", "5",
        "--output-dir", str(tmp_path),
        "--markdown", str(tmp_path / "sens.md"),
    ])
    assert code != 0
    err = capsys.readouterr().err
    assert "Unknown algorithm" in err


def test_param_algo_mismatch_errors(script, tmp_path, capsys):
    """GA.ga_pc with --algo PSO is a user mistake, not a silent no-op."""
    if not _hanoi_present():
        pytest.skip("benchmarks/hanoi not present")
    code = script.main([
        "--benchmark", "hanoi",
        "--algo", "PSO",
        "--param", "GA.ga_pc",
        "--values", "0.6,0.9",
        "--runs", "1",
        "--max-gen", "5",
        "--output-dir", str(tmp_path),
        "--markdown", str(tmp_path / "sens.md"),
    ])
    assert code != 0
    err = capsys.readouterr().err
    assert "belongs to algorithm" in err


def test_param_mode_without_benchmark_errors(script):
    code = script.main([
        "--param", "GA.ga_pc",
        "--values", "0.6,0.9",
    ])
    assert code != 0

# ---------------------------------------------------------------------
#  Annotation logic
# ---------------------------------------------------------------------

def _make_sweep_json_dict(
    *, n_runs=3,
    points=None,
    parameter="GA.ga_pc",
):
    """Helper to build a minimal sweep JSON dict for annotation tests."""
    if points is None:
        points = []
    return {
        "benchmark": "twoloop",
        "algorithm": "GA",
        "parameter": parameter,
        "values": [p["value"] for p in points],
        "n_runs": n_runs,
        "experiment_seed": 42,
        "timestamp": "2026-01-01T00:00:00",
        "points": points,
    }


def _make_point(
    value, mean, std, best=None, status="Success",
    feasible_pct=100.0, time_s=1.0, error="",
):
    return {
        "value": value,
        "status": status,
        "best_score": best if best is not None else mean,
        "mean_score": mean,
        "std_score": std,
        "feasible_pct": feasible_pct,
        "mean_time_s": time_s,
        "error": error,
    }


def test_annotation_detects_flat_line(script):
    """All means within 1% → flat line, no recommendation."""
    data = _make_sweep_json_dict(points=[
        _make_point(0.3, 422_000.0, 3_000.0),
        _make_point(0.5, 422_500.0, 2_500.0),
        _make_point(0.7, 422_200.0, 2_800.0),
    ])
    notes, is_flat = script._annotations_for_sweep(data)
    assert is_flat is True
    assert any("Uninformative sweep" in n for n in notes)


def test_annotation_significant_gap(script):
    """A large, low-variance gap → 'Significant' verdict."""
    data = _make_sweep_json_dict(points=[
        _make_point(0.3, 500_000.0, 5_000.0),
        _make_point(0.9, 420_000.0, 2_000.0),
    ])
    notes, is_flat = script._annotations_for_sweep(data)
    assert is_flat is False
    assert any("Significant" in n for n in notes)


def test_annotation_not_significant_gap(script):
    """A small gap relative to SEM → 'Not significant'.

    The data must NOT be a flat line (>1% spread between means),
    otherwise the flat-line branch fires first and suppresses the
    significance test.  Spread here: 10k / 420k ≈ 2.4% → not flat.
    """
    data = _make_sweep_json_dict(points=[
        _make_point(0.7, 415_000.0, 20_000.0),   # SEM ≈ 11.5k
        _make_point(0.8, 425_000.0, 20_000.0),   # SEM ≈ 11.5k
    ])
    notes, is_flat = script._annotations_for_sweep(data)
    assert is_flat is False
    assert any("Not significant" in n for n in notes)


def test_annotation_high_variance_flag(script):
    """CV > 3% flags the value."""
    data = _make_sweep_json_dict(points=[
        _make_point(0.6, 447_000.0, 24_500.0),   # CV ~ 5.5%
        _make_point(0.7, 428_000.0, 8_000.0),    # CV ~ 1.9%
    ])
    notes, _ = script._annotations_for_sweep(data)
    assert any("High variance" in n and "0.6" in n for n in notes)


def test_annotation_failed_points(script):
    data = _make_sweep_json_dict(points=[
        _make_point(0.6, 420_000.0, 1_000.0),
        _make_point(0.9, None, None, status="Failed",
                    best=None, time_s=0.0, error="boom"),
    ])
    # Patch in "None" mean for failed
    data["points"][1]["mean_score"] = None
    data["points"][1]["best_score"] = None
    data["points"][1]["std_score"] = None
    notes, _ = script._annotations_for_sweep(data)
    assert any("Failed points" in n for n in notes)
    assert any("boom" in n for n in notes)

def test_annotation_mixed_sweep_reports_failures_and_comparison(script):
    """A sweep with 2+ successful points AND some failed ones must
    report both the failure(s) AND the significance result."""
    data = _make_sweep_json_dict(points=[
        _make_point(0.6, 500_000.0, 5_000.0),
        _make_point(0.9, 420_000.0, 2_000.0),
        _make_point(1.0, None, None, status="Failed",
                    best=None, time_s=0.0, error="diverged"),
    ])
    data["points"][2]["mean_score"] = None
    data["points"][2]["best_score"] = None
    data["points"][2]["std_score"] = None
    notes, is_flat = script._annotations_for_sweep(data)
    assert is_flat is False
    assert any("Failed points" in n for n in notes)
    assert any("diverged" in n for n in notes)
    assert any("Significant" in n for n in notes)

def test_annotation_timing_anomaly(script):
    """One value 1.5× slower than median → flag.

    Uses non-flat means (spread 6k / 422.7k ≈ 1.4%) so the flat-line
    branch does not fire.  The timing check runs regardless of
    flat/not-flat, but the significance check would also run here —
    that is fine, the test only asserts the timing note.
    """
    data = _make_sweep_json_dict(points=[
        _make_point(100, 420_000.0, 2_800.0, time_s=4.8),
        _make_point(300, 422_000.0, 2_800.0, time_s=2.3),
        _make_point(500, 426_000.0, 2_800.0, time_s=2.4),
    ])
    notes, is_flat = script._annotations_for_sweep(data)
    assert is_flat is False
    assert any("Timing anomaly" in n and "100" in n for n in notes)


def test_param_section_md_suppresses_recommendation_for_flat(script):
    data = _make_sweep_json_dict(points=[
        _make_point(100, 422_000.0, 2_800.0, time_s=2.0),
        _make_point(300, 422_100.0, 2_900.0, time_s=2.1),
    ])
    lines = script._param_section_md("SADE.sade_stag", data)
    text = "\n".join(lines)
    assert "Uninformative sweep" in text
    assert "_none — the sweep is uninformative" in text
    # Must NOT recommend a specific value
    assert "lowest mean cost" not in text


def test_param_section_md_has_recommendation_when_clear_winner(script):
    data = _make_sweep_json_dict(points=[
        _make_point(0.3, 500_000.0, 5_000.0),
        _make_point(0.9, 420_000.0, 2_000.0),
    ])
    lines = script._param_section_md("DE.de_cr", data)
    text = "\n".join(lines)
    assert "lowest mean cost" in text
    assert "`DE.de_cr = 0.9`" in text
    assert "Significant" in text