
import numpy as np
import pytest

from wds_optimizer.algorithm_api.context import (
    AlgorithmContext,
    OptimizationCancelled,
)


# ============================================================
# RNG
# ============================================================

def test_context_accepts_numpy_generator():
    rng = np.random.default_rng(42)
    context = AlgorithmContext(rng=rng)

    assert context.rng is rng


def test_context_rejects_invalid_rng():
    with pytest.raises(TypeError):
        AlgorithmContext(rng=42)


def test_context_rng_is_reproducible():
    context_a = AlgorithmContext(rng=np.random.default_rng(42))
    context_b = AlgorithmContext(rng=np.random.default_rng(42))

    values_a = context_a.rng.random(10)
    values_b = context_b.rng.random(10)

    np.testing.assert_array_equal(values_a, values_b)


def test_context_rng_instances_are_independent():
    context_a = AlgorithmContext(rng=np.random.default_rng(42))
    context_b = AlgorithmContext(rng=np.random.default_rng(42))

    context_a.rng.random(5)

    values_a = context_a.rng.random(5)
    values_b = context_b.rng.random(5)

    assert not np.array_equal(values_a, values_b)


# ============================================================
# Progress reporting
# ============================================================

def test_report_progress_calls_callback():
    reports = []

    context = AlgorithmContext(
        rng=np.random.default_rng(42),
        progress_callback=reports.append,
    )

    context.report_progress(
        generation=10,
        max_generations=100,
        metrics={"best_cost": 125000.0, "feasible": True},
    )

    assert len(reports) == 1
    assert reports[0] == {
        "generation": 10,
        "max_generations": 100,
        "metrics": {
            "best_cost": 125000.0,
            "feasible": True,
        },
    }


def test_report_progress_without_callback():
    context = AlgorithmContext(rng=np.random.default_rng(42))

    context.report_progress(
        generation=1,
        max_generations=10,
    )


def test_report_progress_without_metrics():
    reports = []

    context = AlgorithmContext(
        rng=np.random.default_rng(42),
        progress_callback=reports.append,
    )

    context.report_progress(
        generation=1,
        max_generations=10,
    )

    assert reports[0]["metrics"] == {}


@pytest.mark.parametrize(
    "generation",
    [0, -1, 11, 1.5, True],
)
def test_report_progress_rejects_invalid_generation(generation):
    context = AlgorithmContext(rng=np.random.default_rng(42))

    with pytest.raises((TypeError, ValueError)):
        context.report_progress(
            generation=generation,
            max_generations=10,
        )


@pytest.mark.parametrize(
    "max_generations",
    [0, -1, 1.5, True],
)
def test_report_progress_rejects_invalid_max_generations(max_generations):
    context = AlgorithmContext(rng=np.random.default_rng(42))

    with pytest.raises((TypeError, ValueError)):
        context.report_progress(
            generation=1,
            max_generations=max_generations,
        )


def test_report_progress_rejects_invalid_metrics():
    context = AlgorithmContext(rng=np.random.default_rng(42))

    with pytest.raises(TypeError):
        context.report_progress(
            generation=1,
            max_generations=10,
            metrics="invalid",
        )


def test_context_rejects_invalid_progress_callback():
    with pytest.raises(TypeError):
        AlgorithmContext(
            rng=np.random.default_rng(42),
            progress_callback=123,
        )


def test_progress_callback_receives_independent_metrics_dictionary():
    original_metrics = {"best_cost": 100.0}
    reports = []

    context = AlgorithmContext(
        rng=np.random.default_rng(42),
        progress_callback=reports.append,
    )

    context.report_progress(
        generation=1,
        max_generations=10,
        metrics=original_metrics,
    )

    original_metrics["best_cost"] = 999.0

    assert reports[0]["metrics"]["best_cost"] == 100.0


# ============================================================
# Cancellation
# ============================================================

def test_check_cancelled_does_not_raise_when_not_cancelled():
    context = AlgorithmContext(
        rng=np.random.default_rng(42),
        cancellation_check=lambda: False,
    )

    context.check_cancelled()


def test_check_cancelled_raises_when_cancelled():
    context = AlgorithmContext(
        rng=np.random.default_rng(42),
        cancellation_check=lambda: True,
    )

    with pytest.raises(OptimizationCancelled):
        context.check_cancelled()


def test_check_cancelled_without_callback():
    context = AlgorithmContext(rng=np.random.default_rng(42))

    context.check_cancelled()


def test_optimization_cancelled_default_partial_result():
    error = OptimizationCancelled()

    assert error.partial_result is None
    assert str(error) == "Optimization was cancelled."


def test_optimization_cancelled_preserves_partial_result():
    partial_result = {"best_cost": 125000.0}

    error = OptimizationCancelled(partial_result=partial_result)

    assert error.partial_result is partial_result


def test_context_rejects_invalid_cancellation_callback():
    with pytest.raises(TypeError):
        AlgorithmContext(
            rng=np.random.default_rng(42),
            cancellation_check=123,
        )
