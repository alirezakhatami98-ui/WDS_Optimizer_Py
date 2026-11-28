import numpy as np
import pytest

from wds_optimizer.hydraulics.results import HydraulicResult


def test_converged_result_accepts_hydraulic_arrays():
    pressures = np.array([30.0, 25.0])
    velocities = np.array([1.2, 0.8])

    result = HydraulicResult(
        pressures=pressures,
        velocities=velocities,
        converged=True,
    )

    np.testing.assert_array_equal(result.pressures, pressures)
    np.testing.assert_array_equal(result.velocities, velocities)
    assert result.converged is True
    assert result.warnings == ()


def test_failed_result_can_have_none_arrays():
    result = HydraulicResult(
        pressures=None,
        velocities=None,
        converged=False,
    )

    assert result.pressures is None
    assert result.velocities is None
    assert result.converged is False


def test_arrays_are_independent_copies():
    pressures = np.array([30.0, 25.0])
    velocities = np.array([1.2, 0.8])

    result = HydraulicResult(
        pressures=pressures,
        velocities=velocities,
        converged=True,
    )

    pressures[0] = 999.0
    velocities[0] = 999.0

    assert result.pressures[0] == 30.0
    assert result.velocities[0] == 1.2


def test_arrays_are_read_only():
    result = HydraulicResult(
        pressures=np.array([30.0, 25.0]),
        velocities=np.array([1.2, 0.8]),
        converged=True,
    )

    assert result.pressures.flags.writeable is False
    assert result.velocities.flags.writeable is False

    with pytest.raises(ValueError):
        result.pressures[0] = 100.0

    with pytest.raises(ValueError):
        result.velocities[0] = 100.0


def test_warnings_are_normalized_to_tuple():
    result = HydraulicResult(
        pressures=np.array([30.0]),
        velocities=np.array([1.2]),
        converged=True,
        warnings=["warning A", "warning B"],
    )

    assert result.warnings == ("warning A", "warning B")
    assert isinstance(result.warnings, tuple)


def test_converged_result_requires_both_arrays():
    with pytest.raises(ValueError):
        HydraulicResult(
            pressures=None,
            velocities=np.array([1.2]),
            converged=True,
        )

    with pytest.raises(ValueError):
        HydraulicResult(
            pressures=np.array([30.0]),
            velocities=None,
            converged=True,
        )


@pytest.mark.parametrize("field_name", ["pressures", "velocities"])
def test_arrays_must_be_one_dimensional(field_name):
    values = {
        "pressures": np.array([30.0]),
        "velocities": np.array([1.2]),
        "converged": True,
    }
    values[field_name] = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="one-dimensional"):
        HydraulicResult(**values)


def test_converged_must_be_boolean():
    with pytest.raises(TypeError, match="converged"):
        HydraulicResult(
            pressures=np.array([30.0]),
            velocities=np.array([1.2]),
            converged=1,
        )


def test_warnings_must_contain_only_strings():
    with pytest.raises(TypeError, match="warning"):
        HydraulicResult(
            pressures=np.array([30.0]),
            velocities=np.array([1.2]),
            converged=True,
            warnings=["valid warning", 123],
        )