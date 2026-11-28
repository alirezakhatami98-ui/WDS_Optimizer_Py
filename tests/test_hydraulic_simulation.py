from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wntr.epanet.exceptions import EpanetException
from wntr.sim.results import ResultsStatus

from wds_optimizer.hydraulics import simulation


class FakeNetwork:
    def __init__(self):
        self.links = {
            "P1": SimpleNamespace(diameter=100.0),
        }

    def get_link(self, name):
        return self.links[name]


def make_results(
    *,
    error_code=ResultsStatus.converged,
    pressures=(30.0, 25.0),
    velocities=(1.2,),
):
    return SimpleNamespace(
        error_code=error_code,
        node={
            "pressure": pd.DataFrame(
                [pressures],
                columns=["J1", "J2"],
            )
        },
        link={
            "velocity": pd.DataFrame(
                [velocities],
                columns=["P1"],
            )
        },
    )


class FakeSimulator:
    result = None
    exception = None
    error_code = 0
    warnings = []

    def __init__(self, wn):
        self.wn = wn
        self.enData = SimpleNamespace(
            errcode=self.error_code,
            errcodelist=list(self.warnings),
        )

    def run_sim(self, **kwargs):
        assert kwargs["convergence_error"] is False

        if self.exception is not None:
            raise self.exception

        return self.result


@pytest.fixture
def setup_simulator(monkeypatch):
    def configure(
        *,
        result=None,
        exception=None,
        error_code=0,
        warnings=None,
    ):
        class ConfiguredSimulator(FakeSimulator):
            pass

        ConfiguredSimulator.result = result
        ConfiguredSimulator.exception = exception
        ConfiguredSimulator.error_code = error_code
        ConfiguredSimulator.warnings = warnings or []

        monkeypatch.setattr(
            simulation,
            "EpanetSimulator",
            ConfiguredSimulator,
        )

    return configure


def run_new(wn):
    return simulation._run_hydraulic_simulation_result(
        wn,
        full_diameters=[150.0],
        pipe_names=["P1"],
        junction_names=["J1", "J2"],
    )


def test_success_preserves_arrays_and_warnings(setup_simulator):
    setup_simulator(
        result=make_results(),
        warnings=["Example EPANET warning"],
    )

    result = run_new(FakeNetwork())

    assert result.converged is True
    np.testing.assert_array_equal(result.pressures, [30.0, 25.0])
    np.testing.assert_array_equal(result.velocities, [1.2])
    assert result.warnings == ("Example EPANET warning",)


def test_incomplete_results_are_hydraulic_failure(setup_simulator):
    setup_simulator(
        result=make_results(error_code=ResultsStatus.error),
        warnings=["Incomplete simulation"],
    )

    result = run_new(FakeNetwork())

    assert result.converged is False
    assert result.pressures is None
    assert result.velocities is None
    assert result.warnings == ("Incomplete simulation",)


def test_epanet_error_110_is_hydraulic_failure(setup_simulator):
    setup_simulator(
        exception=EpanetException(110),
        error_code=110,
        warnings=["Hydraulic warning"],
    )

    result = run_new(FakeNetwork())

    assert result.converged is False
    assert result.pressures is None
    assert result.velocities is None
    assert result.warnings == ("Hydraulic warning",)


@pytest.mark.parametrize("error_code", [101, 999])
def test_unclassified_epanet_errors_propagate(
    setup_simulator,
    error_code,
):
    setup_simulator(
        exception=EpanetException(error_code),
        error_code=error_code,
    )

    with pytest.raises(EpanetException):
        run_new(FakeNetwork())


def test_non_finite_values_are_hydraulic_failure(setup_simulator):
    setup_simulator(
        result=make_results(
            pressures=(30.0, np.nan),
            velocities=(1.2,),
        )
    )

    result = run_new(FakeNetwork())

    assert result.converged is False
    assert np.isnan(result.pressures[1])


def test_negative_finite_pressure_is_not_solver_failure(setup_simulator):
    setup_simulator(
        result=make_results(
            pressures=(-2.0, 25.0),
        )
    )

    result = run_new(FakeNetwork())

    assert result.converged is True
    np.testing.assert_array_equal(result.pressures, [-2.0, 25.0])


def test_new_function_preserves_network_mutation(setup_simulator):
    setup_simulator(result=make_results())

    wn = FakeNetwork()
    run_new(wn)

    assert wn.links["P1"].diameter == 150.0

def test_legacy_function_returns_three_values():
    wn = FakeNetwork()

    result = simulation.run_hydraulic_simulation(
        wn,
        full_diameters=[150.0],
        pipe_names=["P1"],
        junction_names=["J1", "J2"],
    )

    assert isinstance(result, tuple)
    assert len(result) == 3