import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from wds_optimizer.algorithm_api.models import (
    OptimizationProblem,
    PipeInfo,
)
from wds_optimizer.algorithm_api.models import EvaluationResult, INVALID_VIOLATION
from wds_optimizer.algorithm_api.models import ConvergenceHistory
from wds_optimizer.algorithm_api.models import AlgorithmResult


def make_problem():
    return OptimizationProblem(
        pipes=(
            PipeInfo("P1", True, 200.0),
            PipeInfo("P2", False, 150.0),
            PipeInfo("P3", True),
        ),
        diameters=np.array([100.0, 150.0, 200.0]),
        junction_names=("J1", "J2"),
        units="SI",
    )


# ============================================================
# PipeInfo
# ============================================================

def test_pipe_info_creation():
    pipe = PipeInfo(
        name="P1",
        is_variable=True,
        initial_diameter=200.0,
    )

    assert pipe.name == "P1"
    assert pipe.is_variable is True
    assert pipe.initial_diameter == 200.0


def test_pipe_info_allows_missing_initial_diameter():
    pipe = PipeInfo(
        name="P1",
        is_variable=True,
    )

    assert pipe.initial_diameter is None


@pytest.mark.parametrize(
    "name",
    ["", " ", None, 123],
)
def test_pipe_info_rejects_invalid_name(name):
    with pytest.raises(ValueError):
        PipeInfo(name=name, is_variable=True)


@pytest.mark.parametrize(
    "value",
    ["yes", 1, 0, None],
)
def test_pipe_info_rejects_non_boolean_variable_flag(value):
    with pytest.raises(TypeError):
        PipeInfo(name="P1", is_variable=value)


@pytest.mark.parametrize(
    "diameter",
    [0, -10, np.nan, np.inf, -np.inf],
)
def test_pipe_info_rejects_invalid_initial_diameter(diameter):
    with pytest.raises(ValueError):
        PipeInfo(
            name="P1",
            is_variable=True,
            initial_diameter=diameter,
        )


def test_pipe_info_is_immutable():
    pipe = PipeInfo("P1", True, 200.0)

    with pytest.raises(FrozenInstanceError):
        pipe.name = "P2"


# ============================================================
# OptimizationProblem — basic properties
# ============================================================

def test_optimization_problem_properties():
    problem = make_problem()

    assert problem.n_pipes == 3
    assert problem.n_variable_pipes == 2
    assert problem.n_diameters == 3

    assert problem.pipe_names == ("P1", "P2", "P3")
    assert problem.variable_pipe_names == ("P1", "P3")
    assert problem.fixed_pipe_names == ("P2",)

    assert problem.junction_names == ("J1", "J2")
    assert problem.units == "SI"


def test_optimization_problem_preserves_pipe_order():
    problem = make_problem()

    assert problem.pipe_names == ("P1", "P2", "P3")
    assert problem.variable_pipe_names == ("P1", "P3")


def test_optimization_problem_converts_sequences_to_tuples():
    pipes = [
        PipeInfo("P1", True),
        PipeInfo("P2", False),
    ]
    junctions = ["J1", "J2"]

    problem = OptimizationProblem(
        pipes=pipes,
        diameters=[100.0, 150.0],
        junction_names=junctions,
        units="SI",
    )

    assert isinstance(problem.pipes, tuple)
    assert isinstance(problem.junction_names, tuple)

    assert problem.pipes == tuple(pipes)
    assert problem.junction_names == tuple(junctions)


# ============================================================
# OptimizationProblem — validation
# ============================================================

def test_optimization_problem_rejects_invalid_pipe_objects():
    with pytest.raises(TypeError):
        OptimizationProblem(
            pipes=("P1", "P2"),
            diameters=[100.0, 150.0],
            junction_names=("J1",),
            units="SI",
        )


def test_optimization_problem_rejects_duplicate_pipe_names():
    with pytest.raises(ValueError, match="Pipe names must be unique"):
        OptimizationProblem(
            pipes=(
                PipeInfo("P1", True),
                PipeInfo("P1", False),
            ),
            diameters=[100.0, 150.0],
            junction_names=("J1",),
            units="SI",
        )


def test_optimization_problem_rejects_duplicate_junction_names():
    with pytest.raises(
        ValueError,
        match="Junction names must be unique",
    ):
        OptimizationProblem(
            pipes=(PipeInfo("P1", True),),
            diameters=[100.0, 150.0],
            junction_names=("J1", "J1"),
            units="SI",
        )


@pytest.mark.parametrize(
    "junction_names",
    [
        ("",),
        (" ",),
        (None,),
        (123,),
    ],
)
def test_optimization_problem_rejects_invalid_junction_names(
    junction_names,
):
    with pytest.raises(ValueError):
        OptimizationProblem(
            pipes=(PipeInfo("P1", True),),
            diameters=[100.0, 150.0],
            junction_names=junction_names,
            units="SI",
        )


@pytest.mark.parametrize(
    "units",
    ["", " ", None, 123],
)
def test_optimization_problem_rejects_invalid_units(units):
    with pytest.raises(ValueError):
        OptimizationProblem(
            pipes=(PipeInfo("P1", True),),
            diameters=[100.0, 150.0],
            junction_names=("J1",),
            units=units,
        )


# ============================================================
# OptimizationProblem — diameter catalogue
# ============================================================

@pytest.mark.parametrize(
    "diameters",
    [
        [],
        [0.0, 100.0],
        [-10.0, 100.0],
        [100.0, np.nan],
        [100.0, np.inf],
        [100.0, -np.inf],
        [100.0, 100.0],
        [150.0, 100.0],
    ],
)
def test_optimization_problem_rejects_invalid_diameters(diameters):
    with pytest.raises(ValueError):
        OptimizationProblem(
            pipes=(PipeInfo("P1", True),),
            diameters=diameters,
            junction_names=("J1",),
            units="SI",
        )


def test_optimization_problem_rejects_multidimensional_diameters():
    with pytest.raises(ValueError):
        OptimizationProblem(
            pipes=(PipeInfo("P1", True),),
            diameters=[[100.0, 150.0]],
            junction_names=("J1",),
            units="SI",
        )


# ============================================================
# OptimizationProblem — immutability
# ============================================================

def test_optimization_problem_copies_diameter_array():
    original = np.array([100.0, 150.0, 200.0])

    problem = OptimizationProblem(
        pipes=(PipeInfo("P1", True),),
        diameters=original,
        junction_names=("J1",),
        units="SI",
    )

    original[0] = 999.0

    assert problem.diameters[0] == 100.0


def test_optimization_problem_diameter_array_is_read_only():
    problem = make_problem()

    with pytest.raises(ValueError):
        problem.diameters[0] = 999.0


def test_optimization_problem_diameter_array_cannot_be_made_writeable():
    problem = make_problem()

    with pytest.raises(ValueError):
        problem.diameters.flags.writeable = True


def test_optimization_problem_is_immutable():
    problem = make_problem()

    with pytest.raises(FrozenInstanceError):
        problem.units = "US"


def test_optimization_problem_sequences_are_immutable():
    problem = make_problem()

    with pytest.raises(AttributeError):
        problem.pipes.append(PipeInfo("P4", True))

    with pytest.raises(AttributeError):
        problem.junction_names.append("J3")

# ============================================================
# EvaluationResult — valid results
# ============================================================

def test_evaluation_result_feasible():
    result = EvaluationResult(
        cost=125000.0,
        violation=0.0,
        feasible=True,
        converged=True,
        pressures=np.array([30.0, 35.0]),
        velocities=np.array([1.2, 1.5]),
    )

    assert result.cost == 125000.0
    assert result.violation == 0.0
    assert result.feasible is True
    assert result.converged is True
    np.testing.assert_array_equal(result.pressures, [30.0, 35.0])
    np.testing.assert_array_equal(result.velocities, [1.2, 1.5])


def test_evaluation_result_converged_but_infeasible():
    result = EvaluationResult(
        cost=150000.0,
        violation=12.5,
        feasible=False,
        converged=True,
    )

    assert result.cost == 150000.0
    assert result.violation == 12.5
    assert result.feasible is False
    assert result.converged is True


def test_evaluation_result_failed_simulation():
    result = EvaluationResult(
        cost=np.inf,
        violation=INVALID_VIOLATION,
        feasible=False,
        converged=False,
    )

    assert result.cost == np.inf
    assert result.violation == INVALID_VIOLATION
    assert result.feasible is False
    assert result.converged is False
    assert result.pressures is None
    assert result.velocities is None


def test_evaluation_result_allows_missing_hydraulic_arrays():
    result = EvaluationResult(
        cost=1000.0,
        violation=0.0,
        feasible=True,
        converged=True,
    )

    assert result.pressures is None
    assert result.velocities is None


# ============================================================
# EvaluationResult — validation
# ============================================================

@pytest.mark.parametrize(
    "cost",
    [np.nan, -np.inf],
)
def test_evaluation_result_rejects_invalid_cost(cost):
    with pytest.raises(ValueError):
        EvaluationResult(
            cost=cost,
            violation=0.0,
            feasible=True,
            converged=True,
        )


@pytest.mark.parametrize(
    "violation",
    [-1.0, np.nan, np.inf, -np.inf],
)
def test_evaluation_result_rejects_invalid_violation(violation):
    with pytest.raises(ValueError):
        EvaluationResult(
            cost=1000.0,
            violation=violation,
            feasible=False,
            converged=True,
        )


def test_evaluation_result_rejects_non_boolean_feasible():
    with pytest.raises(TypeError):
        EvaluationResult(
            cost=1000.0,
            violation=0.0,
            feasible=1,
            converged=True,
        )


def test_evaluation_result_rejects_non_boolean_converged():
    with pytest.raises(TypeError):
        EvaluationResult(
            cost=1000.0,
            violation=0.0,
            feasible=True,
            converged=1,
        )


def test_evaluation_result_rejects_feasible_with_positive_violation():
    with pytest.raises(ValueError):
        EvaluationResult(
            cost=1000.0,
            violation=1.0,
            feasible=True,
            converged=True,
        )


def test_evaluation_result_rejects_non_converged_feasible_result():
    with pytest.raises(ValueError):
        EvaluationResult(
            cost=np.inf,
            violation=INVALID_VIOLATION,
            feasible=True,
            converged=False,
        )


def test_evaluation_result_rejects_non_converged_finite_cost():
    with pytest.raises(ValueError):
        EvaluationResult(
            cost=1000.0,
            violation=INVALID_VIOLATION,
            feasible=False,
            converged=False,
        )


def test_evaluation_result_rejects_non_converged_wrong_violation():
    with pytest.raises(ValueError):
        EvaluationResult(
            cost=np.inf,
            violation=10.0,
            feasible=False,
            converged=False,
        )


def test_evaluation_result_rejects_hydraulic_arrays_on_failure():
    with pytest.raises(ValueError):
        EvaluationResult(
            cost=np.inf,
            violation=INVALID_VIOLATION,
            feasible=False,
            converged=False,
            pressures=np.array([30.0]),
        )


def test_evaluation_result_rejects_infinite_cost_when_converged():
    with pytest.raises(ValueError):
        EvaluationResult(
            cost=np.inf,
            violation=0.0,
            feasible=True,
            converged=True,
        )


# ============================================================
# EvaluationResult — hydraulic arrays
# ============================================================

@pytest.mark.parametrize(
    "field",
    ["pressures", "velocities"],
)
def test_evaluation_result_rejects_non_finite_hydraulic_values(field):
    with pytest.raises(ValueError):
        EvaluationResult(
            cost=1000.0,
            violation=0.0,
            feasible=True,
            converged=True,
            **{field: np.array([1.0, np.nan])},
        )


@pytest.mark.parametrize(
    "field",
    ["pressures", "velocities"],
)
def test_evaluation_result_hydraulic_arrays_are_read_only(field):
    result = EvaluationResult(
        cost=1000.0,
        violation=0.0,
        feasible=True,
        converged=True,
        **{field: np.array([1.0, 2.0])},
    )

    array = getattr(result, field)

    with pytest.raises(ValueError):
        array[0] = 999.0


def test_evaluation_result_copies_hydraulic_arrays():
    original = np.array([30.0, 35.0])

    result = EvaluationResult(
        cost=1000.0,
        violation=0.0,
        feasible=True,
        converged=True,
        pressures=original,
    )

    original[0] = 999.0

    assert result.pressures[0] == 30.0

# ============================================================
# ConvergenceHistory — valid histories
# ============================================================

def test_convergence_history_creation():
    history = ConvergenceHistory(
        best_cost=np.array([200000.0, 180000.0, 150000.0]),
        best_violation=np.array([10.0, 5.0, 0.0]),
        best_feasible=np.array([False, False, True]),
    )

    np.testing.assert_array_equal(
        history.best_cost,
        [200000.0, 180000.0, 150000.0],
    )
    np.testing.assert_array_equal(
        history.best_violation,
        [10.0, 5.0, 0.0],
    )
    np.testing.assert_array_equal(
        history.best_feasible,
        [False, False, True],
    )


def test_convergence_history_allows_empty_arrays():
    history = ConvergenceHistory(
        best_cost=np.array([]),
        best_violation=np.array([]),
        best_feasible=np.array([]),
    )

    assert len(history.best_cost) == 0
    assert len(history.best_violation) == 0
    assert len(history.best_feasible) == 0


def test_convergence_history_allows_infinite_cost_for_infeasible_incumbent():
    history = ConvergenceHistory(
        best_cost=np.array([np.inf, np.inf]),
        best_violation=np.array([100.0, 50.0]),
        best_feasible=np.array([False, False]),
    )

    assert np.all(np.isinf(history.best_cost))


# ============================================================
# ConvergenceHistory — validation
# ============================================================

def test_convergence_history_rejects_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        ConvergenceHistory(
            best_cost=np.array([100.0, 90.0]),
            best_violation=np.array([10.0]),
            best_feasible=np.array([False, False]),
        )


@pytest.mark.parametrize(
    "field",
    ["best_cost", "best_violation", "best_feasible"],
)
def test_convergence_history_rejects_multidimensional_arrays(field):
    values = {
        "best_cost": [100.0],
        "best_violation": [10.0],
        "best_feasible": [False],
    }
    values[field] = [[values[field][0]]]

    with pytest.raises(ValueError):
        ConvergenceHistory(**{
            key: np.asarray(value)
            for key, value in values.items()
        })


@pytest.mark.parametrize(
    "cost",
    [
        [np.nan],
        [-np.inf],
    ],
)
def test_convergence_history_rejects_invalid_cost(cost):
    with pytest.raises(ValueError):
        ConvergenceHistory(
            best_cost=cost,
            best_violation=[0.0],
            best_feasible=[True],
        )


@pytest.mark.parametrize(
    "violation",
    [
        [-1.0],
        [np.nan],
        [np.inf],
        [-np.inf],
    ],
)
def test_convergence_history_rejects_invalid_violation(violation):
    with pytest.raises(ValueError):
        ConvergenceHistory(
            best_cost=[100.0],
            best_violation=violation,
            best_feasible=[False],
        )


def test_convergence_history_rejects_feasible_entry_with_positive_violation():
    with pytest.raises(ValueError):
        ConvergenceHistory(
            best_cost=[100.0],
            best_violation=[1.0],
            best_feasible=[True],
        )


# ============================================================
# ConvergenceHistory — snapshot and independence
# ============================================================

def test_convergence_history_snapshot_returns_independent_copy():
    history = ConvergenceHistory(
        best_cost=np.array([200.0, 150.0]),
        best_violation=np.array([10.0, 0.0]),
        best_feasible=np.array([False, True]),
    )

    snapshot = history.snapshot()

    assert snapshot is not history

    np.testing.assert_array_equal(
        snapshot.best_cost,
        history.best_cost,
    )

    snapshot.best_cost[0] = 999.0

    assert history.best_cost[0] == 200.0


def test_convergence_history_copies_input_arrays():
    original_cost = np.array([200.0, 150.0])
    original_violation = np.array([10.0, 0.0])
    original_feasible = np.array([False, True])

    history = ConvergenceHistory(
        best_cost=original_cost,
        best_violation=original_violation,
        best_feasible=original_feasible,
    )

    original_cost[0] = 999.0
    original_violation[0] = 999.0
    original_feasible[0] = True

    assert history.best_cost[0] == 200.0
    assert history.best_violation[0] == 10.0
    assert history.best_feasible[0] is np.False_


def test_convergence_history_arrays_are_mutable():
    history = ConvergenceHistory(
        best_cost=[200.0, 150.0],
        best_violation=[10.0, 0.0],
        best_feasible=[False, True],
    )

    history.best_cost[0] = 180.0
    history.best_violation[0] = 5.0

    assert history.best_cost[0] == 180.0
    assert history.best_violation[0] == 5.0

# ============================================================
# AlgorithmResult — valid results
# ============================================================

def make_evaluation():
    return EvaluationResult(
        cost=125000.0,
        violation=0.0,
        feasible=True,
        converged=True,
    )


def make_history():
    return ConvergenceHistory(
        best_cost=[150000.0, 125000.0],
        best_violation=[10.0, 0.0],
        best_feasible=[False, True],
    )


def test_algorithm_result_with_valid_candidate():
    result = AlgorithmResult(
        best_candidate=np.array([0, 2, 1], dtype=int),
        best_evaluation=make_evaluation(),
        convergence=make_history(),
    )

    np.testing.assert_array_equal(
        result.best_candidate,
        [0, 2, 1],
    )

    assert result.best_evaluation.cost == 125000.0
    assert isinstance(result.convergence, ConvergenceHistory)


def test_algorithm_result_without_candidate():
    result = AlgorithmResult(
        best_candidate=None,
        best_evaluation=None,
        convergence=make_history(),
    )

    assert result.best_candidate is None
    assert result.best_evaluation is None


def test_algorithm_result_allows_infeasible_best_solution():
    evaluation = EvaluationResult(
        cost=150000.0,
        violation=10.0,
        feasible=False,
        converged=True,
    )

    result = AlgorithmResult(
        best_candidate=np.array([1, 2, 0]),
        best_evaluation=evaluation,
        convergence=make_history(),
    )

    assert result.best_candidate is not None
    assert result.best_evaluation.feasible is False


# ============================================================
# AlgorithmResult — candidate validation
# ============================================================

def test_algorithm_result_rejects_candidate_without_evaluation():
    with pytest.raises(ValueError):
        AlgorithmResult(
            best_candidate=np.array([0, 1]),
            best_evaluation=None,
            convergence=make_history(),
        )


def test_algorithm_result_rejects_evaluation_without_candidate():
    with pytest.raises(ValueError):
        AlgorithmResult(
            best_candidate=None,
            best_evaluation=make_evaluation(),
            convergence=make_history(),
        )


def test_algorithm_result_rejects_multidimensional_candidate():
    with pytest.raises(ValueError):
        AlgorithmResult(
            best_candidate=np.array([[0, 1], [2, 3]]),
            best_evaluation=make_evaluation(),
            convergence=make_history(),
        )


@pytest.mark.parametrize(
    "candidate",
    [
        np.array([0.0, 1.0]),
        np.array([0.5, 1.5]),
        np.array([True, False]),
    ],
)
def test_algorithm_result_rejects_non_integer_candidate(candidate):
    with pytest.raises(TypeError):
        AlgorithmResult(
            best_candidate=candidate,
            best_evaluation=make_evaluation(),
            convergence=make_history(),
        )


def test_algorithm_result_rejects_negative_candidate_indices():
    with pytest.raises(ValueError):
        AlgorithmResult(
            best_candidate=np.array([0, -1, 2]),
            best_evaluation=make_evaluation(),
            convergence=make_history(),
        )


# ============================================================
# AlgorithmResult — diagnostics
# ============================================================

def test_algorithm_result_default_diagnostics():
    result = AlgorithmResult(
        best_candidate=np.array([0, 1]),
        best_evaluation=make_evaluation(),
        convergence=make_history(),
    )

    assert result.diagnostics == {
        "algorithm_specific": {},
    }


def test_algorithm_result_accepts_algorithm_specific_diagnostics():
    result = AlgorithmResult(
        best_candidate=np.array([0, 1]),
        best_evaluation=make_evaluation(),
        convergence=make_history(),
        diagnostics={
            "algorithm_specific": {
                "mutation_rate": 0.2,
                "iterations": 100,
            }
        },
    )

    assert result.diagnostics["algorithm_specific"]["mutation_rate"] == 0.2
    assert result.diagnostics["algorithm_specific"]["iterations"] == 100


def test_algorithm_result_rejects_core_owned_diagnostics():
    with pytest.raises(ValueError):
        AlgorithmResult(
            best_candidate=np.array([0, 1]),
            best_evaluation=make_evaluation(),
            convergence=make_history(),
            diagnostics={
                "nfe": 100,
            },
        )


def test_algorithm_result_rejects_non_dictionary_diagnostics():
    with pytest.raises(TypeError):
        AlgorithmResult(
            best_candidate=np.array([0, 1]),
            best_evaluation=make_evaluation(),
            convergence=make_history(),
            diagnostics=[],
        )


def test_algorithm_result_rejects_non_dictionary_algorithm_specific():
    with pytest.raises(TypeError):
        AlgorithmResult(
            best_candidate=np.array([0, 1]),
            best_evaluation=make_evaluation(),
            convergence=make_history(),
            diagnostics={
                "algorithm_specific": [],
            },
        )


# ============================================================
# AlgorithmResult — immutability and independence
# ============================================================

def test_algorithm_result_candidate_is_read_only():
    result = AlgorithmResult(
        best_candidate=np.array([0, 1, 2]),
        best_evaluation=make_evaluation(),
        convergence=make_history(),
    )

    with pytest.raises(ValueError):
        result.best_candidate[0] = 2


def test_algorithm_result_copies_candidate():
    candidate = np.array([0, 1, 2])

    result = AlgorithmResult(
        best_candidate=candidate,
        best_evaluation=make_evaluation(),
        convergence=make_history(),
    )

    candidate[0] = 99

    assert result.best_candidate[0] == 0


def test_algorithm_result_copies_convergence_history():
    history = make_history()

    result = AlgorithmResult(
        best_candidate=np.array([0, 1]),
        best_evaluation=make_evaluation(),
        convergence=history,
    )

    history.best_cost[0] = 999999.0

    assert result.convergence.best_cost[0] == 150000.0
    assert result.convergence is not history


def test_algorithm_result_normalizes_diagnostics():
    original = {
        "algorithm_specific": {
            "iterations": 100,
        }
    }

    result = AlgorithmResult(
        best_candidate=np.array([0, 1]),
        best_evaluation=make_evaluation(),
        convergence=make_history(),
        diagnostics=original,
    )

    original["algorithm_specific"]["iterations"] = 999

    # The nested dictionary is copied independently.
    assert result.diagnostics["algorithm_specific"]["iterations"] == 100

# ============================================================
# Cross-model contract and edge cases
# ============================================================

def test_feasible_evaluation_and_history_are_consistent():
    evaluation = make_evaluation()

    history = ConvergenceHistory(
        best_cost=[150000.0, evaluation.cost],
        best_violation=[10.0, evaluation.violation],
        best_feasible=[False, evaluation.feasible],
    )

    assert history.best_cost[-1] == evaluation.cost
    assert history.best_violation[-1] == evaluation.violation
    assert history.best_feasible[-1] == evaluation.feasible


def test_failed_evaluation_is_not_a_feasible_solution():
    evaluation = EvaluationResult(
        cost=np.inf,
        violation=INVALID_VIOLATION,
        feasible=False,
        converged=False,
    )

    assert evaluation.feasible is False
    assert evaluation.converged is False
    assert evaluation.cost == np.inf


def test_algorithm_result_candidate_is_independent_of_problem():
    problem = make_problem()
    candidate = np.array([0, 2])

    result = AlgorithmResult(
        best_candidate=candidate,
        best_evaluation=make_evaluation(),
        convergence=make_history(),
    )

    candidate[0] = 99

    assert result.best_candidate[0] == 0
    assert problem.n_variable_pipes == len(result.best_candidate)


def test_problem_diameter_catalogue_is_independent_of_input():
    original = np.array([100.0, 150.0, 200.0])

    problem = OptimizationProblem(
        pipes=(PipeInfo("P1", True),),
        diameters=original,
        junction_names=("J1",),
        units="SI",
    )

    original[:] = 999.0

    np.testing.assert_array_equal(
        problem.diameters,
        [100.0, 150.0, 200.0],
    )


def test_algorithm_result_preserves_infeasible_incumbent():
    evaluation = EvaluationResult(
        cost=150000.0,
        violation=5.0,
        feasible=False,
        converged=True,
    )

    result = AlgorithmResult(
        best_candidate=np.array([1, 2]),
        best_evaluation=evaluation,
        convergence=make_history(),
    )

    assert result.best_evaluation.feasible is False
    assert result.best_evaluation.converged is True
    assert result.best_evaluation.violation == 5.0