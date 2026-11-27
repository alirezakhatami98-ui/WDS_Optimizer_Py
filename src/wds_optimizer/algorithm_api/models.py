"""Public data models for Algorithm Plugin API v2."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np


INVALID_VIOLATION = 1e8


@dataclass(frozen=True)
class _FrozenMapping(Mapping[str, Any]):
    """Small immutable and pickleable mapping."""

    _items: tuple[tuple[str, Any], ...]

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value

        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class AlgorithmConfig:
    """Immutable configuration provided to an API v2 algorithm."""

    seed: int
    population_size: int
    max_generations: int
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("seed", "population_size", "max_generations"):
            value = getattr(self, name)

            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, Integral
            ):
                raise TypeError(f"{name} must be an integer.")

            value = int(value)

            if name == "seed":
                if value < 0:
                    raise ValueError("seed must be non-negative.")
            elif value < 1:
                raise ValueError(f"{name} must be positive.")

            object.__setattr__(self, name, value)

        if not isinstance(self.params, Mapping):
            raise TypeError("params must be a mapping.")

        normalized_params: dict[str, Any] = {}

        reserved_names = {
            "seed",
            "population_size",
            "max_generations",
        }

        for key, value in self.params.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    "Parameter keys must be non-empty strings."
                )

            key = key.strip()

            if key in reserved_names:
                raise ValueError(
                    f"'{key}' is a reserved configuration field."
                )

            if key in normalized_params:
                raise ValueError(
                    f"Duplicate parameter key: '{key}'."
                )

            if isinstance(value, np.generic):
                value = value.item()

            if isinstance(value, bool):
                pass
            elif isinstance(value, int):
                pass
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(
                        f"Parameter '{key}' must be finite."
                    )
            elif isinstance(value, str):
                pass
            else:
                raise TypeError(
                    f"Parameter '{key}' must be a scalar "
                    "string, boolean, integer, or float."
                )

            normalized_params[key] = value

        object.__setattr__(
            self,
            "params",
            _FrozenMapping(tuple(normalized_params.items())),
        )


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype | type | None = None,
    ndim: int = 1,
) -> np.ndarray:
    """Return an independent NumPy array backed by immutable bytes."""
    array = np.asarray(value, dtype=dtype)

    if array.ndim != ndim:
        raise ValueError(
            f"Expected a {ndim}-dimensional array, got {array.ndim}."
        )

    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(
        contiguous.tobytes(),
        dtype=contiguous.dtype,
    ).reshape(contiguous.shape)


@dataclass(frozen=True)
class PipeInfo:
    """Public information about a single network pipe."""

    name: str
    is_variable: bool
    initial_diameter: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Pipe name must be a non-empty string.")

        if not isinstance(self.is_variable, (bool, np.bool_)):
            raise TypeError("is_variable must be a boolean.")

        object.__setattr__(self, "is_variable", bool(self.is_variable))

        if self.initial_diameter is not None:
            diameter = float(self.initial_diameter)

            if not np.isfinite(diameter) or diameter <= 0:
                raise ValueError(
                    "initial_diameter must be finite and positive."
                )

            object.__setattr__(self, "initial_diameter", diameter)


@dataclass(frozen=True)
class OptimizationProblem:
    """Immutable public representation of an optimization problem."""

    pipes: tuple[PipeInfo, ...]
    diameters: np.ndarray
    junction_names: tuple[str, ...]
    units: str

    def __post_init__(self) -> None:
        pipes = tuple(self.pipes)
        junction_names = tuple(self.junction_names)

        if not all(isinstance(pipe, PipeInfo) for pipe in pipes):
            raise TypeError("pipes must contain only PipeInfo objects.")

        if len({pipe.name for pipe in pipes}) != len(pipes):
            raise ValueError("Pipe names must be unique.")

        if not all(
            isinstance(name, str) and name.strip()
            for name in junction_names
        ):
            raise ValueError(
                "junction_names must contain non-empty strings."
            )

        if len(set(junction_names)) != len(junction_names):
            raise ValueError("Junction names must be unique.")

        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a non-empty string.")

        diameters = _readonly_array(self.diameters, dtype=np.float64)

        if diameters.size == 0:
            raise ValueError("At least one diameter is required.")

        if not np.all(np.isfinite(diameters)) or np.any(diameters <= 0):
            raise ValueError("Diameters must be finite and positive.")

        if np.any(np.diff(diameters) <= 0):
            raise ValueError(
                "Diameters must be strictly increasing."
            )

        object.__setattr__(self, "pipes", pipes)
        object.__setattr__(self, "diameters", diameters)
        object.__setattr__(self, "junction_names", junction_names)
        object.__setattr__(self, "units", self.units.strip())

    @property
    def n_pipes(self) -> int:
        return len(self.pipes)

    @property
    def n_variable_pipes(self) -> int:
        return sum(pipe.is_variable for pipe in self.pipes)

    @property
    def n_diameters(self) -> int:
        return len(self.diameters)

    @property
    def pipe_names(self) -> tuple[str, ...]:
        return tuple(pipe.name for pipe in self.pipes)

    @property
    def variable_pipe_names(self) -> tuple[str, ...]:
        return tuple(
            pipe.name for pipe in self.pipes if pipe.is_variable
        )

    @property
    def fixed_pipe_names(self) -> tuple[str, ...]:
        return tuple(
            pipe.name for pipe in self.pipes if not pipe.is_variable
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable result of evaluating one candidate."""

    cost: float
    violation: float
    feasible: bool
    converged: bool
    pressures: np.ndarray | None = None
    velocities: np.ndarray | None = None

    def __post_init__(self) -> None:
        cost = float(self.cost)
        violation = float(self.violation)

        if np.isnan(cost) or cost == -np.inf:
            raise ValueError("cost must be finite or positive infinity.")

        if not np.isfinite(violation) or violation < 0:
            raise ValueError("violation must be finite and non-negative.")

        if not isinstance(self.feasible, (bool, np.bool_)):
            raise TypeError("feasible must be a boolean.")

        if not isinstance(self.converged, (bool, np.bool_)):
            raise TypeError("converged must be a boolean.")

        feasible = bool(self.feasible)
        converged = bool(self.converged)

        if not converged:
            if feasible:
                raise ValueError("A non-converged result cannot be feasible.")
            if cost != np.inf:
                raise ValueError("A non-converged result must have infinite cost.")
            if violation != INVALID_VIOLATION:
                raise ValueError(
                    f"A non-converged result must have violation "
                    f"{INVALID_VIOLATION}."
                )
            if self.pressures is not None or self.velocities is not None:
                raise ValueError(
                    "Hydraulic arrays must be None when simulation fails."
                )

        elif cost == np.inf:
            raise ValueError(
                "A converged result must have a finite cost."
            )

        if feasible and violation != 0:
            raise ValueError(
                "A feasible result must have zero violation."
            )

        pressures = (
            None
            if self.pressures is None
            else _readonly_array(self.pressures, dtype=np.float64)
        )
        velocities = (
            None
            if self.velocities is None
            else _readonly_array(self.velocities, dtype=np.float64)
        )

        for name, array in (
            ("pressures", pressures),
            ("velocities", velocities),
        ):
            if array is not None and not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain finite values.")

        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "violation", violation)
        object.__setattr__(self, "feasible", feasible)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "pressures", pressures)
        object.__setattr__(self, "velocities", velocities)


@dataclass
class ConvergenceHistory:
    """Mutable history collected during algorithm execution."""

    best_cost: np.ndarray
    best_violation: np.ndarray
    best_feasible: np.ndarray

    def __post_init__(self) -> None:
        self.best_cost = np.asarray(self.best_cost, dtype=np.float64).copy()
        self.best_violation = np.asarray(
            self.best_violation, dtype=np.float64
        ).copy()
        self.best_feasible = np.asarray(
            self.best_feasible, dtype=np.bool_
        ).copy()

        if any(
            array.ndim != 1
            for array in (
                self.best_cost,
                self.best_violation,
                self.best_feasible,
            )
        ):
            raise ValueError("Convergence arrays must be one-dimensional.")

        lengths = {
            len(self.best_cost),
            len(self.best_violation),
            len(self.best_feasible),
        }

        if len(lengths) != 1:
            raise ValueError(
                "All convergence arrays must have the same length."
            )

        if np.any(np.isnan(self.best_cost)) or np.any(
            self.best_cost == -np.inf
        ):
            raise ValueError("best_cost must be finite or positive infinity.")

        if (
            not np.all(np.isfinite(self.best_violation))
            or np.any(self.best_violation < 0)
        ):
            raise ValueError(
                "best_violation must be finite and non-negative."
            )

        if np.any(self.best_feasible & (self.best_violation != 0)):
            raise ValueError(
                "Feasible history entries must have zero violation."
            )

    def snapshot(self) -> ConvergenceHistory:
        """Return an independent copy of this history."""
        return ConvergenceHistory(
            best_cost=self.best_cost,
            best_violation=self.best_violation,
            best_feasible=self.best_feasible,
        )


@dataclass
class AlgorithmResult:
    """Standard result returned by an API v2 algorithm."""

    best_candidate: np.ndarray | None
    best_evaluation: EvaluationResult | None
    convergence: ConvergenceHistory
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.best_candidate is None) != (
            self.best_evaluation is None
        ):
            raise ValueError(
                "best_candidate and best_evaluation must either "
                "both be present or both be None."
            )

        if self.best_candidate is not None:
            candidate = np.asarray(self.best_candidate)

            if candidate.ndim != 1:
                raise ValueError("best_candidate must be one-dimensional.")

            if not np.issubdtype(candidate.dtype, np.integer):
                raise TypeError(
                    "best_candidate must contain integer diameter indices."
                )

            if np.any(candidate < 0):
                raise ValueError("Candidate indices cannot be negative.")

            object.__setattr__(
                self,
                "best_candidate",
                _readonly_array(candidate),
            )

        if not isinstance(self.convergence, ConvergenceHistory):
            raise TypeError("convergence must be a ConvergenceHistory.")

        if not isinstance(self.diagnostics, dict):
            raise TypeError("diagnostics must be a dictionary.")

        if set(self.diagnostics) - {"algorithm_specific"}:
            raise ValueError(
                "Algorithms may only provide algorithm_specific diagnostics."
            )

        specific = self.diagnostics.get("algorithm_specific", {})

        if not isinstance(specific, dict):
            raise TypeError("algorithm_specific must be a dictionary.")

        object.__setattr__(
            self,
            "convergence",
            self.convergence.snapshot(),
        )
        object.__setattr__(
            self,
            "diagnostics",
            {"algorithm_specific": dict(specific)},
        )
