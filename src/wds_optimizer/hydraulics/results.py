"""Internal value objects for hydraulic simulation results."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HydraulicResult:
    """Raw hydraulic output before constraint evaluation.

    Arrays follow the ordering defined by the corresponding optimization
    problem:
    - pressures: junction order
    - velocities: pipe order

    A failed simulation may return None for both arrays.
    Warnings are preserved independently of convergence status.
    """

    pressures: np.ndarray | None
    velocities: np.ndarray | None
    converged: bool
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.converged, (bool, np.bool_)):
            raise TypeError("converged must be a boolean")

        object.__setattr__(self, "converged", bool(self.converged))

        for field_name in ("pressures", "velocities"):
            values = getattr(self, field_name)

            if values is None:
                continue

            array = np.array(values, dtype=float, copy=True)

            if array.ndim != 1:
                raise ValueError(f"{field_name} must be a one-dimensional array")

            array.setflags(write=False)
            object.__setattr__(self, field_name, array)

        if self.converged and (
            self.pressures is None or self.velocities is None
        ):
            raise ValueError(
                "A converged hydraulic result must contain pressures "
                "and velocities"
            )

        if not isinstance(self.warnings, (tuple, list)):
            raise TypeError("warnings must be a tuple or list of strings")

        normalized_warnings = tuple(self.warnings)

        if any(not isinstance(item, str) for item in normalized_warnings):
            raise TypeError("every warning must be a string")

        object.__setattr__(self, "warnings", normalized_warnings)