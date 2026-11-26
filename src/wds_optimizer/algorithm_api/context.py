
"""Execution context for Algorithm Plugin API v2."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


class OptimizationCancelled(Exception):
    """Raised when an optimization run is cancelled."""

    def __init__(self, partial_result: Any = None) -> None:
        super().__init__("Optimization was cancelled.")
        self.partial_result = partial_result


ProgressCallback = Callable[[dict[str, Any]], None]
CancellationCheck = Callable[[], bool]


class AlgorithmContext:
    """Public execution context provided to API v2 algorithms."""

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        progress_callback: ProgressCallback | None = None,
        cancellation_check: CancellationCheck | None = None,
    ) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")

        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None.")

        if cancellation_check is not None and not callable(cancellation_check):
            raise TypeError("cancellation_check must be callable or None.")

        self._rng = rng
        self._progress_callback = progress_callback
        self._cancellation_check = cancellation_check

    @property
    def rng(self) -> np.random.Generator:
        """Return the run-specific random number generator."""
        return self._rng

    def report_progress(
        self,
        *,
        generation: int,
        max_generations: int,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Report algorithm progress to the runner, if configured."""

        if (
            isinstance(generation, (bool, np.bool_))
            or not isinstance(generation, (int, np.integer))
        ):
            raise TypeError("generation must be an integer.")

        if (
            isinstance(max_generations, (bool, np.bool_))
            or not isinstance(max_generations, (int, np.integer))
        ):
            raise TypeError("max_generations must be an integer.")

        if max_generations < 1:
            raise ValueError("max_generations must be positive.")

        if not 1 <= generation <= max_generations:
            raise ValueError(
                "generation must be between 1 and max_generations."
            )

        if metrics is not None and not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping or None.")

        if self._progress_callback is None:
            return

        payload = {
            "generation": int(generation),
            "max_generations": int(max_generations),
            "metrics": dict(metrics) if metrics is not None else {},
        }

        self._progress_callback(payload)

    def check_cancelled(self) -> None:
        """Raise OptimizationCancelled when cancellation is requested."""

        if (
            self._cancellation_check is not None
            and self._cancellation_check()
        ):
            raise OptimizationCancelled()
