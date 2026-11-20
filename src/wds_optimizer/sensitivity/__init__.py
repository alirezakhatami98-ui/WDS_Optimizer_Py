"""Sensitivity analysis package.

Re-exports the public surface of the renamed `sweep` package with
sensitivity-analysis naming.  The old package (`wds_optimizer.sweep`)
is a deprecation shim that re-exports everything here.
"""
from .config import (  # noqa: F401
    LEGACY_MAP,
    SENSITIVITY_PARAMETERS,
    SWEEP_PARAMETERS,        # alias
    build_sensitivity_parameters,
    build_sweep_parameters,  # alias
    normalize_param_name,
    parse_sensitivity_values,
    parse_sweep_values,      # alias
)
from .runner import (  # noqa: F401
    SingleSweepPoint,
    SingleSweepResult,
    SweepPointResult,        # alias
    SweepResult,             # alias
    run_parameter_sweep,     # alias
    run_single_sweep,
)