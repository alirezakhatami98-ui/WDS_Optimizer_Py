"""WDS Optimizer - Water distribution system design optimization.

Public surface
--------------
    >>> from wds_optimizer import OptimizationRequest, run_optimization
    >>> result = run_optimization(request)
    >>> result.to_json()

The GUI (``streamlit_app``) and CLI (``wds-optimize``) both call
through ``run_optimization``, so behaviour is identical across all
three entry points.
"""
from wds_optimizer.api import (
    OptimizationRequest,
    OptimizationResult,
    build_problem,
    load_network_isolated,
    run_optimization,
)

__version__ = "0.2.0"
__all__ = [
    "__version__",
    "OptimizationRequest",
    "OptimizationResult",
    "build_problem",
    "load_network_isolated",
    "run_optimization",
]