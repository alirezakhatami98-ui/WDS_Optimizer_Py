"""Loader for D.txt and Cost.txt."""

import numpy as np

_DIAM_TO_M = {"in": 0.0254, "mm": 1e-3, "m": 1.0, "ft": 0.3048}


def load_optimization_data(d_file, cost_file, diam_file_unit="in"):
    """Load diameter options and unit costs, sorted by diameter.

    Parameters
    ----------
    d_file, cost_file : str or Path
    diam_file_unit : {'in', 'mm', 'm', 'ft'}
        Unit in which D.txt is written (default 'in').

    Returns
    -------
    Din : np.ndarray
        Diameters as written in the file (in ``diam_file_unit``).
    Cost : np.ndarray
        Unit cost per length, same order as Din.
    D_m : np.ndarray
        Diameters converted to meters (WNTR internal unit).
    """
    if diam_file_unit not in _DIAM_TO_M:
        raise ValueError(f"Unsupported diameter file unit: {diam_file_unit!r}")

    Din = np.loadtxt(d_file, dtype=float).ravel()
    Cost = np.loadtxt(cost_file, dtype=float).ravel()
    if Din.size != Cost.size:
        raise ValueError(
            f"D.txt has {Din.size} entries but Cost.txt has {Cost.size}."
        )

    order = np.argsort(Din)
    if not np.array_equal(order, np.arange(Din.size)):
        import warnings
        warnings.warn("Diameter options reordered (they were not ascending).")
    Din = Din[order]
    Cost = Cost[order]

    D_m = Din * _DIAM_TO_M[diam_file_unit]

    if np.any(D_m <= 0):
        raise ValueError("All diameters must be positive.")

    return Din, Cost, D_m