"""Steady-state EPANET simulation via WNTR."""

import numpy as np
from wntr.sim import EpanetSimulator


def run_hydraulic_simulation(wn, full_diameters, pipe_names, junction_names):
    """Apply diameters (meters) to every pipe and run EPANET.

    Parameters
    ----------
    wn : WaterNetworkModel
        Reused across calls. Diameters are overwritten each time.
    full_diameters : array-like of length ``n_pipes``
        Full diameter vector in meters, ordered as ``pipe_names``.
    pipe_names : list[str]
    junction_names : list[str]

    Returns
    -------
    Pj : np.ndarray | None
        Junction pressures (m).
    Vpipes : np.ndarray | None
        Absolute pipe velocities (m/s).
    Converged : bool
    """
    # Apply diameters
    for name, diam in zip(pipe_names, full_diameters):
        wn.get_link(name).diameter = float(diam)

    # Run EPANET
    try:
        results = EpanetSimulator(wn).run_sim()
    except Exception as e:
        # Do NOT swallow silently — print so the caller can see the real cause.
        print(f"[run_hydraulic_simulation] EPANET error: {type(e).__name__}: {e}")
        return None, None, False

    # Steady-state (duration=0) → one row
    Pj = results.node["pressure"].iloc[0][junction_names].to_numpy(dtype=float)
    Vpipes = np.abs(
        results.link["velocity"].iloc[0][pipe_names].to_numpy(dtype=float)
    )

    if not (np.all(np.isfinite(Pj)) and np.all(np.isfinite(Vpipes))):
        return Pj, Vpipes, False
    return Pj, Vpipes, True