"""Steady-state EPANET simulation via WNTR."""

import os
import tempfile
import numpy as np
from wntr.sim import EpanetSimulator


def run_hydraulic_simulation(wn, full_diameters, pipe_names, junction_names):
    # Apply diameters
    for name, diam in zip(pipe_names, full_diameters):
        wn.get_link(name).diameter = float(diam)

    # Run EPANET in an isolated temp folder (avoids temp.inp/.bin/.rpt
    # polluting the project directory)
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "epanet_run")
        try:
            results = EpanetSimulator(wn).run_sim(file_prefix=prefix)
        except Exception as e:
            print(f"[run_hydraulic_simulation] EPANET error: {type(e).__name__}: {e}")
            return None, None, False

    Pj = results.node["pressure"].iloc[0][junction_names].to_numpy(dtype=float)
    Vpipes = np.abs(
        results.link["velocity"].iloc[0][pipe_names].to_numpy(dtype=float)
    )
    if not (np.all(np.isfinite(Pj)) and np.all(np.isfinite(Vpipes))):
        return Pj, Vpipes, False
    return Pj, Vpipes, True