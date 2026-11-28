"""Steady-state EPANET simulation via WNTR."""

import os
import tempfile

import numpy as np
from wntr.sim import EpanetSimulator
from wntr.sim.results import ResultsStatus
from wntr.epanet.exceptions import EpanetException

from .results import HydraulicResult


def _run_hydraulic_simulation_result(
    wn,
    full_diameters,
    pipe_names,
    junction_names,
):
    """Run the WNTR/EPANET slow solver and return a structured result.

    Known hydraulic failures are represented by HydraulicResult with
    converged=False. Unexpected software and infrastructure errors propagate.
    """

    # Apply diameters. This intentionally preserves the existing in-place
    # mutation behavior of the hydraulic simulation function.
    for name, diam in zip(pipe_names, full_diameters):
        wn.get_link(name).diameter = float(diam)

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "epanet_run")
        simulator = EpanetSimulator(wn)

        try:
            results = simulator.run_sim(
                file_prefix=prefix,
                convergence_error=False,
            )

        except EpanetException:
            en_data = getattr(simulator, "enData", None)

            warnings = tuple(
                getattr(en_data, "errcodelist", ()) or ()
            )
            error_code = getattr(en_data, "errcode", None)

            # EPANET error 110: hydraulic equations cannot be solved.
            if error_code == 110:
                return HydraulicResult(
                    pressures=None,
                    velocities=None,
                    converged=False,
                    warnings=warnings,
                )

            # Memory errors and unknown EPANET errors are not silently
            # converted into ordinary hydraulic infeasibility.
            raise

        en_data = getattr(simulator, "enData", None)
        warnings = tuple(
            getattr(en_data, "errcodelist", ()) or ()
        )

        # WNTR marks incomplete hydraulic output with ResultsStatus.error.
        if results.error_code == ResultsStatus.error:
            return HydraulicResult(
                pressures=None,
                velocities=None,
                converged=False,
                warnings=warnings,
            )

    pressures = (
        results.node["pressure"]
        .iloc[0][junction_names]
        .to_numpy(dtype=float)
    )

    velocities = np.abs(
        results.link["velocity"]
        .iloc[0][pipe_names]
        .to_numpy(dtype=float)
    )

    if not (
        np.all(np.isfinite(pressures))
        and np.all(np.isfinite(velocities))
    ):
        return HydraulicResult(
            pressures=pressures,
            velocities=velocities,
            converged=False,
            warnings=warnings,
        )

    return HydraulicResult(
        pressures=pressures,
        velocities=velocities,
        converged=True,
        warnings=warnings,
    )


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