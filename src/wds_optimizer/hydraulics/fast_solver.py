"""In-memory EPANET evaluation for the optimisation loop.

Why this exists
---------------
``run_hydraulic_simulation`` (WNTR ``EpanetSimulator``) does, for EVERY
candidate design: write the whole network to a .inp file, run EPANET,
write .bin/.rpt files, parse the binary output into many pandas
DataFrames and index them.  The hydraulic solve itself takes well under
a millisecond for a network like Hanoi; the file and pandas round trip is
the real cost (10-50 ms per evaluation).

``FastEpanetSolver`` opens the network ONCE with the EPANET 2.2 toolkit
(bundled with WNTR, no new dependency), then for each design only sets the
pipe diameters, solves, and reads pressures/velocities straight from
memory.  It is validated against the slow path before first use and the
code silently falls back to the slow path if anything disagrees or fails.

Set the environment variable ``WDS_FAST_SOLVER=0`` to force the slow path.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import weakref

import numpy as np
import wntr
from wntr.epanet import util as _eu
from wntr.epanet.exceptions import EpanetException
from wntr.epanet.toolkit import ENepanet

from .simulation import run_hydraulic_simulation

# EPANET 2.2 toolkit parameter codes
_EN_DIAMETER = 0
_EN_VELOCITY = 9
_EN_PRESSURE = 11

# Agreement required with the slow path before the fast path is trusted
_P_TOL_M = 1e-3       # metres of head
_V_TOL_MS = 1e-4      # m/s


class _Project:
    """EPANET project + its temp folder, closed exactly once.

    Kept separate from the solver so a ``weakref.finalize`` can close it
    at garbage collection OR at interpreter exit (Ctrl+C) without holding
    the solver alive.  Order matters on Windows: the project must be
    closed (releasing m.rpt / m.bin) BEFORE the folder is deleted.
    """

    def __init__(self, en, tmpdir):
        self.en = en
        self.tmpdir = tmpdir
        self.lock = threading.RLock()     # a worker thread may be mid-solve
        self.closed = False

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for fn in (self.en.ENcloseH, self.en.ENclose):
                try:
                    fn()
                except Exception:
                    pass
            shutil.rmtree(self.tmpdir, ignore_errors=True)


class FastEpanetSolver:
    """One EPANET project kept open; use from one thread at a time."""

    def __init__(self, wn, pipe_names, junction_names):
        # WNTR logs every EPANET warning (e.g. negative pressure); with
        # ~10^5 evaluations that is pure overhead.
        logging.getLogger("wntr.epanet.toolkit").setLevel(logging.ERROR)

        flow = _eu.FlowUnits[str(wn.options.hydraulic.inpfile_units).upper()]
        # Use WNTR's own conversion constants so results are identical to
        # the slow path (e.g. psi -> m is 0.70344 in WNTR).
        self._d_from_m = _eu.from_si(flow, 1.0, _eu.HydParam.PipeDiameter)
        self._p_to_m = _eu.to_si(flow, 1.0, _eu.HydParam.Pressure)
        self._v_to_ms = _eu.to_si(flow, 1.0, _eu.HydParam.Velocity)

        tmpdir = tempfile.mkdtemp(prefix="wds_epanet_")
        try:
            inp = os.path.join(tmpdir, "model.inp")
            wntr.network.write_inpfile(wn, inp)           # once
            en = ENepanet()
            en.ENopen(inp, os.path.join(tmpdir, "m.rpt"),
                      os.path.join(tmpdir, "m.bin"))
            en.ENopenH()
            self._link_idx = [en.ENgetlinkindex(n) for n in pipe_names]
            self._node_idx = [en.ENgetnodeindex(n) for n in junction_names]
        except BaseException:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

        self._proj = _Project(en, tmpdir)
        # runs on garbage collection and at interpreter exit
        self._finalizer = weakref.finalize(self, self._proj.close)

    # ------------------------------------------------------------------
    def solve(self, full_diameters_m):
        """Return (pressures_m, velocities_ms, converged) like the slow path."""
        proj = self._proj
        with proj.lock:
            if proj.closed:
                raise RuntimeError("EPANET solver is closed")
            en = proj.en
            try:
                for idx, d in zip(self._link_idx, full_diameters_m):
                    en.ENsetlinkvalue(idx, _EN_DIAMETER,
                                      float(d) * self._d_from_m)
                en.ENinitH(10)      # re-initialise flows, do not save
                en.ENrunH()         # first time step only (== iloc[0])
                P = np.fromiter(
                    (en.ENgetnodevalue(i, _EN_PRESSURE)
                     for i in self._node_idx),
                    dtype=float, count=len(self._node_idx),
                ) * self._p_to_m
                V = np.abs(np.fromiter(
                    (en.ENgetlinkvalue(i, _EN_VELOCITY)
                     for i in self._link_idx),
                    dtype=float, count=len(self._link_idx),
                )) * self._v_to_ms
            except EpanetException:
                self._recover()
                return None, None, False
            finally:
                en.errcodelist.clear()  # WNTR appends every warning

        ok = bool(np.all(np.isfinite(P)) and np.all(np.isfinite(V)))
        return P, V, ok

    def _recover(self):
        try:
            self._proj.en.ENcloseH()
            self._proj.en.ENopenH()
        except Exception:
            pass

    def close(self):
        self._finalizer()       # idempotent; closes the project once


# ----------------------------------------------------------------------
def _validated_solver(problem):
    """Build a solver and check it against the slow path; None if unsafe."""
    wn = problem["wn"]
    pipes = problem["pipe_names"]
    juncs = problem["junction_names"]
    solver = FastEpanetSolver(wn, pipes, juncs)

    initial = np.asarray(problem["InitialD"], dtype=float)
    largest = initial.copy()
    largest[problem["VariablePipes"]] = float(np.max(problem["D"]))

    for design in (initial, largest):
        P_ref, V_ref, ok_ref = run_hydraulic_simulation(
            wn, design, pipes, juncs
        )
        if not ok_ref:
            continue                      # cannot compare on this design
        P, V, ok = solver.solve(design)
        if (ok and np.allclose(P, P_ref, atol=_P_TOL_M, rtol=0)
                and np.allclose(V, V_ref, atol=_V_TOL_MS, rtol=0)):
            return solver
        solver.close()
        raise RuntimeError(
            "fast solver disagrees with the reference simulation "
            f"(max |dP| = {np.max(np.abs(P - P_ref)):.3g} m)"
        )
    solver.close()
    raise RuntimeError("no solvable design available for validation")


def get_solver(problem):
    """Lazily create/validate the solver and cache it on ``problem``.

    Returns ``None`` when the fast path is disabled or unsafe; callers
    then use ``run_hydraulic_simulation``.
    """
    cached = problem.get("_solver")
    if cached is not None:
        return cached or None
    if os.environ.get("WDS_FAST_SOLVER", "1") == "0":
        problem["_solver"] = False
        return None
    try:
        solver = _validated_solver(problem)
        problem["_solver"] = solver
        return solver
    except Exception as e:       # fall back, never break the optimisation
        print(f"[fast_solver] disabled, using slow path: "
              f"{type(e).__name__}: {e}")
        problem["_solver"] = False
        return None