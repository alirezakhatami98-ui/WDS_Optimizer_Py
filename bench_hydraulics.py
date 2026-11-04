"""Compare the reference EPANET path with the in-memory solver.

Usage (from the project root, same environment as the app):
    python bench_hydraulics.py path/to/Hanoi.inp [n_evals]

Prints the time per evaluation of both paths and the maximum difference
in pressures/velocities (should be ~1e-5 or smaller).
"""
import sys
import time

import numpy as np
import wntr

from wds_optimizer.hydraulics.fast_solver import FastEpanetSolver
from wds_optimizer.hydraulics.simulation import run_hydraulic_simulation


def main(inp_path, n=50):
    wn = wntr.network.WaterNetworkModel(inp_path)
    pipes = list(wn.pipe_name_list)
    juncs = list(wn.junction_name_list)
    base = np.array([wn.get_link(p).diameter for p in pipes])  # metres
    rng = np.random.default_rng(0)
    designs = [base * rng.choice([0.6, 0.8, 1.0, 1.25, 1.5], size=len(pipes))
               for _ in range(n)]

    t0 = time.perf_counter()
    ref = [run_hydraulic_simulation(wn, d, pipes, juncs) for d in designs]
    t_ref = (time.perf_counter() - t0) / n

    solver = FastEpanetSolver(wn, pipes, juncs)
    t0 = time.perf_counter()
    fast = [solver.solve(d) for d in designs]
    t_fast = (time.perf_counter() - t0) / n
    solver.close()

    dp = max(np.abs(a[0] - b[0]).max() for a, b in zip(ref, fast)
             if a[2] and b[2])
    dv = max(np.abs(a[1] - b[1]).max() for a, b in zip(ref, fast)
             if a[2] and b[2])
    print(f"reference path : {t_ref * 1000:8.2f} ms / evaluation")
    print(f"in-memory path : {t_fast * 1000:8.2f} ms / evaluation")
    print(f"speed-up       : {t_ref / t_fast:8.0f} x")
    print(f"max |dP| = {dp:.2e} m,  max |dV| = {dv:.2e} m/s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)
