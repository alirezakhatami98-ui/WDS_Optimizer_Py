"""Phase 1 smoke test: run GA and PSO on the 2Loops network."""

import sys
from pathlib import Path

import numpy as np
import wntr

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)

from wds_optimizer.algorithms.config import build_algorithm_config
from wds_optimizer.algorithms.ga import run_ga
from wds_optimizer.algorithms.pso import run_pso
from wds_optimizer.data.loaders import load_optimization_data
from wds_optimizer.hydraulics.simulation import run_hydraulic_simulation
from wds_optimizer.hydraulics.units import detect_network_units
from wds_optimizer.optimization.problem import build_optimization_problem


def main():
    net_dir = Path(__file__).resolve().parent.parent / "networks" / "2loops"
    inp = str(net_dir / "2Loops.inp")
    d_file = str(net_dir / "D.txt")
    cost_file = str(net_dir / "Cost.txt")

    # ---- 1. Load network ----
    wn = wntr.network.WaterNetworkModel(inp)

    units = detect_network_units(wn)
    print("Units:", units)

    Din, Cost, D_m = load_optimization_data(
        d_file, cost_file, diam_file_unit="in"
    )
    print(f"Diameters (in) : {Din.tolist()}")
    print(f"Diameters (m)  : {D_m.round(4).tolist()}")
    print(f"Costs          : {Cost.tolist()}")

    # ---- 2. Extract static network info ----
    pipe_names = list(wn.pipe_name_list)
    junction_names = list(wn.junction_name_list)
    lengths = np.array([wn.get_link(n).length for n in pipe_names])
    initial_D = np.array([wn.get_link(n).diameter for n in pipe_names])

    # ---- 3. Sanity: evaluate initial .inp ----
    Pj, Vp, ok = run_hydraulic_simulation(
        wn, initial_D, pipe_names, junction_names
    )
    if Pj is None:
        print("ERROR: initial hydraulic simulation failed (see message above).")
        return
    print("\nInitial state:")
    print(f"  pressures (m)   : "
          f"{dict(zip(junction_names, Pj.round(3)))}")
    print(f"  velocities (m/s): "
          f"{dict(zip(pipe_names, Vp.round(3)))}")
    print(f"  Converged       : {ok}")

    # ---- 4. Problem ----
    problem = build_optimization_problem(
        wn=wn, Din=Din, Cost=Cost, D_m=D_m,
        full_lengths=lengths, initial_D=initial_D,
        fixed_pipes=[],
        variable_pipes=list(range(len(pipe_names))),
        Pmin=30.0, Vmax=2.5,
        pipe_names=pipe_names, junction_names=junction_names,
        units=units,
    )

    # ---- 5. GA ----
    print("\n=== GA (NS=30, MaxGen=60, tournament) ===")
    cfg = build_algorithm_config(NS=30, MaxGen=60)
    cfg.GA.Selection = "tournament"
    cfg.Seed = 42

    score_ga, pos_ga, conv_ga, feas_ga = run_ga(problem, cfg)
    print(f"  best cost : {score_ga:.2f}")
    print(f"  feasible  : {feas_ga}")
    print(f"  diameters : "
          f"{(pos_ga[problem['VariablePipes']] / 0.0254).round(1).tolist()} in")

    # ---- 6. PSO ----
    print("\n=== PSO (NS=30, MaxGen=60) ===")
    cfg_pso = build_algorithm_config(NS=30, MaxGen=60)
    cfg_pso.Seed = 42

    score_pso, pos_pso, conv_pso, feas_pso = run_pso(problem, cfg_pso)
    print(f"  best cost : {score_pso:.2f}")
    print(f"  feasible  : {feas_pso}")
    print(f"  diameters : "
          f"{(pos_pso[problem['VariablePipes']] / 0.0254).round(1).tolist()} in")

    # ---- 7. Verify final hydraulics ----
    print("\nFinal hydraulic check (GA solution):")
    Pj2, Vp2, ok2 = run_hydraulic_simulation(
        wn, pos_ga, pipe_names, junction_names
    )
    if Pj2 is None:
        print("ERROR: final hydraulic simulation failed.")
        return
    print(f"  pressures (m)   : "
          f"{dict(zip(junction_names, Pj2.round(3)))}")
    print(f"  min pressure    : {Pj2.min():.3f} m  (Pmin = 30)")
    print(f"  max velocity    : {Vp2.max():.3f} m/s (Vmax = 2.5)")
    print(f"  Converged       : {ok2}")


if __name__ == "__main__":
    main()