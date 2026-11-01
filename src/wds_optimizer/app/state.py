"""Session-state management for the Streamlit app."""

import streamlit as st


_KEYS = {
    "inp_path": None, "d_path": None, "cost_path": None,
    "diam_unit": "in",
    "ns": 50, "max_gen": 100,
    "pmin": 30.0, "vmax": 2.5,
    "fixed_pipes": "",
    "ga_pc": 0.8, "ga_pm": 0.03, "ga_sel": "tournament",
    "pso_w": 0.7, "pso_c1": 1.5, "pso_c2": 1.5,
    "n_runs": 3, "exp_seed": 1,
    "experiment": None,
    "best_algo": None, "best_run_index": None,
    "best_position_m": None, "best_score": None, "best_feasible": False,
    "node_pressures": None, "pipe_velocities": None,
    "hydraulics_converged": False,
    "problem_snapshot": None,
}


def init_state():
    for key, default in _KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_experiment():
    for key in (
        "experiment", "best_algo", "best_run_index",
        "best_position_m", "best_score", "best_feasible",
        "node_pressures", "pipe_velocities",
        "hydraulics_converged", "problem_snapshot",
    ):
        st.session_state[key] = _KEYS[key]