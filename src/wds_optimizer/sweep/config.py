"""Sweep parameter parsing and application."""

import copy


SWEEP_PARAMETERS = {
    "GA_Pc":         "GA crossover probability (Pc)",
    "GA_Pm":         "GA mutation probability (Pm)",
    "PSO_w":         "PSO inertia weight (w)",
    "PSO_c1":        "PSO cognitive coefficient (c1)",
    "PSO_c2":        "PSO social coefficient (c2)",
    "NS":            "Population / swarm size",
    "MaxGen":        "Max generations / iterations",
    "NumberOfRuns":  "Runs per algorithm",
    "ExperimentSeed": "Experiment seed",
    "Pmin":          "Min pressure (m)",
    "Vmax":          "Max velocity (m/s)",
}


def parse_sweep_values(raw: str) -> list[float]:
    """Split a comma-separated string into unique numeric values."""
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise ValueError("At least one value is required.")
    try:
        values = [float(t) for t in tokens]
    except ValueError:
        raise ValueError("Values must be numbers separated by commas.")
    if len(set(values)) != len(values):
        raise ValueError("Values must be unique.")
    return values


def apply_sweep_parameter(base_cfg, problem, param_name, value, n_runs):
    """Return (cfg, problem, n_runs) with the sweep parameter applied."""
    cfg = copy.deepcopy(base_cfg)
    prob = dict(problem)  # shallow copy — enough for scalar overrides
    n = n_runs

    if param_name == "GA_Pc":
        cfg.GA.Pc = float(value)
    elif param_name == "GA_Pm":
        cfg.GA.Pm = float(value)
    elif param_name == "PSO_w":
        cfg.PSO.w = float(value)
    elif param_name == "PSO_c1":
        cfg.PSO.c1 = float(value)
    elif param_name == "PSO_c2":
        cfg.PSO.c2 = float(value)
    elif param_name == "NS":
        cfg.NS = int(value)
    elif param_name == "MaxGen":
        cfg.MaxGen = int(value)
    elif param_name == "NumberOfRuns":
        n = int(value)
    elif param_name == "Pmin":
        prob["Pmin"] = float(value)
    elif param_name == "Vmax":
        prob["Vmax"] = float(value)
    elif param_name == "ExperimentSeed":
        pass  # seed is passed separately to run_experiment
    else:
        raise ValueError(f"Unknown sweep parameter: {param_name}")

    return cfg, prob, n