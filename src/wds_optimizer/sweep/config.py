"""Sweep parameter parsing and application."""

import copy


SWEEP_PARAMETERS = {
    "GA_Pc":         "GA crossover probability (Pc)",
    "GA_Pm":         "GA mutation probability (Pm)",
    "PSO_w":         "PSO inertia weight (w)",
    "PSO_c1":        "PSO cognitive coefficient (c1)",
    "PSO_c2":        "PSO social coefficient (c2)",
    "DE_F":          "DE differential weight (F)",
    "DE_CR":         "DE crossover rate (CR)",
    "NS":            "Population / swarm size",
    "MaxGen":        "Max generations / iterations",
    "NumberOfRuns":  "Runs per algorithm",
    "ExperimentSeed": "Experiment seed",
    "RestartMax":    "Max restarts per run (0 = restart off)",
    "RestartPatience": "Restart patience in generations (turns restart on)",
    "RestartPerturb": "Restart perturbation rate (turns on 'perturb' restart)",
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
    cfg = copy.deepcopy(base_cfg)
    prob = dict(problem)
    n = n_runs
    units = problem.get("Units", {})

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
    elif param_name == "DE_F":
        cfg.DE.F = float(value)
    elif param_name == "DE_CR":
        cfg.DE.CR = float(value)
    elif param_name == "NS":
        cfg.NS = int(value)
    elif param_name == "MaxGen":
        cfg.MaxGen = int(value)
    elif param_name == "RestartMax":
        # 0 gives a no-restart baseline inside the same sweep
        cfg.Restart.MaxRestarts = int(value)
        cfg.Restart.Enabled = int(value) > 0
    elif param_name == "RestartPatience":
        cfg.Restart.Patience = int(value)
        cfg.Restart.Enabled = True
    elif param_name == "RestartPerturb":
        cfg.Restart.PerturbRate = float(value)
        cfg.Restart.Mode = "perturb"
        cfg.Restart.Enabled = True
    elif param_name == "NumberOfRuns":
        n = int(value)
    elif param_name == "Pmin":
        # value is in network units; convert to SI
        factor = 0.70307 if units.get("PressureUnit") == "psi" else 1.0
        prob["Pmin"] = float(value) * factor
    elif param_name == "Vmax":
        factor = 0.3048 if units.get("VelocityUnit") == "ft/s" else 1.0
        prob["Vmax"] = float(value) * factor
    elif param_name == "ExperimentSeed":
        pass
    else:
        raise ValueError(f"Unknown sweep parameter: {param_name}")

    return cfg, prob, n