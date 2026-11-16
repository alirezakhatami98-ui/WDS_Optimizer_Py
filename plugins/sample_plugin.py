"""Sample WDS Optimizer plugin — template for your own algorithms.

Copy this file to your plugins directory:

    Windows:  %USERPROFILE%\\.wds_optimizer\\plugins\\
    Linux:    ~/.wds_optimizer/plugins/
    macOS:    ~/.wds_optimizer/plugins/

Then run ``wds-optimize plugins list`` (CLI) or click "Reload
plugins" in the GUI.  The algorithm will appear in the "Algorithms
to run" dropdown as ``MyRandomSearch``.

This sample implements a trivial random search — for educational
purposes only.  It demonstrates every part of the plugin API:

1. ALGORITHM_META (required) — describes the algorithm and its params
2. run() (required) — the algorithm itself

Read docs/plugins.md for the full guide.
"""
import numpy as np


# Required: algorithm metadata.  Every key here is checked at load
# time by wds_optimizer/plugins.py.  Missing or invalid keys cause
# the plugin to be rejected with a clear error message.
ALGORITHM_META = {
    # Unique short id.  Must not collide with any built-in or another
    # plugin.  Used in the CLI (--algo MyRandomSearch), in the GUI
    # dropdown, and in the API.
    "short_id": "MyRandomSearch",

    # Human-readable name shown in the GUI.
    "display_name": "My Random Search",

    # Free-form category (shown in the GUI).
    "category": "Custom",

    # True if this faithfully reproduces a published paper.
    "paper_faithful": False,

    # One-sentence description shown under the algorithm's name.
    "description": (
        "A trivial random search that draws random designs each "
        "generation.  Useful as a baseline for comparison."
    ),

    # Citations.  Use "-" if none.
    "references": [
        "Solely for demonstration.",
    ],

    # Params shown in the GUI.  For plugins, every param MUST have
    # "maps_to": None.  The value is stored in config.extra[short_id]
    # and read inside run() as:
    #
    #     iters = config.extra["MyRandomSearch"]["my_iters"]
    "params": [
        {
            "key": "my_iters",
            "name": "iters",
            "display_name": "Random designs per generation",
            "type": "int",
            "default": 100,
            "min": 1,
            "max": 10000,
            "step": 10,
            "maps_to": None,
            "description": (
                "How many random designs to try each generation.  "
                "Higher = more exploration, more compute."
            ),
        },
    ],

    # Constraint-handling policy this algorithm uses.  See
    # wds_optimizer/optimization/policies.py.
    "constraint_policy": "deb_prefers",

    # Required for plugins.  Current API version is 1.  If we change
    # the run() signature in a future release, we will bump this and
    # keep backward support for old versions.
    "api_version": 1,
}


def run(problem, config, *, progress_callback=None,
        cancel_callback=None, meta=None):
    """Run the algorithm.  Same contract as built-in algorithms.

    Parameters
    ----------
    problem : dict
        Optimization problem (see wds_optimizer.optimization.problem).
    config : AlgorithmConfig
        Global config.  Plugin params are in
        ``config.extra["MyRandomSearch"]``.
    progress_callback : callable(info: dict) -> None, optional
        Called at intervals with a payload for the GUI's live panel.
    cancel_callback : callable() -> bool, optional
        Called frequently; return True to stop gracefully.
    meta : dict, optional
        If given, fill with NFE counters (same keys as built-ins).

    Returns
    -------
    (score, full_position_m, convergence, feasible)
    """
    from wds_optimizer.optimization.evaluate import evaluate_population
    from wds_optimizer.optimization.policies import get_policy

    rng = (np.random.default_rng(config.Seed)
           if config.Seed is not None
           else np.random.default_rng())

    NP = config.NS
    MaxGen = config.MaxGen
    NVar = len(problem["VariablePipes"])
    ND = len(problem["D"])
    policy = get_policy(config.constraint_policy)

    # Read plugin params.
    my_iters = int(
        config.extra.get("MyRandomSearch", {}).get("my_iters", 100)
    )

    cache = config.Cache if config.Cache is not None else {}
    n_evals = 0

    best_score = float("inf")
    best_viol = float("inf")
    best_sol = None
    best_feasible = False
    conv = np.zeros(MaxGen)

    for G in range(MaxGen):
        if cancel_callback is not None and cancel_callback():
            break

        Pop = rng.integers(0, ND, size=(NVar, NP))
        cost, viol, feas = evaluate_population(Pop, problem, cache)
        n_evals += NP

        for i in range(NP):
            if policy(cost[i], viol[i], best_score, best_viol):
                best_score = float(cost[i])
                best_viol = float(viol[i])
                best_sol = Pop[:, i].copy()
                best_feasible = bool(feas[i])

        conv[G] = best_score if best_feasible else float("inf")

        if progress_callback is not None and (G == 0 or G == MaxGen - 1):
            try:
                progress_callback({
                    "Algorithm": "MyRandomSearch",
                    "Generation": G + 1,
                    "MaxGenerations": MaxGen,
                    "BestCost": float(conv[G]),
                    "Feasible": best_feasible,
                    "Convergence": conv[:G + 1].copy(),
                    "NFE": n_evals,
                    "Restarts": 0,
                    "RestartGens": [],
                })
            except Exception:
                pass

    if meta is not None:
        meta["nfe"] = n_evals
        meta["n_sim"] = n_evals
        meta["nfe_to_best"] = n_evals
        meta["restart_gens"] = []
        meta["epoch_bests"] = []

    if best_sol is None:
        best_sol = np.zeros(NVar, dtype=int)

    full_pos = problem["InitialD"].copy()
    full_pos[problem["VariablePipes"]] = problem["D"][best_sol]
    return best_score, full_pos, conv, best_feasible