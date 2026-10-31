"""Roulette and tournament selection."""

import numpy as np

from .compare import is_better_solution


def selection_tournament(cost, viol, n_select, k, rng):
    """Tournament selection with Deb's rules."""
    N = len(cost)
    idx = np.zeros(n_select, dtype=int)
    for s in range(n_select):
        cand = rng.integers(0, N, size=k)
        best = int(cand[0])
        for j in range(1, k):
            cj = int(cand[j])
            if is_better_solution(
                cost[cj], viol[cj], cost[best], viol[best]
            ):
                best = cj
        idx[s] = best
    return idx


def selection_roulette(fitness, n_select, rng):
    """Standard roulette-wheel selection with replacement."""
    prob = fitness / fitness.sum()
    cum = np.cumsum(prob)
    r = rng.random(n_select)
    return np.clip(np.searchsorted(cum, r, side="left"), 0, len(cum) - 1)