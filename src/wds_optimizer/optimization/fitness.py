"""Roulette fitness, matching calculateFitness.m."""

import numpy as np


def calculate_fitness(cost, viol):
    return 1.0 / (np.asarray(cost) + 2e6 * np.asarray(viol) ** 2 + 1e-6)