"""Configuration objects, matching buildAlgorithmConfig.m."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GAConfig:
    Pc: float = 0.8
    Pm: float = 0.03
    Selection: str = "roulette"   # "roulette" | "tournament"
    TournamentSize: int = 3


@dataclass
class PSOConfig:
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5


@dataclass
class DEConfig:
    # Note on CR: pipe-sizing cost is separable (a sum over pipes) while the
    # hydraulic constraints couple them.  Low CR (few genes changed per
    # trial) suited this structure better than the textbook 0.9 in our
    # tests on two synthetic networks - tune it with a DE_CR sweep.
    F: float = 0.5                    # differential weight
    CR: float = 0.3                   # binomial crossover rate (see note)
    # "rand/1/bin" | "best/1/bin" | "current-to-best/1/bin"
    Strategy: str = "rand/1/bin"

@dataclass
class SADEConfig:
    Fmin: float = 0.1
    Fmax: float = 0.9
    CRmin: float = 0.1
    CRmax: float = 0.9
    UseConvergence: bool = True
    Tol: float = 1e-6
    # MATLAB safeguard: stop after this many generations without any
    # improvement in the current epoch. 0 disables it. The paper relies
    # purely on Cv < Tol; 300 matches the reference MATLAB implementation.
    StagnationMax: int = 300

@dataclass
class RestartConfig:
    """Restart-on-stagnation (shared by GA and PSO).

    When the best solution of the *current epoch* has not improved for
    ``Patience`` consecutive generations/iterations, the population is
    re-initialised at random.  The best solution over all epochs is always
    kept, so a restart can never make the final result worse.  The total
    number of generations (``MaxGen``) is NOT extended: restarts
    redistribute the existing budget.
    """
    Enabled: bool = False
    Patience: int = 30            # generations without improvement
    MaxRestarts: int = 3          # per run
    # "random"  : fresh random population (optionally seeded with the best)
    # "perturb" : copies of the best design with a fraction of the pipes
    #             re-randomised (a "cataclysmic" restart, as in CHC)
    Mode: str = "perturb"
    KeepBest: bool = False        # random mode only
    PerturbRate: float = 0.3      # perturb mode: share of genes re-drawn


@dataclass
class ParallelConfig:
    Enabled: bool = False
    NumWorkers: int = 2


@dataclass
class AlgorithmConfig:
    NS: int = 100
    MaxGen: int = 500
    Seed: Optional[int] = None
    Cache: Optional[dict] = None
    GA: GAConfig = field(default_factory=GAConfig)
    PSO: PSOConfig = field(default_factory=PSOConfig)
    DE: DEConfig = field(default_factory=DEConfig)
    SADE: SADEConfig = field(default_factory=SADEConfig)
    Restart: RestartConfig = field(default_factory=RestartConfig)
    Parallel: ParallelConfig = field(default_factory=ParallelConfig)


def build_algorithm_config(NS: int, MaxGen: int) -> AlgorithmConfig:
    return AlgorithmConfig(NS=NS, MaxGen=MaxGen)