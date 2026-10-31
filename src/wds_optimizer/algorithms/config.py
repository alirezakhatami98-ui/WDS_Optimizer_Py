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
    Parallel: ParallelConfig = field(default_factory=ParallelConfig)


def build_algorithm_config(NS: int, MaxGen: int) -> AlgorithmConfig:
    return AlgorithmConfig(NS=NS, MaxGen=MaxGen)