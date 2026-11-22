"""Policy inputs and built-in dispatch baselines."""

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class Decision:
    observation: np.ndarray
    mask: np.ndarray
    prior: np.ndarray
    features: np.ndarray
    candidates: Mapping[int, tuple[int, int, int]]
    rng: np.random.Generator


class Policy(Protocol):
    def __call__(self, decision: Decision) -> int: ...


def random_valid(decision: Decision) -> int:
    return int(decision.rng.choice(np.flatnonzero(decision.mask)))
