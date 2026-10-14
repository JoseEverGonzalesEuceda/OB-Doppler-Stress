from dataclasses import dataclass, asdict
from typing import Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    """Fixed configuration used in the manuscript experiments."""

    seeds: Tuple[int, ...] = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
    test_size: float = 0.25
    augmentation_vmax_kms: float = 100.0
    velocities_kms: Tuple[int, ...] = tuple(range(-200, 201, 20))
    control_velocities_kms: Tuple[int, ...] = (-200, -100, 0, 100, 200)
    fill_value: float = 1.0
    n_jobs: int = -1

    def to_dict(self):
        return asdict(self)
