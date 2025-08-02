# Place this at the top of your file or in a shared types module.
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class OptimizerState:
    """A structured container for the evolutionary optimizer's state."""
    prev_cross_rate: float
    curr_cross_rate: float
    prev_cross_mom: float
    prev_cross_vel: float
    prev_mut_rate: float
    curr_mut_rate: float
    prev_mut_mom: float
    prev_mut_vel: float
    beta1: float
    beta2: float
    eta: float
    dataset_iteration: int = 0