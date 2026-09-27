# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np


# PYTHON PROJECT IMPORTS


class QualityFunction(ABC):

    @abstractmethod
    def quality(self: QualityFunction,
                parent_gt: np.ndarray,
                child_gts: Sequence[np.ndarray]) -> float:
        ...
