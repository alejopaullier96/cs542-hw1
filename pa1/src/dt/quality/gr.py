# SYSTEM IMPORTS
from __future__ import annotations
from collections.abc import Sequence
import numpy as np


# PYTHON PROJECT IMPORTS
from .quality_function import QualityFunction
from .ig import InformationGain


class GainRatio(QualityFunction):
    def __init__(self: GainRatio) -> None:
        ...

    def quality(self: QualityFunction,
                parent_gt: np.ndarray,
                child_gts: Sequence[np.ndarray]) -> float:
        x: InformationGain = InformationGain()
        ig: float = x.quality(parent_gt, child_gts)
        return ig / len(child_gts)

