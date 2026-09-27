# SYSTEM IMPORTS
from __future__ import annotations
from collections.abc import Sequence
import numpy as np


# PYTHON PROJECT IMPORTS
from .quality_function import QualityFunction


class Entropy(object):
    def __init__(self: Entropy) -> None:
        ...

    def quality(self: Entropy,
                y_gt: np.ndarray) -> float:
        _, counts = np.unique(y_gt, return_counts=True)
        probs = counts / counts.sum()
        return -(probs * np.log2(probs)).sum()


class InformationGain(QualityFunction):
    def __init__(self: InformationGain) -> None:
        ...

    def quality(self: QualityFunction,
                parent_gt: np.ndarray,
                child_gts: Sequence[np.ndarray]) -> float:
        # print(child_gts)
        # for child_y in child_gts:
        #     print(child_gts)

        e: Entropy = Entropy()
        H_y: float = e.quality(parent_gt)
        H_children: Sequence[float] = [e.quality(child_y) for child_y in child_gts]
        child_probs: Sequence[float] = [float(child_y.shape[0])/parent_gt.shape[0] for child_y in child_gts]
        return H_y - sum(pr_child * H_child for pr_child, H_child in zip(child_probs, H_children))

