# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


# PYTHON PROJECT IMPORTS


class Model(ABC):

    def __init__(self: Model) -> None:
        ...

    @abstractmethod
    def fit(self: Model,
            X: np.ndarray,
            Y_gt: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self: Model,
                X: np.ndarray) -> np.ndarray:
        ...

