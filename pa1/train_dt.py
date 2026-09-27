# SYSTEM IMPORTS
from __future__ import annotations
import argparse as ap
import numpy as np
import os
import sys
import unittest


# can't do relative file importing since this is an executable
# so need to ensure required base paths are on sys.path
_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from src.dt.clf import DecisionTreeClassifier, RandomForestClassifier
from src.dt.data.c45 import load_play_outside, load_spam, load_volcanoes, load_voting
from src.dt.data.header import Header
from src.dt.quality.ig import InformationGain
from src.dt.quality.gr import GainRatio
from src.dt.model import Model


def load_data(dataset_name: str) -> tuple[Header, np.ndarray, np.ndarray]:
    if dataset_name == "play_outside":
        return load_play_outside()
    elif dataset_name == "spam":
        return load_spam()
    elif dataset_name == "volcanoes":
        return load_volcanoes()
    elif dataset_name == "voting":
        return load_voting()
    else:
        # TODO: add more datasets if you want
        return None, None, None


def make_model(header: Header, args) -> Model:
    quality_function = None
    if args.quality_function == "ig":
        quality_function = InformationGain()
    elif args.quality_function == "gr":
        quality_function = GainRatio()
    else:
        # TODO: add more quality functions if you want
        ...

    model: Model = None
    if args.model_type == "dt":
        model = DecisionTreeClassifier(header, quality_function)
    else:
        model = RandomForestClassifier(header, quality_function) # TODO: pass more args
    return model


def main() -> None:
    parser = ap.ArgumentParser()
    parser.add_argument("dataset", choices=["play_outside", "spam", "volcanoes", "voting"],
                        help="dataset name to load")
    parser.add_argument("quality_function", choices=["ig", "gr"], help="quality function")
    parser.add_argument("model_type", choices=["dt", "rf"], help="what kind of model to use")
    args = parser.parse_args()

    header, X, y_gt = load_data(args.dataset)
    model = make_model(header, args)

    model.fit(X, y_gt)
    print(model)

    # TODO: eval performance?
    Y_hat: np.ndarray = model.predict(X)


if __name__ == "__main__":
    main()

