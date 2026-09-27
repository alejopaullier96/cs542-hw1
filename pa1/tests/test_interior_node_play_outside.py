# SYSTEM IMPORTS
from __future__ import annotations
from collections.abc import Sequence, Set
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
from src.dt.clf import InteriorNode
from src.dt.quality.ig import InformationGain
from src.dt.data.c45 import load_play_outside


class TestInteriorNodePlayOutsideDataset(unittest.TestCase):

    # load the data only once...loading data in each test is unnecessary and slow
    @classmethod
    def setUpClass(cls) -> None:
        header, X, y_gt = load_play_outside()
        cls.header_play_outside = header
        cls.X_play_outside = X
        cls.y_gt_play_outside = y_gt

    def test_first_node_ig(self: TestInteriorNodePlayOutsideDataset) -> None:

        quality_function = InformationGain()
        available_feature_idxs: Set[int] = set(range(len(self.header_play_outside)))

        node = InteriorNode(self.header_play_outside,                       # data header
                            quality_function,                               # quality function
                            self.X_play_outside, self.y_gt_play_outside,    # training data
                            available_feature_idxs)                         # set of features this node will consider

        # node should have chosen outlook as the first feature
        self.assertEqual("outlook", self.header_play_outside[node.feature_idx].name)
        self.assertAlmostEqual(0.247, node.feature_quality, places=3)
        self.assertEqual(3, len(node.feature_split_values))
        self.assertEqual(len(self.header_play_outside[node.feature_idx].domain()), len(node.feature_split_values))

        # children aren't assigned when we create the node, let the wrapper class 'DecisionTreeClassifier'
        # build the tree
        self.assertEqual(0, len(node.children))

        # we should be able to "execute" this node on the training data though
        self.assertEqual(3, len(node.get_child_datasets()))

        for split_value, (X_child, y_gt_child) in zip(node.feature_split_values, node.get_child_datasets()):
            # check that this dataset corresponds to this child
            X_col: np.ndarray = X_child[:, node.feature_idx]
            unique_feature_values = np.unique(X_col)
            self.assertEqual(1, unique_feature_values.shape[0])
            self.assertEqual(split_value, unique_feature_values[0])


if __name__ == "__main__":
    unittest.main()

