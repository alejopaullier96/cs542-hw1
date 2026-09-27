# SYSTEM IMPORTS
from __future__ import annotations
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
from src.dt.clf import DecisionTreeClassifier, Node, LeafNode, InteriorNode
from src.dt.quality.ig import InformationGain
from src.dt.data.c45 import load_play_outside


class TestDTClfPlayOutsideDataset(unittest.TestCase):

    # load the data only once...loading data in each test is unnecessary and slow
    @classmethod
    def setUpClass(cls) -> None:
        header, X, y_gt = load_play_outside()
        cls.header_play_outside = header
        cls.X_play_outside = X
        cls.y_gt_play_outside = y_gt

    def test_tree_structure(self: TestDTClfPlayOutsideDataset) -> None:
        quality_function = InformationGain()

        model = DecisionTreeClassifier(self.header_play_outside, quality_function)
        model.fit(self.X_play_outside, self.y_gt_play_outside)

        self.assertEqual(8, model.num_nodes)

        # first node should be an InteriorNode and focus on the outlook feature
        root: Node = model.root
        self.assertIsInstance(root, InteriorNode)
        self.assertEqual("outlook", root.header[root.feature_idx].name)
        self.assertEqual(3, len(root.children))

        # check the children
        child_a, child_b, child_c = root.children
        self.assertIsInstance(child_a, LeafNode)
        self.assertIsInstance(child_b, InteriorNode)
        self.assertIsInstance(child_c, InteriorNode)

        self.assertEqual(1, child_a.majority_class)
        self.assertEqual("windy?", child_b.header[child_b.feature_idx].name)
        self.assertEqual(2, len(child_b.children))
        self.assertEqual("humidity", child_c.header[child_c.feature_idx].name)
        self.assertEqual(2, len(child_c.children))


        # check child_b's children
        child_ba, child_bb = child_b.children
        self.assertIsInstance(child_ba, LeafNode)
        self.assertIsInstance(child_bb, LeafNode)

        self.assertEqual(1, child_ba.majority_class)
        self.assertEqual(0, child_bb.majority_class)

        # check child_c's children
        child_ca, child_cb = child_c.children
        self.assertIsInstance(child_ca, LeafNode)
        self.assertIsInstance(child_cb, LeafNode)

        self.assertEqual(0, child_ca.majority_class)
        self.assertEqual(1, child_cb.majority_class)


if __name__ == "__main__":
    unittest.main()

