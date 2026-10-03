# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Sequence, Set
from typing import Tuple, Union
import numpy as np


# PYTHON PROJECT IMPORTS
from .quality.gr import GainRatio
from .data.header import Feature, FeatureType, ContinuousFeature, DiscreteFeature, Header
from .quality.quality_function import QualityFunction
from .model import Model


class Node(ABC):
    def __init__(self: Node,
                 header: Header,
                 quality_function: QualityFunction,
                 X: np.ndarray,
                 y_gt: np.ndarray) -> None:
        self.header: Header = header
        self.quality_function = quality_function

        # some stats
        self.num_samples: int = X.shape[0]
        self.unique_classes, self.unique_class_counts = np.unique(y_gt, return_counts=True)

        # majority class
        self.majority_class = self.get_majority_class(X, y_gt)

        # children of this Node
        self.children = list()

    def get_majority_class(self: Node,
                           X: np.ndarray,
                           y_gt: np.ndarray) -> int:
        unique_classes, counts = np.unique(y_gt, return_counts=True)
        counts_argmax = np.argmax(counts)
        return unique_classes[counts_argmax]

    @abstractmethod
    def predict(self: Node,
                x: np.ndarray) -> Union[int, Node]:
        ...

    @abstractmethod
    def get_child_datasets(self: Node,
                           X: np.ndarray = None,
                           y_gt: np.ndarray = None) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        ...

    @abstractmethod
    def is_leaf(self: Node) -> bool:
        ...

    @abstractmethod
    def _to_str(self: Node,
                depth: int) -> str:
        ...

    def __str__(self: Node) -> str:
        return self._to_str(0)

    def __repr__(self: Node) -> str:
        return str(self)


class LeafNode(Node):
    def __init__(self: Node,
                 header: Header,
                 quality_function: QualityFunction,
                 X: np.ndarray,
                 y_gt: np.ndarray) -> None:
        super().__init__(header, quality_function, X, y_gt)

    def predict(self: LeafNode,
                x: np.ndarray) -> Union[int, Node]:
        return self.majority_class

    def get_child_datasets(self: LeafNode,
                           X: np.ndarray = None,
                           y_gt: np.ndarray = None) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        return None

    def is_leaf(self: LeafNode) -> bool:
        return True

    def _to_str(self: LeafNode,
                depth: int) -> str:
        return ("\t" * depth) + f"LeafNode(majority_class={self.majority_class})"


class InteriorNode(Node):
    def __init__(self: InteriorNode,
                 header: Header,
                 quality_function: QualityFunction,
                 X: np.ndarray,
                 y_gt: np.ndarray,
                 available_feature_idxs: Set[int]) -> None:
        super().__init__(header, quality_function, X, y_gt)

        self.X = X
        self.y_gt = y_gt

        # these will be set by self._pick_best_feature
        self.feature_quality: float = None
        self.feature_idx: int = None
        self.feature_split_values: Sequence[float] = None
        self.feature_type: FeatureType = None

        # the feature idxs that children can use
        self.child_feature_idxs = set(available_feature_idxs)

        # call this to choose which feature this InteriorNode will focus on
        self._pick_best_feature(X, y_gt, available_feature_idxs)

        # remember to delete the chosen feature from child datasets if we choose a discrete feature!
        if self.feature_type == FeatureType.DISCRETE:
            self.child_feature_idxs.remove(self.feature_idx)

    def _pick_best_feature(self: InteriorNode,
                           X: np.ndarray,
                           y_gt: np.ndarray,
                           available_feature_idxs: Set[int]) -> int:
        best_feature_quality = -np.inf
        best_feature_idx: int = -1
        best_feature_split_values: np.ndarray = None

        # argmax the features (sorted so ties always go to the smallest feature_idx)
        for feature_idx in sorted(available_feature_idxs):
            feature_quality, feature_split_values = self._eval_feature(X, y_gt, feature_idx)

            # argmax and settle ties with the smallest feature_idx
            if (feature_quality > best_feature_quality) or (feature_quality == best_feature_quality and
                                                            feature_idx < best_feature_idx):
                best_feature_quality = feature_quality
                best_feature_idx = feature_idx
                best_feature_split_values = feature_split_values

        # set the fields
        self.feature_quality = best_feature_quality
        self.feature_idx = best_feature_idx
        self.feature_split_values = best_feature_split_values
        if best_feature_idx < 0:
            return
        self.feature_type = self.header[self.feature_idx].type

    def _eval_feature(self: InteriorNode,
                      X: np.ndarray,
                      y_gt: np.ndarray,
                      feature_idx: int) -> tuple[float, Sequence[float]]:
        feature_quality: float = None
        feature_split_values: Sequence[float] = list()

        X_col: np.ndarray = X[:, feature_idx]
        feature_type = self.header[feature_idx].type

        # TODO: evaluate the quality of this feature!
        #       don't forget that you need to consider two cases:
        #           - the feature is DISCRETE (eval this feature by considering the sole split)
        #           - the feature is CONTINUOUS (eval this feature by choosing the best "version" of this feature)
        if feature_type == FeatureType.DISCRETE:
            feature_quality, feature_split_values = self._eval_discrete_feature(X, y_gt, feature_idx)
        elif feature_type == FeatureType.CONTINUOUS:
            feature_quality, feature_split_values = self._eval_continuous_feature(X, y_gt, feature_idx)

        return feature_quality, feature_split_values

    def _eval_discrete_feature(
            self: InteriorNode,
            X: np.ndarray,
            y_gt: np.ndarray,
            feature_idx: int
        ) -> tuple[float, Sequence[float]]:
        """
        Evaluate the quality of a discrete feature.
        :param X: one column example: np.array([1, 2, 2, 4, 3, 2, 1, 1, 3, 4])
        :param y_gt: one target example: np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
        """
        X_col = X[:, feature_idx] # take the feature column
        feature_split_values = list(np.unique(X_col))# get unique values of the feature column 
        child_gts = [] # ground truth values for each unique value of the feature column
        for value in feature_split_values:
            child_gts.append(y_gt[X_col == value])
        feature_quality = float(self.quality_function.quality(y_gt, child_gts))
        return feature_quality, feature_split_values

    def _entropy_from_counts(self: InteriorNode, counts: np.ndarray) -> float:

        counts = counts[counts > 0]
        probs = counts / counts.sum()
        return float(-(probs * np.log2(probs)).sum()) # the closer to 0 the better or more pure

    def _eval_continuous_feature(
        self: InteriorNode,
        X: np.ndarray,
        y_gt: np.ndarray,
        feature_idx: int) -> tuple[float, Sequence[float]]:
        """
        Evaluate the quality of a continuous feature.
        :param X: one column example: np.array([1.1, 2.3, 3.2, 4.8, 5.7])
        :param y_gt: one target example: np.array([0, 0, 1, 1, 0])
        """
        X_col = X[:, feature_idx] # take the feature column
        order = np.argsort(X_col) # sort order
        sorted_x = X_col[order] # sort the feature
        sorted_y = y_gt[order].astype(int, copy=False)# sorted y
        n = sorted_y.shape[0] # number of examples, eg 5
        n_classes = int(sorted_y.max()) + 1 # number of classes, eg 2
        total_counts = np.bincount(sorted_y, minlength=n_classes) # count the number of examples for each class, eg [3 2]
        parent_entropy = self._entropy_from_counts(total_counts) # entropy root, eg 0.971
        left_counts = np.zeros(n_classes, dtype=int) # [0 0]
        best_quality = -np.inf # initialize with the worst possible case
        best_threshold = None # no initial threshold
        i = 0
        while i < n - 1: # iterate over the examples
            value = sorted_x[i]
            group_counts = np.zeros(n_classes, dtype=int)
            while i < n and sorted_x[i] == value: # move every sample with this value to the 'left' side
                group_counts[sorted_y[i]] += 1
                left_counts[sorted_y[i]] += 1
                i += 1
            if i == n:# reached the end examples
                break
            
            next_counts = np.zeros(n_classes, dtype=int) # a cut is useful unless both adjacent value-groups are pure and share one class
            j = i
            next_value = sorted_x[i]
            while j < n and sorted_x[j] == next_value:
                next_counts[sorted_y[j]] += 1
                j += 1
            group_classes = np.flatnonzero(group_counts)
            next_classes = np.flatnonzero(next_counts)
            both_pure_same = (group_classes.size == 1 and next_classes.size == 1 and group_classes[0] == next_classes[0])
            if both_pure_same:
                continue
            n_left = i # number of examples on the left side
            left_y = sorted_y[:n_left]
            right_y = sorted_y[n_left:]

            quality = float(self.quality_function.quality(y_gt, [left_y, right_y]))
            if quality > best_quality:
                best_quality = quality
                best_threshold = float((value + next_value) / 2.0)

        if best_threshold is None:
            return -np.inf, []
        return best_quality, [best_threshold]
    
    def _get_continuous_feature_thresholds(
        self: InteriorNode,
        X_col: np.ndarray,
        y_gt: np.ndarray
    ) -> Sequence[float]:
        """
        Get the potential thresholds for a continuous feature.
        :param X_col: one column example: np.array([1.0, 2.0, 2.0, 3.0])
        :param y_gt: one target example: np.array([0, 0, 1, 1])
        """
        thresholds: Sequence[float] = list()

        # TODO: calculate the potential thresholds this continuous feature
        #       could choose! Remember the algorithm from lecture!
        order = np.argsort(X_col) # sort the feature column, this returns the indices
        sorted_X_col = X_col[order] # sort the feature column
        sorted_y_gt = y_gt[order] # sort the target column according to the feature column
        
        # Example, unique_vals = [1.0, 2.0, 3.0] and start_idxs = [0, 1, 3]
        unique_vals, start_idxs = np.unique(sorted_X_col, return_index=True) # get unique values and their indices
        label_groups = np.split(sorted_y_gt, start_idxs[1:]) # split the target column into groups
        label_sets = []
        for group in label_groups:
            label_set = set(np.unique(group).tolist())
            label_sets.append(label_set)# [{0}, {0, 1}, {1}] for example

        for i in range(len(unique_vals) - 1): # iterate over the unique values
            left_labels = label_sets[i]
            right_labels = label_sets[i + 1]
            c1 = left_labels != right_labels # the class changes if the neighbors have different labels
            c2 = len(left_labels) > 1 #the left labels have more than one class
            c3 = len(right_labels) > 1 #the right labels have more than one class
            if c1 or c2 or c3:
                val = (unique_vals[i] + unique_vals[i + 1]) / 2.0 # average two consecutive values
                thresholds.append(float(val))

        return thresholds

    def predict(self: InteriorNode,
                x: np.ndarray) -> Union[int, Node]:

        """
        Predict the class of a sample.
        :param x: example:one sample with two features (discrete and continuous): np.array([1,1.5])
        """
        child: Node = None

        # TODO: choose the child node the sample 'x' would flow to
        value = x[self.feature_idx] # the value of the feature this node focuses on
        if self.feature_type == FeatureType.DISCRETE: # discrete column
            for i, split_value in enumerate(self.feature_split_values): # children follow the split value order
                if value == split_value:
                    return self.children[i]
        elif self.feature_type == FeatureType.CONTINUOUS: # continuous column
            threshold = self.feature_split_values[0]# there is only one threshold
            if value <= threshold:
                return self.children[0]
            else:
                return self.children[1]
        return LeafNode(self.header, self.quality_function, self.X, self.y_gt) # unseen discrete values majority class

    def get_child_datasets(self: InteriorNode,
                           X: np.ndarray = None,
                           y_gt: np.ndarray = None) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        if X is None:
            X = self.X
        if y_gt is None:
            y_gt = self.y_gt

        child_datasets: Sequence[tuple[np.ndarray, np.ndarray]] = list()

        # get the column of data that this interior node focuses on
        X_col: np.ndarray = X[:, self.feature_idx]
        feature_type = self.header[self.feature_idx].type

        # split (X, y_gt) according to this node.
        #       don't forget that you need to consider two cases:
        #           - the feature is DISCRETE: generate one dataset per feature value
        #           - the feature is CONTINUOUS: make a binary split
        if feature_type == FeatureType.DISCRETE: # discrete column
            for value in self.feature_split_values: # iterate over the feature split values
                X_child = X[X_col == value] # get the child dataset for the feature value
                y_child = y_gt[X_col == value] # get the child ground truth for the feature value
                child_datasets.append((X_child, y_child))
        elif feature_type == FeatureType.CONTINUOUS: # continuous column
            for threshold in self.feature_split_values:
                X_child_left = X[X_col <= threshold] # get the child dataset for the left side
                y_child_left = y_gt[X_col <= threshold] # get the child ground truth for the left side
                child_datasets.append((X_child_left, y_child_left))

                X_child_right = X[X_col > threshold] # get the child dataset for the right side
                y_child_right = y_gt[X_col > threshold] # get the child ground truth for the right side
                child_datasets.append((X_child_right, y_child_right))

        return child_datasets

    def is_leaf(self: InteriorNode) -> bool:
        return False

    # helpful for printing a decision tree
    def _to_str(self: InteriorNode,
                depth: int) -> str:
        feature_name: str = self.header[self.feature_idx].name
        split_values: str = self.feature_split_values

        preamble = "InteriorNode("
        children = "children=["

        return ("\t" * depth) + f"{preamble}feature={feature_name}, split_values={split_values})" +\
            "\n" + "\n".join([c._to_str(depth+1) for c in self.children])


class DecisionTreeClassifier(Model):
    def __init__(self: DecisionTreeClassifier,
                 header: Header,
                 quality_function: QualityFunction,
                 available_feature_idxs: Set[int] = None) -> None:
        self.header: Header = header
        self.quality_function = quality_function
        self.available_feature_idxs: Set[int] = set(available_feature_idxs) \
                                                if available_feature_idxs is not None \
                                                else set(range(len(self.header)))

        self.num_nodes: int = 0
        self.root: Node = None

    def _build(self: DecisionTreeClassifier,
               X: np.ndarray,
               y_gt: np.ndarray,
               available_feature_idxs: Set[int],
               depth: int,
               pre_prune_function: Callable[[np.ndarray, np.ndarray, Set[int], int], bool] = None) -> Node:
        node: Node = None

        # TODO: build the tree! This method needs to turn a dataset into a node.
        #       If that node is an InteriorNode we need to get the child datasets and
        #       turn them into nodes too!
        #
        #       The argument 'pre_prune_function' will not be 'None' if the caller requests pre-pruning
        #       to occur. Pre-pruning is something like setting a "max depth" or "minimum samples", you don't
        #       have to care because this is a function pointer. If this argument is not 'None' and you call
        #       it, it will return 'true' when you should clip this path and generate a LeafNode (even if
        #       an InteriorNode would be chosen otherwise).
        #
        #       you should expect this to be called like this:
        #           pre_prune_function(X, y_gt, available_feature_idxs, depth)
        from collections import deque
        pending = deque() # queue
        pending.append((X, y_gt, set(available_feature_idxs), depth, None))# we start with a queue with all the data
        root = None # at first the root is none
        self.num_nodes = 0
        while pending: 
            X_node, y_node, feature_idxs, depth_node, parent = pending.popleft() # get the next node from the queue
            prune = pre_prune_function is not None and pre_prune_function(X_node, y_node, feature_idxs, depth_node)
            pure = len(np.unique(y_node)) <= 1 # all the labels in the leaf are one class
            no_features = len(feature_idxs) == 0 # no features left
            if prune or pure or no_features:
                node: Node = LeafNode(self.header, self.quality_function, X_node, y_node) 
            else:
                node = InteriorNode(self.header, self.quality_function, X_node, y_node, feature_idxs)
                if node.feature_idx is None or node.feature_idx < 0 or not node.feature_split_values:
                    node = LeafNode(self.header, self.quality_function, X_node, y_node)
                else:
                    for X_child, y_child in node.get_child_datasets(X_node, y_node):# add child nodes to the queue
                        # Drop empty branches so a missing discrete value does not create a node
                        if X_child.shape[0] == 0:
                            continue
                        new_depth = depth_node + 1 
                        pending.append((X_child, y_child, set(node.child_feature_idxs), new_depth, node))

            self.num_nodes += 1 
            if parent is None:
                root = node
            else:
                parent.children.append(node)

        return root

    def _leaf_errors(self: DecisionTreeClassifier, node: Node) -> int:
        samples = node.num_samples # number of samples at this node
        samples_majority = np.max(node.unique_class_counts) # most common class
        errors = int(samples - samples_majority) # number of samples that are not the majority class
        return errors

    def _subtree_stats(
            self: DecisionTreeClassifier,
            node: Node,
            pruned: Set[Node],
            stats: dict
        ) -> tuple[int, int]:
        """
        Count the training errors and number of leaves of the subtree rooted at 'node'
        Nodes inside 'pruned' are treated as if they were leaves
        Every interior node visited is saved in 'stats' as dict with 'node': (errors, num_leaves)
        """
        if node.is_leaf() or node in pruned: # this node acts like a leaf
            errors = self._leaf_errors(node) # number of errors at this node
            num_leaves = 1 # number of leaves at this node
            return errors, num_leaves
        errors, num_leaves = 0, 0
        for child in node.children: # iterate over the childern
            child_errors, child_leaves = self._subtree_stats(child, pruned, stats) # get the errors and leaves of each child
            errors += child_errors # add errors to the total errors
            num_leaves += child_leaves # add leaves to the total leaves
        stats[node] = (errors, num_leaves) # save the errors and leaves of this node
        return errors, num_leaves

    def _apply_pruning(
            self: DecisionTreeClassifier,
            node: Node,
            pruned: Set[Node]
        ) -> Node:
        # replace every node in 'pruned' with a real LeafNode
        if node.is_leaf(): # if the node is a leaf, return it
            return node
        if node in pruned: # if the node is in the pruned set, return a leaf node
            return LeafNode(self.header, self.quality_function, node.X, node.y_gt)
        node.children = [self._apply_pruning(child, pruned) for child in node.children] # recursively apply pruning to the children
        return node
        
    def fit(self: DecisionTreeClassifier,
            X: np.ndarray,
            y_gt: np.ndarray,
            pre_prune_function: Callable[[np.ndarray, np.ndarray, Set[int], int], bool] = None,
            mcc_prune: bool = False,
            alpha: float = 0.5) -> None:
        # alpha is the hyperparameter coefficient for the pessimistic error estimate

        # build the tree
        self.root = self._build(X, y_gt, self.available_feature_idxs, 1, pre_prune_function=pre_prune_function)

        # TODO: implement minimum-cost-complexity pruning algorithm!
        if mcc_prune:
            pruned = set() # interior nodes turned into leaves
            sequence = [] # tuple with the cost and the pruned set
            while True:
                stats = dict()
                errors, num_leaves = self._subtree_stats(self.root, pruned, stats) # get the errors of the root
                cost = errors + alpha * num_leaves # e'(T)
                sequence.append((cost, set(pruned)))
                if len(stats) == 0: # no interior nodes left, T is a single leaf
                    break

                best_node, best_score = None, np.inf# inicializar con el peor caso
                for node, (sub_errors, sub_leaves) in stats.items(): # iterar sobre los nodos internos
                    # sub_errors is the number of mistakes made by the leaves under the node
                    # errors is the number of mistakes by the entire tree
                    # leaf errors is the number of mistakes if the node was made into a leaf
                    new_errors = errors - sub_errors + self._leaf_errors(node)
                    new_num_leaves = num_leaves - sub_leaves + 1
                    new_cost = new_errors + alpha * new_num_leaves # e'(prune(T, node))
                    if sub_leaves == 1: # node has a single leaf below it, pruning changes nothing
                        score = -np.inf
                    else:
                        score = (new_cost - cost) / (num_leaves - new_num_leaves)
                    if score < best_score:
                        best_node, best_score = node, score
                pruned.add(best_node)# add the best node to the pruned set

            best_cost, best_pruned = np.inf, None
            for cost, pruned_nodes in sequence:
                if cost <= best_cost:
                    best_cost, best_pruned = cost, pruned_nodes
            self.root = self._apply_pruning(self.root, best_pruned) # apply pruning to the root
            self.num_nodes = self._count_nodes(self.root) # pruning replaces subtrees with leaves, so recount

    def _count_nodes(self: DecisionTreeClassifier,
                     node: Node) -> int:
        if node.is_leaf():
            return 1
        return 1 + sum(self._count_nodes(child) for child in node.children)

    def _predict_sample(self: DecisionTreeClassifier,
                        x: np.ndarray) -> int:
        node: Node = self.root
        while not node.is_leaf():
            node = node.predict(x)

        # node should be a leaf node
        return node.predict(x)

    def predict(self: DecisionTreeClassifier,
                X: np.ndarray) -> np.ndarray:

        # pre-allocate space for each prediction
        num_samples: int = X.shape[0]
        y_hat = np.zeros(num_samples)

        # one sample at a time
        for sample_idx in range(num_samples):
            y_hat[sample_idx] = self._predict_sample(X[sample_idx, :])
        return y_hat

    def __str__(self: DecisionTreeClassifier) -> str:
        return str(self.root)

    def __repr__(self: DecisionTreeClassifier) -> str:
        return str(self)



"""
    A bagging model. This model will train a bunch of decision trees and then have each of its internal
    decision trees vote during inference. The class that gets the most votes wins (settling ties arbitrarily).
"""
class RandomForestClassifier(Model):
    def __init__(self: RandomForestClassifier,
                 header: Header,
                 quality_function: QualityFunction,
                 max_num_features: int = None,
                 num_trees: int = 100) -> None:
        self.header = header
        self.quality_function = quality_function
        self.max_num_features = min(len(self.header), max_num_features) \
                                if max_num_features is not None else len(self.header)
        self.num_trees = num_trees
        self.trees: Sequence[Model] = list()

    def _bootstrap_sample(self: RandomForestClassifier,
                          X: np.ndarray,
                          y_gt: np.ndarray,
                          num_samples) -> tuple[np.ndarray, np.ndarray]:
        # draw num_samples uniformly and independently at random
        sample_idxs: np.ndarray = np.random.choice(X.shape[0], size=num_samples, replace=True)

        # note that this **applies** the indices to the data 
        return X[sample_idxs, :], y_gt[sample_idxs]

    def _sample_features(self: RandomForestClassifier,
                         max_features: int) -> Set[int]:
        # choose 'max_features' from the available features
        num_features: int = np.random.choice(max_features)

        # note that this returns a **set** of feature indices
        return set(np.random.choice(len(self.header), size=num_features, replace=False))

    def fit(self: RandomForestClassifier,
            X: np.ndarray,
            y_gt: np.ndarray,
            pre_prune_function: Callable[[np.ndarray, np.ndarray, Set[int], int], bool] = None,
            mcc_prune: bool = False,
            alpha: float = 0.5) -> None:
        """
            train 'num_trees' number of DecisionTreeClassifiers
            each tree gets its own dataset:
                - the samples in that dataset are bootstrap sampled. This means that you sample
                  uniformly and independently at random a bunch of samples from X (the same sample
                  can appear multiple times within the generated dataset)
                - the features for that specific dataset are restricted to be only a subset of the
                  original features. The features a specific tree is allowed to use are chosen
                  uniformly at random (not independently though: no duplicate features!). There
                  is a parameter called 'max_num_features' which gives an upper bound on how
                  many features a single tree can see.
        """

        # TODO: build the forest! 
        self.trees = list()
        num_samples = X.shape[0] # number of rows
        # num_samples = min(X.shape[0], 3000)
        for i in range(self.num_trees):
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y_gt, num_samples) # bootstrap sample with replacement
            feature_idxs = self._sample_features(self.max_num_features) # sample features, eg {0, 1, 2}
            tree = DecisionTreeClassifier(self.header, self.quality_function, available_feature_idxs=feature_idxs) # build the tree
            tree.fit(X_bootstrap, y_bootstrap, pre_prune_function=pre_prune_function, mcc_prune=mcc_prune, alpha=alpha) # train the tree
            self.trees.append(tree) # add the tree to the forest


    def predict(self: RandomForestClassifier,
                X: np.ndarray) -> np.ndarray:
        # TODO: ask each tree to predict 'X' and then implement majority voting!
        num_samples = X.shape[0] # number of rows
        preds = []
        for tree in self.trees: # predict X with each tree
            preds.append(tree.predict(X)) # shape (num_samples,)
        all_preds = np.stack(preds, axis=1) # stack the predictions into a 2D array, shape (num_samples, num_trees)
        y_hat = np.zeros(num_samples) # placeholder to store preds
        for idx in range(num_samples):
            # for each row count the votes for each class
            values, counts = np.unique(all_preds[idx, :], return_counts=True)
            y_hat[idx] = values[np.argmax(counts)] # pred is the class with most votes

        return y_hat
        

    # helpful for printing the forest
    def __str__(self: RandomForestClassifier) -> str:
        return f"RandomForestClassifier(header={self.header}, self.quality_function={type(self.quality_function).__name__}, max_num_features={self.max_num_features}, num_trees={self.num_trees})"

    def __repr__(self: RandomForestClassifier) -> str:
        return str(self)

