import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.dt.data.c45 import load_spam, load_voting, load_volcanoes
from src.dt.clf import DecisionTreeClassifier
from src.dt.quality.ig import InformationGain
from src.dt.quality.gr import GainRatio

def sep():
    print("-"*100)

dataset_names = ["voting", "volcanoes", "spam"]

def experiment_a():
    print("Experiment (a):\n")
    for dataset_name in dataset_names:
        if dataset_name == "spam":
            sep(), print("Spam dataset")
            header, X, y_gt = load_spam()
        elif dataset_name == "voting":
            sep(), print("Voting dataset")
            header, X, y_gt = load_voting()
        elif dataset_name == "volcanoes":
            sep(), print("Volcanoes dataset")
            header, X, y_gt = load_volcanoes()
            
        def max_depth_function(max_depth: int):
            def pre_prune(X, y_gt, available_feature_idxs, depth: int) -> bool:
                return depth > max_depth
            return pre_prune

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345) # validation
        accuracies = []
        i = 0
        for train_idx, test_idx in skf.split(X, y_gt): # split data
            i += 1
            print(f"Fold: {i}")
            print(f"Train size: {len(train_idx)}")
            print(f"Test size: {len(test_idx)}")
            X_train, X_test = X[train_idx], X[test_idx] # train 
            y_train, y_test = y_gt[train_idx], y_gt[test_idx]# test
            model = DecisionTreeClassifier(header, InformationGain()) # model
            model.fit(X_train, y_train, pre_prune_function=max_depth_function(max_depth=1)) # train
            y_hat = model.predict(X_test) # predict
            accuracy = (y_hat == y_test).mean()
            accuracies.append(accuracy)
        print("Mean Accuracy: ", np.mean(accuracies))
        print("Variance: ", np.var(accuracies))

def experiment_b():
    print("Experiment (b):\n")
    loaders = {
        # "voting": load_voting,
        "volcanoes": load_volcanoes,
        "spam": load_spam,
    }

    def max_depth_function(max_depth: int):
        def pre_prune(X, y_gt, available_feature_idxs, depth: int) -> bool:
            return depth > max_depth
        return pre_prune

    depths = [2, 3, 4, 5, 6]
    means_by_dataset = {}
    stds_by_dataset = {}

    for dataset_name, loader in loaders.items():
        sep(), print(f"{dataset_name} dataset")
        header, X, y_gt = loader()
        means = []
        stds = []
        for depth in depths:
            print(f"Depth: {depth}")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
            accuracies = []
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_gt), start=1):
                print(f"Fold: {fold}")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_gt[train_idx], y_gt[test_idx]
                model = DecisionTreeClassifier(header, InformationGain())
                model.fit(X_train, y_train, pre_prune_function=max_depth_function(max_depth=depth))
                y_hat = model.predict(X_test)
                accuracies.append((y_hat == y_test).mean())
            accuracies = np.asarray(accuracies)
            means.append(accuracies.mean())
            stds.append(accuracies.std())
            print("Mean Accuracy: ", means[-1])
            print("Std: ", stds[-1])
        means_by_dataset[dataset_name] = np.asarray(means)
        stds_by_dataset[dataset_name] = np.asarray(stds)

    plt.figure(figsize=(7, 4.5))
    for dataset_name in loaders:
        plt.errorbar(
            depths, means_by_dataset[dataset_name], yerr=stds_by_dataset[dataset_name],
            marker="o", capsize=4, label=dataset_name,
        )
        
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy")
    plt.xticks(depths)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiment_5b.png", dpi=150)
    print("Saved experiment_5b.png")

# experiment_a()
experiment_b()