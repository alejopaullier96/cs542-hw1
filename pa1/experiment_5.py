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


def experiment_c():
    print("Experiment (c):\n")
    loaders = {
        # "voting": load_voting,
        "volcanoes": load_volcanoes,
        # "spam": load_spam,
    }
    quality_funcs = {
        "information gain": InformationGain(),
        "gain ratio": GainRatio(),
    }

    def max_depth_function(max_depth: int):
        def pre_prune(X, y_gt, available_feature_idxs, depth: int) -> bool:
            return depth > max_depth
        return pre_prune

    depths = [2, 4, 6]
    means_by_key = {}
    vars_by_key = {}

    for dataset_name, loader in loaders.items():
        sep(), print(f"{dataset_name} dataset")
        header, X, y_gt = loader()
        for quality_name, quality_func in quality_funcs.items():
            print(quality_name)
            means = []
            variances = []
            for depth in depths:
                print(f"Depth: {depth}")
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
                accuracies = []
                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_gt), start=1):
                    print(f"Fold: {fold}")
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y_gt[train_idx], y_gt[test_idx]
                    model = DecisionTreeClassifier(header, quality_func)
                    model.fit(X_train, y_train, pre_prune_function=max_depth_function(max_depth=depth))
                    y_hat = model.predict(X_test)
                    accuracies.append((y_hat == y_test).mean())
                accuracies = np.asarray(accuracies)
                means.append(accuracies.mean())
                variances.append(accuracies.var())
                print("Mean Accuracy: ", means[-1])
                print("Variance: ", variances[-1])
            key = f"{dataset_name} ({quality_name})"
            means_by_key[key] = np.asarray(means)
            vars_by_key[key] = np.asarray(variances)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for key in means_by_key:
        axes[0].plot(depths, means_by_key[key], marker="o", label=key)
        axes[1].plot(depths, vars_by_key[key], marker="o", label=key)
    axes[0].set_xlabel("Max depth")
    axes[0].set_ylabel("Mean accuracy")
    axes[0].set_xticks(depths)
    axes[0].legend()
    axes[1].set_xlabel("Max depth")
    axes[1].set_ylabel("Variance")
    axes[1].set_xticks(depths)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("experiment_5c.png", dpi=150)
    print("Saved experiment_5c.png")

def experiment_d():
    print("Experiment (d):\n")
    loaders = {
        # "voting": load_voting,
        "volcanoes": load_volcanoes,
        "spam": load_spam,
    }

    def max_depth_function(max_depth: int):
        def pre_prune(X, y_gt, available_feature_idxs, depth: int) -> bool:
            return depth > max_depth
        return pre_prune

    depths = [1, 2]
    full_by_dataset = {}
    cv_means_by_dataset = {}
    cv_stds_by_dataset = {}
    for dataset_name, loader in loaders.items():
        sep(), print(f"{dataset_name} dataset")
        header, X, y_gt = loader()
        full_accs = []
        cv_means = []
        cv_stds = []
        for depth in depths:
            print(f"Depth: {depth}")
            model = DecisionTreeClassifier(header, InformationGain())
            model.fit(X, y_gt, pre_prune_function=max_depth_function(max_depth=depth))
            full_accuracy = (model.predict(X) == y_gt).mean()

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
            accuracies = []
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_gt), start=1):
                print(f"Fold: {fold}")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_gt[train_idx], y_gt[test_idx]
                cv_model = DecisionTreeClassifier(header, InformationGain())
                cv_model.fit(X_train, y_train, pre_prune_function=max_depth_function(max_depth=depth))
                accuracies.append((cv_model.predict(X_test) == y_test).mean())
            accuracies = np.asarray(accuracies)
            full_accs.append(full_accuracy)
            cv_means.append(accuracies.mean())
            cv_stds.append(accuracies.std())
            print("Full-data accuracy: ", full_accs[-1])
            print("CV mean accuracy: ", cv_means[-1])
            print("CV variance: ", accuracies.var())
        full_by_dataset[dataset_name] = np.asarray(full_accs)
        cv_means_by_dataset[dataset_name] = np.asarray(cv_means)
        cv_stds_by_dataset[dataset_name] = np.asarray(cv_stds)

    plt.figure(figsize=(7, 4.5))
    for dataset_name in loaders:
        plt.plot(depths, full_by_dataset[dataset_name], marker="o", label=f"{dataset_name} (full data)")
        plt.errorbar(
            depths, cv_means_by_dataset[dataset_name], yerr=cv_stds_by_dataset[dataset_name],
            marker="s", capsize=4, label=f"{dataset_name} (5-fold CV)",
        )
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy")
    plt.xticks(depths)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiment_5d.png", dpi=150)
    print("Saved experiment_5d.png")


### cd pa1 /opt/anaconda3/bin/python -m src.experiment_5
experiment_a()
# experiment_b()
# experiment_c()
# experiment_d()