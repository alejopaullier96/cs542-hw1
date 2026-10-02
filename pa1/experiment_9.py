import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.dt.data.c45 import load_spam, load_voting, load_volcanoes
from src.dt.clf import RandomForestClassifier
from src.dt.quality.ig import InformationGain
from src.dt.quality.gr import GainRatio

def sep():
    print("-"*100)

dataset_names = ["voting", "volcanoes", "spam"]

def experiment_a():
    print("Experiment (a):\n")
    np.random.seed(12345)
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

        for mcc_prune in [False, True]:
            print(f"Post-pruning: {mcc_prune}")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345) # validation
            accuracies = []
            i = 0
            for train_idx, test_idx in skf.split(X, y_gt): # split data
                i += 1
                print(f"Fold: {i}")
                print(f"Train size: {len(train_idx)}")
                print(f"Test size: {len(test_idx)}")
                X_train, X_test = X[train_idx], X[test_idx] # train
                y_train, y_test = y_gt[train_idx], y_gt[test_idx] # test
                model = RandomForestClassifier(header, InformationGain(), num_trees=100) # model
                model.fit(X_train, y_train, mcc_prune=mcc_prune, alpha=0.5) # train
                y_hat = model.predict(X_test) # predict
                accuracy = (y_hat == y_test).mean()
                accuracies.append(accuracy)
            print("Mean Accuracy: ", np.mean(accuracies))
            print("Variance: ", np.var(accuracies))

def experiment_b():
    print("Experiment (b):\n")
    np.random.seed(12345)
    loaders = {
        # "voting": load_voting,
        "volcanoes": load_volcanoes,
        "spam": load_spam,
    }

    num_trees_list = [20, 40, 60, 80, 100]
    means_by_dataset = {}
    stds_by_dataset = {}

    for dataset_name, loader in loaders.items():
        sep(), print(f"{dataset_name} dataset")
        header, X, y_gt = loader()
        means = []
        stds = []
        for num_trees in num_trees_list:
            print(f"Number of trees: {num_trees}")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
            accuracies = []
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_gt), start=1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_gt[train_idx], y_gt[test_idx]
                model = RandomForestClassifier(header, InformationGain(), num_trees=num_trees)
                model.fit(X_train, y_train)
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
            num_trees_list, means_by_dataset[dataset_name], yerr=stds_by_dataset[dataset_name],
            marker="o", capsize=4, label=dataset_name,
        )

    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.xticks(num_trees_list)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiment_9b.png", dpi=150)
    print("Saved experiment_9b.png")


def experiment_c():
    print("Experiment (c):\n")
    np.random.seed(12345)
    loaders = {
        # "voting": load_voting,
        "volcanoes": load_volcanoes,
        "spam": load_spam,
    }
    quality_funcs = {
        "information gain": InformationGain(),
        "gain ratio": GainRatio(),
    }

    num_trees_list = [20, 60, 100]
    means_by_key = {}
    vars_by_key = {}

    for dataset_name, loader in loaders.items():
        sep(), print(f"{dataset_name} dataset")
        header, X, y_gt = loader()
        for quality_name, quality_func in quality_funcs.items():
            print(quality_name)
            means = []
            variances = []
            for num_trees in num_trees_list:
                print(f"Number of trees: {num_trees}")
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
                accuracies = []
                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_gt), start=1):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y_gt[train_idx], y_gt[test_idx]
                    model = RandomForestClassifier(header, quality_func, num_trees=num_trees)
                    model.fit(X_train, y_train)
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
        axes[0].plot(num_trees_list, means_by_key[key], marker="o", label=key)
        axes[1].plot(num_trees_list, vars_by_key[key], marker="o", label=key)
    axes[0].set_xlabel("Number of trees")
    axes[0].set_ylabel("Mean accuracy")
    axes[0].set_xticks(num_trees_list)
    axes[0].legend()
    axes[1].set_xlabel("Number of trees")
    axes[1].set_ylabel("Variance")
    axes[1].set_xticks(num_trees_list)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("experiment_9c.png", dpi=150)
    print("Saved experiment_9c.png")

def experiment_d():
    print("Experiment (d):\n")
    np.random.seed(12345)
    loaders = {
        # "voting": load_voting,
        "volcanoes": load_volcanoes,
        "spam": load_spam,
    }

    num_trees_list = [100, 200]
    full_by_dataset = {}
    cv_means_by_dataset = {}
    cv_stds_by_dataset = {}
    for dataset_name, loader in loaders.items():
        sep(), print(f"{dataset_name} dataset")
        header, X, y_gt = loader()
        full_accs = []
        cv_means = []
        cv_stds = []
        for num_trees in num_trees_list:
            print(f"Number of trees: {num_trees}")
            model = RandomForestClassifier(header, InformationGain(), num_trees=num_trees)
            model.fit(X, y_gt)
            full_accuracy = (model.predict(X) == y_gt).mean()

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
            accuracies = []
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_gt), start=1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_gt[train_idx], y_gt[test_idx]
                cv_model = RandomForestClassifier(header, InformationGain(), num_trees=num_trees)
                cv_model.fit(X_train, y_train)
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
        plt.plot(num_trees_list, full_by_dataset[dataset_name], marker="o", label=f"{dataset_name} (full data)")
        plt.errorbar(
            num_trees_list, cv_means_by_dataset[dataset_name], yerr=cv_stds_by_dataset[dataset_name],
            marker="s", capsize=4, label=f"{dataset_name} (5-fold CV)",
        )
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.xticks(num_trees_list)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiment_9d.png", dpi=150)
    print("Saved experiment_9d.png")


### cd pa1
### /opt/anaconda3/bin/python -m experiment_9
experiment_a()
# experiment_b()
# experiment_c()
# experiment_d()
