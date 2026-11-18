"""Binary classification with numeric splits and categorical ID3."""

import csv

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from .id3 import ID3
from .reporting import draw_confusion, save_figure, write_metrics


FEATURES = ("profession", "education", "house_loan", "car_loan", "married", "child", "revenue")
LABEL = "approve"
REVENUE_CUTS = (10_000, 15_000, 20_000, 28_000, 40_000)
TREE_SETTINGS = (
    {"criterion": "gini", "max_depth": 3, "min_samples_split": 4, "min_samples_leaf": 1},
    {"criterion": "entropy", "max_depth": 4, "min_samples_split": 6, "min_samples_leaf": 1},
    {"criterion": "entropy", "max_depth": 5, "min_samples_split": 5, "min_samples_leaf": 2},
)
ID3_DEPTHS = (3, 5, 10, None)


def read_bank(path):
    features, labels = [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set((*FEATURES, LABEL)) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                values = [int(row[column]) for column in FEATURES[:-1]]
                revenue = float(row["revenue"])
                approval = int(row[LABEL])
                if not np.isfinite(revenue) or approval not in (0, 1):
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path.name}, line {reader.line_num}: expected numeric features and a 0/1 label."
                ) from error
            features.append([*values, revenue])
            labels.append(approval)
    if not labels:
        raise ValueError(f"{path.name} contains no samples.")
    return np.array(features, dtype=float), np.array(labels, dtype=int)


def demo_bank(seed):
    rng = np.random.default_rng(seed)
    profession = rng.integers(1, 6, 1_000)
    education = rng.integers(1, 4, 1_000)
    loans_and_household = rng.integers(0, 2, (1_000, 4))
    revenue = rng.integers(5_000, 52_000, 1_000)
    features = np.column_stack((profession, education, loans_and_household, revenue))
    threshold = 20_000 + 7_000 * loans_and_household[:, 0] - 2_000 * education
    labels = (revenue + rng.normal(0, 3_000, len(revenue)) >= threshold).astype(int)
    return train_test_split(features, labels, test_size=0.2, random_state=seed, stratify=labels)


def discretize(features):
    discrete = features.astype(int, copy=True)
    # A threshold belongs to its lower bin, as in right-closed revenue intervals.
    discrete[:, -1] = np.searchsorted(REVENUE_CUTS, features[:, -1], side="left")
    return discrete


def binary_scores(actual, predicted):
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(count) for count in matrix.ravel())
    return {
        "accuracy": (tp + tn) / matrix.sum().item(),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "confusion_matrix": matrix.tolist(),
    }


def run(args):
    if args.demo:
        full_x, test_x, full_y, test_y = demo_bank(args.seed)
    else:
        full_x, full_y = read_bank(args.data_dir / "bank_train.csv")
        test_x, test_y = read_bank(args.data_dir / "bank_test.csv")

    if len(np.unique(full_y)) != 2:
        raise ValueError("The bank training set must include both labels.")
    train_x, val_x, train_y, val_y = train_test_split(
        full_x, full_y, test_size=0.2, random_state=args.seed, stratify=full_y
    )

    sklearn_validation = []
    for settings in TREE_SETTINGS:
        tree = DecisionTreeClassifier(random_state=args.seed, **settings).fit(train_x, train_y)
        scores = binary_scores(val_y, tree.predict(val_x))
        sklearn_validation.append({"settings": settings, **scores})
    best_settings = max(sklearn_validation, key=lambda row: row["f1"])["settings"]
    final_tree = DecisionTreeClassifier(random_state=args.seed, **best_settings).fit(full_x, full_y)

    train_discrete, val_discrete = discretize(train_x), discretize(val_x)
    id3_validation = []
    for depth in ID3_DEPTHS:
        model = ID3(depth).fit(train_discrete, train_y)
        id3_validation.append({"max_depth": depth, **binary_scores(val_y, model.predict(val_discrete))})
    best_depth = max(id3_validation, key=lambda row: row["f1"])["max_depth"]
    full_discrete, test_discrete = discretize(full_x), discretize(test_x)
    final_id3 = ID3(best_depth).fit(full_discrete, full_y)
    unpruned_id3 = ID3().fit(full_discrete, full_y)

    test_scores = {
        "scikit-learn tree": binary_scores(test_y, final_tree.predict(test_x)),
        "validation-selected ID3": binary_scores(test_y, final_id3.predict(test_discrete)),
        "unpruned ID3": binary_scores(test_y, unpruned_id3.predict(test_discrete)),
    }
    report = {
        "dataset": "synthetic" if args.demo else "local CSV",
        "seed": args.seed,
        "split_sizes": {"train": len(train_y), "validation": len(val_y), "test": len(test_y)},
        "positive_fraction": {
            "train": float(train_y.mean()), "validation": float(val_y.mean()), "test": float(test_y.mean())
        },
        "revenue_cuts": REVENUE_CUTS,
        "sklearn_validation": sklearn_validation,
        "id3_validation": id3_validation,
        "selected_sklearn_settings": best_settings,
        "selected_id3_depth": best_depth,
        "test": test_scores,
    }
    write_metrics(args.out, report)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    for axis, (title, scores) in zip(axes, test_scores.items()):
        draw_confusion(axis, np.array(scores["confusion_matrix"]), [0, 1], title)
        print(f"{title}: accuracy={scores['accuracy']:.4f}, F1={scores['f1']:.4f}")
    save_figure(fig, args.out / "confusion.png")

    fig, axis = plt.subplots(figsize=(15, 7))
    plot_tree(
        final_tree, ax=axis, feature_names=FEATURES, class_names=["decline", "approve"],
        filled=True, rounded=True, fontsize=8,
    )
    save_figure(fig, args.out / "decision-tree.png")
