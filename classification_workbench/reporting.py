"""Common output helpers and multiclass metrics."""

import json

from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def multiclass_scores(actual, predicted, labels):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": precision_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(actual, predicted, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            actual, predicted, labels=labels, average="weighted", zero_division=0
        ),
    }


def draw_confusion(axis, matrix, labels, title):
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(title, pad=12)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")


def save_figure(figure, path):
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_metrics(out_dir, report):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics: {path}")
