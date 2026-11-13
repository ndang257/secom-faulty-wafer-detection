"""Metrics, the business cost function and evaluation plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import COST_FN, COST_FP, RANDOM_STATE


def misclassification_cost(y_true, y_pred, cost_fp: float = COST_FP, cost_fn: float = COST_FN) -> float:
    """Total cost = cost_fp * false positives + cost_fn * false negatives."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp * cost_fp + fn * cost_fn)


# Scorer for cross-validation (sklearn maximises scores, so cost is negated).
cost_scorer = make_scorer(misclassification_cost, greater_is_better=False)

CV_SCORING = {
    "cost": cost_scorer,
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
}


def classification_metrics(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "cost": misclassification_cost(y_true, y_pred),
    }


def bootstrap_ci(
    y_true, y_pred, y_score, n_boot: int = 2000, alpha: float = 0.05, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Percentile bootstrap confidence intervals for the test-set metrics.

    The test set holds only ~21 failures, so point estimates are noisy;
    the intervals make that uncertainty explicit.
    """
    rng = np.random.default_rng(random_state)
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    rows = []
    n = len(y_true)
    while len(rows) < n_boot:
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        m = classification_metrics(y_true[idx], y_pred[idx], y_score[idx])
        rows.append({k: m[k] for k in ("recall", "precision", "f1", "roc_auc", "cost")})
    boot = pd.DataFrame(rows)
    return boot.quantile([alpha / 2, 1 - alpha / 2]).T.set_axis(["ci_low", "ci_high"], axis=1)


def plot_confusion_and_roc(y_true, y_pred, y_score, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["pass", "fail"], cmap="Blues", colorbar=False, ax=axes[0]
    )
    axes[0].set_title(f"Confusion matrix - {title}")
    RocCurveDisplay.from_predictions(y_true, y_score, name=title, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title(f"ROC curve - {title}")
    fig.tight_layout()
    return fig


def plot_cost_breakdown(results: pd.DataFrame, label_col: str = "model"):
    """Stacked bar chart of false-positive and false-negative cost per model."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(results))
    fp_cost = results["fp"] * COST_FP
    fn_cost = results["fn"] * COST_FN
    ax.bar(x, fp_cost, label=f"False alarms ({COST_FP} each)", color="#4C72B0")
    ax.bar(x, fn_cost, bottom=fp_cost, label=f"Missed failures ({COST_FN} each)", color="#C44E52")
    for i, total in enumerate(fp_cost + fn_cost):
        ax.text(i, total, f"{total:,.0f}", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x, results[label_col])
    ax.set_ylabel("Total misclassification cost")
    ax.set_title("Test-set cost by model")
    ax.legend(loc="upper right")
    ax.margins(y=0.12)
    fig.tight_layout()
    return fig
