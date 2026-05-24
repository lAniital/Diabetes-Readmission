"""Clinical evaluation utilities for the diabetes readmission project."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    """Evaluate a trained classifier on the test set."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Confusion Matrix:")
    print(matrix)
    print("\nClassification Report:")
    print(report)
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return {
        "confusion_matrix": matrix,
        "classification_report": report,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }
