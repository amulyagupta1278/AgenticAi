import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(model, X_test, y_test, class_names=None) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    print(f"\nAccuracy : {metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {metrics['f1']:.4f}")

    if class_names is not None:
        print("\n" + classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    return metrics


def print_confusion_matrix(model, X_test, y_test, class_names=None):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    if class_names is not None:
        header = " " * 20 + "  ".join(f"{c[:8]:>8}" for c in class_names)
        print(header)
        for i, row in enumerate(cm):
            label = class_names[i][:18] if class_names is not None else str(i)
            print(f"{label:>20}  " + "  ".join(f"{v:>8}" for v in row))
    else:
        print(cm)
