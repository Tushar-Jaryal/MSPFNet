from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    confusion_matrix: np.ndarray

    def to_percent_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy * 100.0,
            "macro_f1": self.macro_f1 * 100.0,
            "macro_precision": self.macro_precision * 100.0,
            "macro_recall": self.macro_recall * 100.0,
        }


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> ClassificationMetrics:
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    return metrics_from_confusion_matrix(cm)


def metrics_from_confusion_matrix(cm: np.ndarray) -> ClassificationMetrics:
    num_classes = int(cm.shape[0])
    total = max(int(cm.sum()), 1)
    acc = float(np.trace(cm) / total)

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return ClassificationMetrics(
        accuracy=acc,
        macro_f1=float(np.mean(f1s)),
        macro_precision=float(np.mean(precisions)),
        macro_recall=float(np.mean(recalls)),
        confusion_matrix=cm,
    )


def aggregate_file_logits(
    y_true: np.ndarray,
    logits: np.ndarray,
    source_files: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate window-level logits to file-level predictions by averaging logits
    over all windows from the same source file.
    """
    if len(y_true) != len(logits) or len(y_true) != len(source_files):
        raise ValueError("y_true, logits, and source_files must have the same length")

    order: list[str] = []
    grouped_logits: dict[str, np.ndarray] = {}
    grouped_labels: dict[str, list[int]] = {}

    for yt, lg, src in zip(y_true.astype(int), logits, source_files.astype(str)):
        if src not in grouped_logits:
            order.append(src)
            grouped_logits[src] = np.asarray(lg, dtype=np.float64).copy()
            grouped_labels[src] = [int(yt)]
        else:
            grouped_logits[src] += np.asarray(lg, dtype=np.float64)
            grouped_labels[src].append(int(yt))

    file_true, file_pred = [], []
    for src in order:
        labels = np.asarray(grouped_labels[src], dtype=np.int64)
        label_counts = np.bincount(labels)
        file_true.append(int(label_counts.argmax()))
        file_pred.append(int(np.asarray(grouped_logits[src]).argmax()))

    return np.asarray(file_true, dtype=np.int64), np.asarray(file_pred, dtype=np.int64)
