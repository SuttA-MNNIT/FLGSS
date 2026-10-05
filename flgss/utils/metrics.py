import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes standard evaluation metrics: Accuracy, Precision, Recall, Macro-F1.
    """
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {
        "accuracy": acc,
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "f1_score": f1 * 100.0
    }

def compute_anomaly_detection_auc(scores: np.ndarray, labels_binary: np.ndarray) -> float:
    """
    Computes ROC-AUC for anomaly detection benchmark.
    - scores: higher values mean more anomalous.
    - labels_binary: 0 = normal, 1 = anomalous.
    """
    try:
        auc = roc_auc_score(labels_binary, scores)
        return float(auc)
    except Exception as e:
        print(f"Error computing AUC: {e}")
        return 0.5
