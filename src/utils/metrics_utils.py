"""
metrics_utils.py
Funciones métricas y de evaluación
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "confusion_matrix": cm}


def frobenius_norm(X, X_rec):
    """Distancia Frobenius total entre X y su reconstrucción."""
    diff = X - X_rec
    return np.linalg.norm(diff, ord="fro")
