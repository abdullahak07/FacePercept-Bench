from __future__ import annotations

import numpy as np


def real_reference_centroid_distances(X: np.ndarray, y: np.ndarray, split: np.ndarray):
    """Compute held-out distances to a centroid fit on training REAL faces only."""
    train = split == "train"
    test = split == "test"
    real_train = train & (y == "real")
    if real_train.sum() < 2:
        raise ValueError("Too few REAL training samples for centroid estimation")
    centroid = X[real_train].mean(axis=0, keepdims=True)
    distances = np.linalg.norm(X[test] - centroid, axis=1)
    return distances, y[test]


def hyperrealism_supported(diff_ci: dict[str, float], predictions: np.ndarray) -> bool:
    """Require a positive rate-difference CI and more than one predicted label."""
    return bool(diff_ci.get("lo", float("nan")) > 0 and len(np.unique(predictions)) > 1)
