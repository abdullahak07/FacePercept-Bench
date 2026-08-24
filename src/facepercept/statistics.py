import numpy as np


def real_reference_centroid_distances(X, y, split):
    train = split == "train"
    test = split == "test"
    real_train = train & (y == "real")

    if real_train.sum() < 2:
        raise ValueError("need at least two real training samples")

    real_centroid = X[real_train].mean(axis=0, keepdims=True)
    d = np.linalg.norm(X[test] - real_centroid, axis=1)
    return d, y[test]


def hyperrealism_supported(diff_ci, predictions):
    # only count it when the interval is actually above zero
    return diff_ci.get("lo", np.nan) > 0 and len(np.unique(predictions)) > 1
