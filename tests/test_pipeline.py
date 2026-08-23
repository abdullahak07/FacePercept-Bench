import numpy as np

from facepercept.inference import parse_binary_prediction
from facepercept.statistics import hyperrealism_supported, real_reference_centroid_distances


def test_prediction_parser():
    assert parse_binary_prediction("REAL") == "real"
    assert parse_binary_prediction("SYNTHETIC") == "synthetic"


def test_hyperrealism_requires_positive_ci_and_nonconstant_predictions():
    assert hyperrealism_supported({"lo": 0.01, "hi": 0.1}, np.array(["real", "synthetic"]))
    assert not hyperrealism_supported({"lo": -0.01, "hi": 0.1}, np.array(["real", "synthetic"]))
    assert not hyperrealism_supported({"lo": 0.01, "hi": 0.1}, np.array(["real", "real"]))


def test_real_reference_centroid():
    X = np.array([[0., 0.], [0., 2.], [4., 4.], [1., 1.]])
    y = np.array(["real", "real", "synthetic", "synthetic"])
    split = np.array(["train", "train", "test", "test"])
    d, yt = real_reference_centroid_distances(X, y, split)
    assert d.shape == (2,)
    assert list(yt) == ["synthetic", "synthetic"]
