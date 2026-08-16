import numpy as np
from sklearn.linear_model import LogisticRegression

from src.simulate import evaluate, make_data, strategic_response


def test_make_data_is_reproducible():
    first_x, first_y = make_data(n=50, seed=7)
    second_x, second_y = make_data(n=50, seed=7)
    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y, second_y)


def test_strategic_response_never_decreases_manipulable_feature():
    features, target = make_data(n=300, seed=3)
    model = LogisticRegression(max_iter=1_000).fit(features, target)
    changed, deltas = strategic_response(features, model)
    assert np.all(deltas >= 0)
    assert np.allclose(changed[:, 0], features[:, 0] + deltas)
    assert np.allclose(changed[:, 1], features[:, 1])


def test_evaluation_metrics_are_bounded():
    features, target = make_data(n=300, seed=11)
    model = LogisticRegression(max_iter=1_000).fit(features, target)
    metrics = evaluate(model, features, target)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
