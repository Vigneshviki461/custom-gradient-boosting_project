import numpy as np
import pytest

from gradient_boosting import DecisionTree, GradientBoostedRegressor, _soft_threshold

X_SAMPLE = np.array([[0.1, 0.2], [0.5, 0.8], [1.2, 0.3], [1.5, 1.1], [0.8, 1.5]])
Y_SAMPLE = np.array([0.19, 0.65, 0.28, 0.95, 0.05])


def test_soft_threshold():
    assert _soft_threshold(5, 2) == 3
    assert _soft_threshold(-5, 2) == -3
    assert _soft_threshold(1, 2) == 0


def test_leaf_weight_formula():
    tree = DecisionTree(max_depth=0, reg_lambda=2, reg_alpha=0.5)
    tree.fit(X_SAMPLE[:2], np.array([1.5, 2.5]), np.ones(2))
    assert tree.root.value == pytest.approx(-0.875)


def test_fit_predict_shapes_and_loss_history():
    model = GradientBoostedRegressor(n_estimators=5, max_depth=2).fit(X_SAMPLE, Y_SAMPLE)
    prediction = model.predict(X_SAMPLE)
    assert prediction.shape == Y_SAMPLE.shape
    assert np.all(np.isfinite(prediction))
    assert len(model.loss_history_) == model.n_estimators + 1


def test_same_seed_gives_same_predictions():
    first = GradientBoostedRegressor(n_estimators=5, random_state=7).fit(X_SAMPLE, Y_SAMPLE)
    second = GradientBoostedRegressor(n_estimators=5, random_state=7).fit(X_SAMPLE, Y_SAMPLE)
    np.testing.assert_allclose(first.predict(X_SAMPLE), second.predict(X_SAMPLE))


def test_tree_requires_fit_before_predict():
    with pytest.raises(RuntimeError, match="Call fit before using predict"):
        DecisionTree().predict(X_SAMPLE)


def test_target_length_must_match():
    with pytest.raises(ValueError, match="same non-zero number of rows"):
        GradientBoostedRegressor().fit(X_SAMPLE, Y_SAMPLE[:-1])


def test_wrong_feature_count_is_rejected():
    model = GradientBoostedRegressor(n_estimators=2).fit(X_SAMPLE, Y_SAMPLE)
    with pytest.raises(ValueError, match="exactly 2 features"):
        model.predict(np.array([[1.0, 2.0, 3.0]]))


def test_invalid_learning_rate_is_rejected():
    with pytest.raises(ValueError, match="learning_rate"):
        GradientBoostedRegressor(learning_rate=0)
