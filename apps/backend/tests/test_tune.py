import numpy as np
import pandas as pd
import pytest

from app.ml.tune.linear import tune_linear_family
from app.ml.tune.tree import tune_tree_family
from app.ml.tune.nn import tune_nn_family


@pytest.fixture
def mock_cross_val_score(monkeypatch):
    def mock(model, X, y, cv, scoring):
        return np.array([-1, -1, -1])

    monkeypatch.setattr(
        "app.ml.tune.linear.cross_val_score",
        mock,
    )

    monkeypatch.setattr(
        "app.ml.tune.tree.cross_val_score",
        mock,
    )

    monkeypatch.setattr(
        "app.ml.tune.nn.cross_val_score",
        mock,
    )
    yield


def test_tune_linear_family(mock_cross_val_score):
    X = np.random.randn(10, 5)
    y = np.random.randn(10)

    best_model, best_params = tune_linear_family(X, y, n_trials=1)

    assert isinstance(best_model, str)
    assert isinstance(best_params, dict)


def test_tune_tree_family(mock_cross_val_score):
    X = np.random.randn(10, 5)
    y = np.random.randn(10)

    best_model, best_params = tune_tree_family(X, y, n_trials=1)

    assert isinstance(best_model, str)
    assert isinstance(best_params, dict)


def test_tune_nn_family(mock_cross_val_score):
    X = np.random.randn(10, 5)
    y = np.random.randn(10)

    best_model, best_params = tune_nn_family(X, y, n_trials=1)

    assert isinstance(best_model, str)
    assert isinstance(best_params, dict)

