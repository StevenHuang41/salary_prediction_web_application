import numpy as np
import pandas as pd
import pytest

from app.ml.tune import tune_models
from app.ml.tune.search_spaces import SEARCH_SPACE_REGISTRY


@pytest.fixture
def mock_cross_val_score(monkeypatch):
    def mock(model, X, y, cv, scoring):
        return np.array([-1, -1, -1])

    monkeypatch.setattr(
        "app.ml.tune.tuner.cross_val_score",
        mock,
    )
    yield


def test_tune_models_except_nn(mock_cross_val_score):
    X = np.random.randn(10, 5)
    y = np.random.randn(10)

    for model_name in SEARCH_SPACE_REGISTRY:
        best_params = tune_models(model_name, X, y, n_trials=1)

        assert isinstance(best_params, dict)
