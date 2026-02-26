import pytest
import numpy as np

from app.ml.train.compare import compare_model_family


def test_compare_return(monkeypatch):

    class MockModel:
        def __init__(self, name):
            self.name = name

    def mock_cross_val_score(model, X, y, cv, scoring):
        if model.name == "A":
            return np.array([-1, -1, -1])
        elif model.name == "B":
            return np.array([-2, -2, -2])
        else :
            return np.array([0, 0, 0])

    monkeypatch.setattr(
        "app.ml.train.compare.cross_val_score",
        mock_cross_val_score,
    )

    models = {
        "modela": lambda: MockModel("A"),
        "modelb": lambda: MockModel("B"),
        "modelc": lambda: MockModel("C"),
    }

    best_model_name = compare_model_family(models, X=None, y=None)

    assert best_model_name == "modelb"



