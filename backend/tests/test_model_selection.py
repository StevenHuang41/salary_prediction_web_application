import numpy as np

from app.ml.training.model_selection import model_selection

# mock model
class DummyModel:
    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X, **kwargs):
        # always predict zero
        return np.zeros(len(X))

def test_model_selection_basic(monkeypatch):
    # mock cv score
    def mock_cross_val_score(model, X, y, cv, scoring):
        # fix mse loss = 100
        return np.array([-100])

    monkeypatch.setattr("app.ml.training.model_selection.cross_val_score", mock_cross_val_score)

    # mock all sklearn models & keras models 
    monkeypatch.setattr("app.ml.training.model_selection.LinearRegression", lambda **kw: DummyModel())
    monkeypatch.setattr("app.ml.training.model_selection.Ridge", lambda **kw: DummyModel())
    monkeypatch.setattr("app.ml.training.model_selection.Lasso", lambda **kw: DummyModel())
    monkeypatch.setattr("app.ml.training.model_selection.ElasticNet", lambda **kw: DummyModel())
    monkeypatch.setattr("app.ml.training.model_selection.SVR", lambda **kw: DummyModel())
    monkeypatch.setattr("app.ml.training.model_selection.KNeighborsRegressor", lambda **kw: DummyModel())
    monkeypatch.setattr("app.ml.training.model_selection.DecisionTreeRegressor", lambda **kw: DummyModel())
    monkeypatch.setattr("app.ml.training.model_selection.RandomForestRegressor", lambda **kw: DummyModel())
    monkeypatch.setattr("app.ml.training.model_selection.XGBRFRegressor", lambda **kw: DummyModel())
    monkeypatch.setattr("app.ml.training.model_selection.GradientBoostingRegressor", lambda **kw: DummyModel())
    monkeypatch.setattr("app.ml.training.model_selection.XGBRegressor", lambda **kw: DummyModel())
    monkeypatch.setattr("app.ml.training.model_selection.LGBMRegressor", lambda **kw: DummyModel())

    # mock build_nn_model
    monkeypatch.setattr(
        "app.ml.training.model_selection.build_nn_model",
        lambda trial, *args: DummyModel(),
    )

    # fake dataset
    X = np.random.rand(20, 5)
    y = np.random.rand(20)

    # runs model_selection (without actually run it)
    best_params = model_selection(X, y)

    assert isinstance(best_params, dict)
    assert "model" in best_params  
