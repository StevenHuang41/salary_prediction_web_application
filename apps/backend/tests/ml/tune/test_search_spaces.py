import optuna
import pytest

from app.ml.tune.search_spaces import SEARCH_SPACE_REGISTRY

def test_space_linear_return_dict():
    trial = optuna.trial.FixedTrial({})
    params = SEARCH_SPACE_REGISTRY["linear"](trial)

    assert isinstance(params, dict)

def test_space_ridge_return_dict():
    trial = optuna.trial.FixedTrial({
        "alpha": 1e-3,
    })
    params = SEARCH_SPACE_REGISTRY["ridge"](trial)

    assert isinstance(params, dict)

def test_space_lasso_return_dict():
    trial = optuna.trial.FixedTrial({
        "alpha": 1e-3,
        "selection": "random",
    })
    params = SEARCH_SPACE_REGISTRY["lasso"](trial)

    assert isinstance(params, dict)

def test_space_elasticNet_return_dict():
    trial = optuna.trial.FixedTrial({
        "alpha": 1e-3,
        "l1_ratio": 0.01,
    })
    params = SEARCH_SPACE_REGISTRY["elasticNet"](trial)

    assert isinstance(params, dict)


def test_space_HGBR_return_dict():
    trial = optuna.trial.FixedTrial({
        "learning_rate": 0.01,
        "max_iter": 100,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "l2_regularization": 0.0,
    })
    params = SEARCH_SPACE_REGISTRY["HGBR"](trial)

    assert isinstance(params, dict)

def test_space_xgb_return_dict():
    trial = optuna.trial.FixedTrial({
        "learning_rate": 0.01,
        "max_depth": 3,
        "n_estimators": 100,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "reg_lambda": 0.0,
    })
    params = SEARCH_SPACE_REGISTRY["xgb"](trial)

    assert isinstance(params, dict)

def test_space_xgbrf_return_dict():
    trial = optuna.trial.FixedTrial({
        "max_depth": 3,
        "n_estimators": 100,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "reg_lambda": 0.0,
    })
    params = SEARCH_SPACE_REGISTRY["xgbrf"](trial)

    assert isinstance(params, dict)


def test_space_nn_return_dict():
    trial = optuna.trial.FixedTrial({
        "n_layers": 3,
        "n_node_0": 8,
        "n_node_1": 8,
        "n_node_2": 8,
        "learning_rate": "adaptive",
        "learning_rate_init": 1e-4,
        "max_iter": 800,
        "alpha": 1e-6,
        "activation": "relu",
        "batch_size": 32,
    })
    params = SEARCH_SPACE_REGISTRY["nn"](trial)

    assert isinstance(params, dict)
    assert params["hidden_layer_sizes"] == (8, 8, 8)
    assert params["early_stopping"] is True

def test_space_nn_dynamic_layer():
    with pytest.raises(ValueError):
        trial = optuna.trial.FixedTrial({
            "n_layers": 3,
            "n_node_0": 8,
            "learning_rate": "adaptive",
            "learning_rate_init": 1e-4,
            "max_iter": 800,
            "alpha": 1e-6,
            "activation": "relu",
            "batch_size": 32,
        })
        params = SEARCH_SPACE_REGISTRY["nn"](trial)
