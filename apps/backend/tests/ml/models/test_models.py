import pytest
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from app.ml.models.linear import (
    build_linear,
    build_ridge,
    build_lasso,
    build_elasticNet,
)
from app.ml.models.tree import (
    build_HGBR,
    build_xgb,
    build_xgbrf,
)
from app.ml.models.nn import (
    build_MLP,
)

@pytest.fixture
def sample_X():
    df = pd.DataFrame({
        "age": [25, 32, 24, 31, 29, 33],
        "gender": ["female", "male", "female", "male", "other", "male"],
        "education_level": ["high_school", "phd", "master", "bachelor", "master", "phd"],
        "job_title": [
            "back end engineer",
            "data scientist",
            "junior software engineer",
            "senior data scientist",
            "senior project engineer",
            "senior data engineer"
        ],
        "job_seniority": ["mid", "mid", "junior", "senior", "senior", "senior"],
        "job_group": ["backend", "data", "software", "data", "project", "data"],
        "job_role": [
            "engineer",
            "scientist",
            "engineer",
            "scientist",
            "engineer",
            "engineer"
        ],
        "years_of_experience": [3, 5, 2, 3, 7, 5],
    })
    return df

@pytest.fixture
def sample_y() -> pd.Series:
    return pd.Series(
        [120000, 210000, 110000, 100000, 220000, 150000],
        name="salary"
    )

@pytest.fixture
def sample_X_test() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 32],
        "gender": ["female", "male"],
        "education_level": ["high_school", "phd"],
        "job_title": ["back end engineer", "senior data worker",],
        "job_seniority": ["mid", "senior"],
        "job_group": ["backend", "data"],
        "job_role": ["engineer", "worker",],
        "years_of_experience": [3, 10],
    })

# linear
def test_build_linear(sample_X, sample_y, sample_X_test):
    model = build_linear()
    assert isinstance(model, Pipeline)

    model.fit(sample_X, sample_y)
    y_pred = model.predict(sample_X_test)
    assert len(y_pred) == 2
    assert np.issubdtype(y_pred.dtype, np.number)

def test_build_ridge(sample_X, sample_y, sample_X_test):
    model = build_ridge()
    assert isinstance(model, Pipeline)

    model.fit(sample_X, sample_y)
    y_pred = model.predict(sample_X_test)
    assert len(y_pred) == 2
    assert np.issubdtype(y_pred.dtype, np.number)

def test_build_lasso(sample_X, sample_y, sample_X_test):
    model = build_lasso()
    assert isinstance(model, Pipeline)

    model.fit(sample_X, sample_y)
    y_pred = model.predict(sample_X_test)
    assert len(y_pred) == 2
    assert np.issubdtype(y_pred.dtype, np.number)

def test_build_elasticNet(sample_X, sample_y, sample_X_test):
    model = build_elasticNet()
    assert isinstance(model, Pipeline)

    model.fit(sample_X, sample_y)
    y_pred = model.predict(sample_X_test)
    assert len(y_pred) == 2
    assert np.issubdtype(y_pred.dtype, np.number)


# tree
def tejt_build_HGBR(sample_X, sample_y, sample_X_test):
    model = build_HGBR()
    assert isinstance(model, Pipeline)

    model.fit(sample_X, sample_y)
    y_pred = model.predict(sample_X_test)
    assert len(y_pred) == 2
    assert np.issubdtype(y_pred.dtype, np.number)

def test_build_xgb(sample_X, sample_y, sample_X_test):
    model = build_xgb()
    assert isinstance(model, Pipeline)

    model.fit(sample_X, sample_y)
    y_pred = model.predict(sample_X_test)
    assert len(y_pred) == 2
    assert np.issubdtype(y_pred.dtype, np.number)

def test_build_xgbrf(sample_X, sample_y, sample_X_test):
    model = build_xgbrf()
    assert isinstance(model, Pipeline)

    model.fit(sample_X, sample_y)
    y_pred = model.predict(sample_X_test)
    assert len(y_pred) == 2
    assert np.issubdtype(y_pred.dtype, np.number)


# nn
def test_build_MLP_default(sample_X, sample_y, sample_X_test):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        model = build_MLP()
        assert isinstance(model, Pipeline)

        model.fit(sample_X, sample_y)
        y_pred = model.predict(sample_X_test)
        assert model.named_steps["model"].hidden_layer_sizes == (100,)
        assert len(y_pred) == 2
        assert np.issubdtype(y_pred.dtype, np.number)

def test_build_MLP_arguments(sample_X, sample_y, sample_X_test):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        model = build_MLP(hidden_layer_sizes=(64, 32), max_iter=2000)
        assert isinstance(model, Pipeline)

        model.fit(sample_X, sample_y)
        y_pred = model.predict(sample_X_test)
        assert model.named_steps["model"].hidden_layer_sizes == (64, 32)
        assert model.named_steps["model"].max_iter == 2000
        assert len(y_pred) == 2
        assert np.issubdtype(y_pred.dtype, np.number)

