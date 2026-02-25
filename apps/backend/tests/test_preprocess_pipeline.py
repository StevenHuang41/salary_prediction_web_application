import pytest
import pandas as pd

from sklearn.compose import ColumnTransformer
import numpy as np

from app.ml.preprocess.linear import build as build_linear_pipe
from app.ml.preprocess.tree import build as build_tree_pipe
from app.ml.preprocess.nn import build as build_nn_pipe


@pytest.fixture
def sample_df():
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
def sample_target() -> pd.Series:
    return pd.Series(
        [120000, 210000, 110000, 100000, 220000, 150000],
        name="salary"
    )


def test_linear_pipeline(sample_df: pd.DataFrame, sample_target: pd.Series):
    linear_pipe = build_linear_pipe()
    assert isinstance(linear_pipe, ColumnTransformer)

    result = linear_pipe.fit_transform(sample_df, sample_target)
    columns = linear_pipe.get_feature_names_out()

    assert result.shape == (6, 8)

    assert "age" not in columns

    assert np.issubdtype(result.dtype, np.number)


def test_tree_pipeline(sample_df: pd.DataFrame, sample_target: pd.Series):
    tree_pipe = build_tree_pipe()
    assert isinstance(tree_pipe, ColumnTransformer)

    result = tree_pipe.fit_transform(sample_df, sample_target)
    columns = tree_pipe.get_feature_names_out()

    assert result.shape == (6, 10)

    assert "age" in columns

    assert np.issubdtype(result.dtype, np.number)


def test_nn_pipeline(sample_df: pd.DataFrame, sample_target: pd.Series):
    nn_pipe = build_nn_pipe()
    assert isinstance(nn_pipe, ColumnTransformer)

    result = nn_pipe.fit_transform(sample_df, sample_target)
    columns = nn_pipe.get_feature_names_out()

    assert result.shape == (6, 8)

    assert "age" in columns

    assert np.issubdtype(result.dtype, np.number)


