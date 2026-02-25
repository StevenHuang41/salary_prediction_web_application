import pytest
import pandas as pd

from app.ml.data.spliting import split_data

@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "age": [25, 32, 24, 31, 29, 33],
        "gender": ["female", "male", "female", "male", "female", "male"],
        "education_level": ["high_school", "phd", "master", "bachelor", "master", "phd"],
        "job_title": [
            "ai engineer",
            "data scientist",
            "ai engineer",
            "senior data scientist",
            "senior ai engineer",
            "senior data engineer"
        ],
        "years_of_experience": [3, 5, 2, 3, 7, 5],
        "salary": [120000, 210000, 110000, 100000, 220000, 150000],
    })
    return df

# type
def test_split_data_return_type(sample_df):
    train_df, test_df = split_data(sample_df)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

# cols
def test_split_data_return_cols(sample_df):
    train_df, test_df = split_data(sample_df)

    assert "job_seniority" in train_df.columns
    assert "job_group" in train_df.columns
    assert "job_role" in train_df.columns

    assert "job_seniority" in test_df.columns
    assert "job_group" in test_df.columns
    assert "job_role" in test_df.columns

# ratio
def test_split_data_ratio(sample_df):
    train_df, test_df = split_data(sample_df, test_size=0.3)

    assert train_df.shape[0] == 4
    assert test_df.shape[0] == 2

    train_df, test_df = split_data(sample_df, test_size=0.5)

    assert train_df.shape[0] == 3
    assert test_df.shape[0] == 3

# reproducible
def test_split_data_random(sample_df):
    s1 = split_data(sample_df, random_state=42)
    s2 = split_data(sample_df, random_state=42)

    assert (s1[0].index == s2[0].index).all()
    assert (s1[1].index == s2[1].index).all()
