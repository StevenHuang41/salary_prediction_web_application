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
    X_train, X_test, y_train, y_test = split_data(sample_df)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

# ratio
def test_split_data_ratio(sample_df):
    X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.3)

    assert X_train.shape[0] == 4
    assert y_train.shape[0] == 4
    assert X_test.shape[0] == 2
    assert y_test.shape[0] == 2

    X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.5)

    assert X_train.shape[0] == 3
    assert y_train.shape[0] == 3
    assert X_test.shape[0] == 3
    assert y_test.shape[0] == 3

# reproducible
def test_split_data_random(sample_df):
    s1 = split_data(sample_df, random_state=42)
    s2 = split_data(sample_df, random_state=42)

    assert (s1[0].index == s2[0].index).all()
    assert (s1[2].index == s2[2].index).all()
