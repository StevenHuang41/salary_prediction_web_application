import pytest
import pandas as pd
import numpy as np

from app.ml.features.ml_transformers import (
    JobSeniorityTransformer,
    JobGroupTransformer,
    TextExistTransformer,
    MathTransformer,
    MultiLabelWrapper,
)

@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "age": [25, 30],
        "gender": ["female", "male"],
        "education_level": ["master", "phd"],
        "job_title": ["data scientist", "senior software engineer"],
        "years_of_experience": [3, 5],
        "salary": [120000, 150000],
    })
    return df


def test_JobSeniority(sample_df):
    jobT = JobSeniorityTransformer()

    result = jobT.fit_transform(sample_df.job_title)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 1

    assert result[0, 0] == "mid"
    assert result[1, 0] == "senior"

    col_name = jobT.get_feature_names_out()[0]
    assert col_name == 'seniority'

    assert isinstance(jobT.fit_transform(sample_df[["job_title"]]), np.ndarray)
    assert isinstance(jobT.fit_transform(sample_df.job_title.to_numpy()), np.ndarray)


def test_JobGroup(sample_df):
    jobT = JobGroupTransformer()

    result = jobT.fit_transform(sample_df.job_title)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 1

    assert result[0, 0] == "data"
    assert result[1, 0] == "software_engineer_developer"

    col_name = jobT.get_feature_names_out()[0]
    assert col_name == "group"

    assert isinstance(jobT.fit_transform(sample_df[["job_title"]]), np.ndarray)
    assert isinstance(jobT.fit_transform(sample_df.job_title.to_numpy()), np.ndarray)


def test_TextExist(sample_df):
    jobT = TextExistTransformer(text="senior")

    result = jobT.fit_transform(sample_df.job_title)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 1

    assert result[0, 0] == 0
    assert result[1, 0] == 1

    col_name = jobT.get_feature_names_out()[0]
    assert col_name == "is_senior"

    assert isinstance(jobT.fit_transform(sample_df[["job_title"]]), np.ndarray)
    assert isinstance(jobT.fit_transform(sample_df.job_title.to_numpy()), np.ndarray)


def test_MathTransformer_wrong_method(sample_df):
    with pytest.raises(ValueError, match="Unknown method: s"):
        jobT = MathTransformer(method="s")          # type: ignore
        result = jobT.fit_transform(sample_df.age)  # type: ignore

def test_MathTransformer(sample_df):
    jobT = MathTransformer(method="square", suffix='test_suffix')

    result = jobT.fit_transform(sample_df.age)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 1

    assert result[0, 0] == 625
    assert result[1, 0] == 900

    col_name = jobT.get_feature_names_out()[0]
    assert col_name == "age_test_suffix"

    assert isinstance(jobT.fit_transform(sample_df[["age"]]), np.ndarray)
    assert isinstance(jobT.fit_transform(sample_df.age.to_numpy()), np.ndarray)


def test_MultiLabelWrapper(sample_df):
    jobT = MultiLabelWrapper()
    result = jobT.fit_transform(sample_df.job_title)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 5

    assert (result[0] == [1, 0, 1, 0, 0]).all()
    assert (result[1] == [0, 1, 0, 1, 1]).all()

    col_name = jobT.get_feature_names_out()
    assert (col_name == ['data', 'engineer', 'scientist', 'senior', 'software']).all()

    assert isinstance(jobT.fit_transform(sample_df[["job_title"]]), np.ndarray)
    assert isinstance(jobT.fit_transform(sample_df.job_title.to_numpy()), np.ndarray)
