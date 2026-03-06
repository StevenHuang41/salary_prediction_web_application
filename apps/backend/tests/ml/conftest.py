import pytest
import pandas as pd


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


