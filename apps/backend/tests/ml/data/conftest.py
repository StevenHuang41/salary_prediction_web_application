import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def raw_column_df():
    df = pd.DataFrame({
        'Age': [20],
        'gender': ['Female'],
        ' education    Level': ["master's degree"],
        ' Job   title': ['Data Scientist'],
        ' years  Of experience': [2],
        '   salary ': [36_000]
    })
    return df

@pytest.fixture
def raw_full_df():
    df = pd.DataFrame({
        'age': [20, 19, 28, 27],
        'gender': [np.nan, 'male', 'other', 'male'],
        'education_level': ["master's degree", 'Bachelor', 'PhD', 'high school'],
        'job_title': ['Data Scientist', 'Data Engineer', 'Data Analyst', 'driver'],
        'years_of_experience': [2, 1, 3, 5],
        'salary': [36_000, np.nan, 900_000, 1_000]
    })
    return df

@pytest.fixture
def raw_df():
    df = pd.DataFrame({
        'Age': [20, 19, 28, 27, 27],
        'gender': ['Female', 'male', 'other', 'male', 'male'],
        'education Level': ["master's degree", 'Bachelor', 'PhD', 'high school', 'high school'],
        'Job   title': ['Data Scientist', 'Data Engineer', 'Data Analyst', 'driver', 'driver'],
        ' years  Of experience': [2, 1, 3, 5, 5],
    })
    return df


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

