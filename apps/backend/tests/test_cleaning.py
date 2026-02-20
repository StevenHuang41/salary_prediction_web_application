import numpy as np
import pandas as pd
from app.ml.data.cleaning import rename_cols, clean_salary, clean_data

def test_rename_cols():
    df = pd.DataFrame({
        'Age': [20],
        'gender': ['Female'],
        ' education    Level': ["master's degree"],
        ' Job   title': ['Data Scientist'],
        ' years  Of experience': [2],
        '   salary ': [36_000]
    })

    rename_cols(df)

    assert df.columns.str.islower().all()
    assert not df.columns.str.strip().str.contains(r'\s').any()
    assert not df.columns.str.contains(r'[^a-z0-9_]').any()

def test_clean_salary():
    df = pd.DataFrame({
        'age': [20, 19, 28, 27],
        'gender': ['Female', 'male', 'other', 'male'],
        'education_level': ["master's degree", 'Bachelor', 'PhD', 'high school'],
        'job_title': ['Data Scientist', 'Data Engineer', 'Data Analyst', 'driver'],
        'years_of_experience': [2, 1, 3, 5],
        'salary': [36_000, np.nan, 900_000, 1_000]
    })

    cleaned = clean_salary(df)

    assert cleaned.shape[0] == 1

def test_clean_data():
    df = pd.DataFrame({
        'Age': [20, 19, 28, 27, 27],
        'gender': ['Female', 'male', 'other', 'male', 'male'],
        'education Level': ["master's degree", 'Bachelor', 'PhD', 'high school', 'high school'],
        'Job   title': ['Data Scientist', 'Data Engineer', 'Data Analyst', 'driver', 'driver'],
        ' years  Of experience': [2, 1, 3, 5, 5],
    })

    cleaned = clean_data(df)

    # check remove nan
    assert not cleaned.isna().any().any()

    # col: age
    assert cleaned.age.dtype.name == 'int32'

    # col: gender
    assert list(cleaned.gender.unique().sort_values()) == ['female', 'male', 'other']

    # col: education level
    assert list(cleaned.education_level.unique().sort_values()) == \
    ['high_school', 'bachelor', 'master', 'phd']

    # col: job title
    assert not cleaned.job_title.str.contains(r'\s{2,}').any()
    assert not cleaned.job_title.str.contains(r'juniour').any()
    assert not cleaned.job_title.str.contains(r'rep\b').any()
    assert not cleaned.job_title.str.contains(r'\bman\b').any()
    assert not cleaned.job_title.str.contains(r'director of.*').any()

    # col: years of experience
    assert cleaned.years_of_experience.dtype.name == 'float32'

    # check remove incorrect age-experience rows
    assert not (cleaned.age - cleaned.years_of_experience < 18).any()

    # check drop duplicated
    assert not (cleaned.duplicated()).any()
