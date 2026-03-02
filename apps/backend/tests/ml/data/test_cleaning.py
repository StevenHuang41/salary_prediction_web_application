import numpy as np
import pandas as pd
from app.ml.data.cleaning import rename_cols, clean_salary, clean_data


def test_rename_cols(raw_column_df):
    rename_cols(raw_column_df)

    assert raw_column_df.columns.str.islower().all()
    assert not raw_column_df.columns.str.strip().str.contains(r'\s').any()
    assert not raw_column_df.columns.str.contains(r'[^a-z0-9_]').any()

def test_clean_salary(raw_full_df):
    cleaned = clean_salary(raw_full_df)

    assert cleaned.shape[0] == 1

def test_clean_data(raw_df):
    cleaned = clean_data(raw_df)

    # check no nan
    assert not cleaned.isna().any().any()

    # col: age
    assert cleaned.age.dtype.name == 'float32'

    # col: gender
    assert list(cleaned["gender"].cat.categories) == \
        ["female", "male", "other"]

    # col: education level
    assert list(cleaned["education_level"].cat.categories) == \
        ["unknown", "high_school", "bachelor", "master", "phd"]

    # col: job title
    assert not cleaned.job_title.str.contains(r'\s{2,}').any()
    assert not cleaned.job_title.str.contains(r'juniour').any()
    assert not cleaned.job_title.str.contains(r'junior').any()
    assert not cleaned.job_title.str.contains(r'senior').any()
    assert not cleaned.job_title.str.contains(r'rep\b').any()
    assert not cleaned.job_title.str.contains(r'\bman\b').any()
    assert not cleaned.job_title.str.contains(r'director of.*').any()
    assert not cleaned.job_title.str.contains(r'vp of.*').any()

    # col: years of experience
    assert cleaned.years_of_experience.dtype.name == 'float32'

    # check removed incorrect age-experience rows
    assert not (cleaned.age - cleaned.years_of_experience < 18).any()

    # check dropped duplicated
    assert not (cleaned.duplicated()).any()

    assert "job_seniority" in cleaned.columns
    assert "job_group" in cleaned.columns
    assert "job_role" in cleaned.columns
