import pandas as pd
import numpy as np
from app.ml.data.cleaning import clean_data

df1_cols = [
    'Age', ' gender', 'education   level ',
    'Job title', 'years Of experience', 'SAlary'
]
correct1_cols = [
    'age', 'gender', 'education_level',
    'job_title', 'years_of_experience', 'salary'
]
df1 = pd.DataFrame([
    [20, "Female", "master's degree", "Data Scientist", 2, 9_000],
    [19, "male", "other", "Data Engineer", 1, 310_000],
    [28, "other", "PhD", "Data Analyst", 3, 100_000],
    [28, "Other", "phd", "Data Analyst", 3, 100_000],
    [18, "Female", "Master", "Accountant", 10, 111_000],
    [18, "Female", "Master", "Accountant", 10, None],
], columns=df1_cols)


df2_cols = [
    'Age', ' gender', 'education   level ',
    'Job title', 'years Of experience'
]
correct2_cols = [
    'age', 'gender', 'education_level',
    'job_title', 'years_of_experience'
]
df2 = pd.DataFrame([
    [20, " Female", "master's degree", "Data Scientist", 2],
    [19, "male", "other", "Data Engineer", 1],
    [28, "other ", "PhD", "Data Analyst", 3],
    [28, "Other", "phd", "Data Analyst", 3],
    [18, "Female", "Master", "Accountant", 10],
    [28, None, "Master", "Accountant", 10],
    [8, "male", "Master", "Director of Marketing", 0],
    [38, "male", "phd", "Director of Marketing", 0],
    [38, "male", "phd", "juniour Marketing", 0],
    [23, "male", "phd", "sales rep", 0],
    [23, "male", "high school", "IT man", 0],
    [18, "male", "bachelor", "Data Scientist", 10],
    [18, "male", None, "Data Scientist", 0],
], columns=df2_cols)

# test df included target
def test_column_names_lowercase_1(df=df1):
    cleaned = clean_data(df, has_target_col=True)
    assert (cleaned.columns == correct1_cols).all() 

def test_target_col_no_nan(df=df1):
    cleaned = clean_data(df, has_target_col=True)
    assert cleaned.iloc[:, -1].isna().sum() == 0 

def test_duplicated_1(df=df1):
    cleaned = clean_data(df, has_target_col=True)
    assert cleaned.duplicated().sum() == 0 

def test_target_outlier(df=df1, lower_bound=int(10_000), upper_bound=int(300_000)):
    cleaned = clean_data(df, has_target_col=True)
    assert (cleaned.iloc[:, -1] <= lower_bound).sum() == 0 
    assert (cleaned.iloc[:, -1] >= upper_bound).sum() == 0 
    
    
# test df excluded target
def test_column_names_lowercase_2(df=df2):
    cleaned = clean_data(df)
    assert (cleaned.columns == correct2_cols).all() 
    
def test_no_nan(df=df2):
    cleaned = clean_data(df)
    assert cleaned.isna().any().sum() == 0
    
def test_age_val(df=df2):
    cleaned = clean_data(df)
    assert (cleaned.age < 18).sum() == 0
    
def test_gender_val(df=df2):
    cleaned = clean_data(df)
    correct_gen_val = {'female', 'male', 'other'}
    assert set(cleaned['gender'].unique()).issubset(correct_gen_val)

def test_edu_val(df=df2):
    cleaned = clean_data(df)
    correct_edu_val = {'high school', 'bachelor', 'master', 'phd'}
    assert set(cleaned['education_level'].unique()).issubset(correct_edu_val)
    
def test_job_title_val(df=df2):
    cleaned = clean_data(df)
    assert not cleaned.job_title.str.contains(r'juniour').any()
    assert not cleaned.job_title.str.contains(r'rep\b', regex=True).any()
    assert not cleaned.job_title.str.contains(r'\bman\b', regex=True).any()
    assert not cleaned.job_title.str.contains(r'director of', regex=True).any()
    
def test_age_yearE_val(df=df2):
    cleaned = clean_data(df)
    assert ((cleaned.age - cleaned.years_of_experience) >= 18).all()
    
def test_exist_duplicated_rows_when_no_target(df=df2):
    cleaned = clean_data(df)
    dup_rows = pd.Series([28, "other", "phd", "data analyst", 3], index=correct2_cols)
    assert ((cleaned == dup_rows).all(axis=1)).sum() == 2