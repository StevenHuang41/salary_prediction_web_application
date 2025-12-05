import pandas as pd
import numpy as np

from app.ml.preprocessing import (
    scaler_wrapper,
    build_preprocessor,
    EDU_ORDER,
    SENIORITY_ORDER,
    MathTransformer,
    TfidfWrapper,
    TextEncoder,
    FeatureCreation,
)

cols = ['age', 'gender', 'education_level', 'job_title', 'years_of_experience']

train_set = pd.DataFrame([
    [25, "male", "bachelor", "junior data engineer", 1],
    [30, "female", "master", "senior sales manager", 3],
    [35, "other", "phd", "project manager", 8],
], columns=cols)

train_y = np.array([70_000, 90_000, 120_000])

test_set = pd.DataFrame([
    [26, "female", "bachelor", "junior data engineer", 2],
    [28, "other", "high school", "sales manager", 6],
    [33, "male", "phd", "senior project manager", 3],
], columns=cols)

def test_preprocessor_runs():
    pre = build_preprocessor()
    train_pre = pre.fit_transform(train_set, train_y)
    assert isinstance(train_pre, np.ndarray)
    assert train_pre.shape[0] == train_set.shape[0]
    assert train_pre.shape[1] > train_set.shape[1]
    
    test_pre = pre.transform(test_set)
    print(test_pre)
    assert isinstance(test_pre, np.ndarray)
    assert test_pre.shape[0] == test_set.shape[0]
    assert test_pre.shape[1] == train_pre.shape[1]
    
def test_preprocessor_scaler():
    pre = build_preprocessor(use_scaler=True)
    train_pre = pre.fit_transform(train_set, train_y)
    assert isinstance(train_pre, np.ndarray)
    assert train_pre.shape[0] == train_set.shape[0]
    assert train_pre.shape[1] > train_set.shape[1]
    assert np.isclose(train_pre[:, 0].mean(), 0)
    assert np.isclose(train_pre[:, 0].std(), 1)
    assert train_pre[:, -2].max() > 1
    
    test_pre = pre.transform(test_set)
    print(test_pre)
    assert isinstance(test_pre, np.ndarray)
    assert test_pre.shape[0] == test_set.shape[0]
    assert test_pre.shape[1] == train_pre.shape[1]

def test_preprocessor_scaler_normalization():
    pre = build_preprocessor(use_scaler=True, scaler='minmax')
    train_pre = pre.fit_transform(train_set, train_y)
    assert isinstance(train_pre, np.ndarray)
    assert train_pre.shape[0] == train_set.shape[0]
    assert train_pre.shape[1] > train_set.shape[1]
    assert train_pre[:, 0].max() <= 1 
    assert train_pre[:, 0].min() >= 0 
    assert train_pre[:, -2].max() > 1
    
    test_pre = pre.transform(test_set)
    print(test_pre)
    assert isinstance(test_pre, np.ndarray)
    assert test_pre.shape[0] == test_set.shape[0]
    assert test_pre.shape[1] == train_pre.shape[1]