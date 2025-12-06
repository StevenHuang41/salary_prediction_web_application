import pandas as pd

from app.ml.data.splitting import split_data
from app.ml.config import DF_COLS

def sample_df():
    return pd.DataFrame([
        [20, 'female', 'high school', 'data analyst', 2, 60_000],
        [22, 'male', 'phd', 'data scientist', 1, 70_000],
        [24, 'female', 'unknown', 'data engineer', 4, 90_000],
        [21, 'other', 'bachelor', 'accountant', 0, 50_000],

        [25, 'male', 'master', 'marketing manager', 3, 90_000],
        [32, 'female', 'bachelor', 'software engineer', 10, 160_000],
        [30, 'other', 'phd', 'senior software engineer', 11, 180_000],
        [43, 'female', 'high school', 'it support', 24, 120_000],
    ], columns=DF_COLS)

def test_split_shapes(df=sample_df()):
    X_train, X_test, y_train, y_test = split_data(df)
    # 8 samples * 0.2 = 2 samples for test
    assert len(X_train) == 6
    assert len(y_train) == 6
    assert len(X_test) == 2
    assert len(y_test) == 2

def test_split_consistent(df=sample_df()):
    t1, _, y1, _ = split_data(df, random_state=34)
    t2, _, y2, _ = split_data(df, random_state=34)
    assert (t1 == t2).all().all()
    assert (y1 == y2).all()