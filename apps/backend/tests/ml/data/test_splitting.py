import pandas as pd
import numpy as np

from app.ml.data.spliting import split_data


# type
def test_split_data_return_type(sample_df):
    train_df, test_df = split_data(sample_df)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

# ratio
def test_split_data_ratio(sample_df):
    n_rows = sample_df.shape[0]

    _, test_df = split_data(sample_df, test_size=0.33)

    assert test_df.shape[0] == int(np.ceil(n_rows * 0.33))

    _, test_df = split_data(sample_df, test_size=0.5)

    assert test_df.shape[0] == int(np.ceil(n_rows * 0.5))

# reproducible
def test_split_data_random(sample_df):
    s1 = split_data(sample_df, random_state=42)
    s2 = split_data(sample_df, random_state=42)

    assert (s1[0].index == s2[0].index).all()
    assert (s1[1].index == s2[1].index).all()
