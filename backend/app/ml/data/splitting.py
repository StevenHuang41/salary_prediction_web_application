from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return train_test_split(X, y,
                            test_size=test_size,
                            shuffle=shuffle,
                            random_state=random_state)