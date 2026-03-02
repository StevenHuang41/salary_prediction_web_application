import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratify_on: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_df, test_df = train_test_split( # type: ignore
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_on,
    )
    return train_df, test_df # type: ignore
