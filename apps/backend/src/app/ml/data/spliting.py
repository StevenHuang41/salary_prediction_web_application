import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.base import BaseEstimator, TransformerMixin

from app.ml.features.ml_transformers import JobGroupTransformer


def split_data(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    transf = JobGroupTransformer()
    df["job_group"] = transf.fit_transform(df.job_title).ravel()

    y = df["salary"]
    X = df.drop(["salary"], axis=1)

    return train_test_split( # type: ignore
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=X["job_group"]
    )
