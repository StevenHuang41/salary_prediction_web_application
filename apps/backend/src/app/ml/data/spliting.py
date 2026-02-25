import pandas as pd
from sklearn.model_selection import train_test_split

from app.ml.features.transformers import (
    JobSeniorityTransformer,
    JobGroupTransformer,
    JobRoleTransformer,
)


def split_data(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratify_on: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df["job_seniority"] = JobSeniorityTransformer() \
                        .fit_transform(df.job_title).ravel()
    df["job_group"] = JobGroupTransformer() \
                        .fit_transform(df.job_title).ravel()
    df["job_role"] = JobRoleTransformer() \
                        .fit_transform(df.job_title).ravel()


    return train_test_split( # type: ignore
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_on,
    )
