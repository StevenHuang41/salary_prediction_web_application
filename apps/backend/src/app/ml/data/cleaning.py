import pandas as pd

from app.ml.features.transformers import (
    JobSeniorityTransformer,
    JobGroupTransformer,
    JobRoleTransformer,
)


# rename column
def rename_cols(df: pd.DataFrame) -> None:
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r'\s+', '_', regex=True)
        .str.lower()
    )

# col: salary
def clean_salary(
    df: pd.DataFrame,
    lower_bound=10000,
    upper_bound=300000,
) -> pd.DataFrame:
    df = df.dropna(subset=["salary"]).reset_index(drop=True).copy()
    df["salary"] = df["salary"].astype("float64")
    df = df.loc[(df["salary"] > lower_bound) & (df["salary"] < upper_bound)]
    return df

def clean_data(
    df: pd.DataFrame,
    has_target_col: bool = False,
    **kws,
) -> pd.DataFrame:
    df = df.copy()

    # rename column names
    rename_cols(df)

    # col: salary
    if has_target_col:
        df = clean_salary(df, **kws)

    # col: age
    df["age"] = df["age"].astype('float32')

    # col: gender
    gender_order = ["female", "male", "other"]
    df["gender"] = df["gender"].str.lower()
    df["gender"] = pd.Categorical(
        df["gender"],
        categories=gender_order,
        ordered=True
    )

    # col: education level
    edu_order = ["unknown", "high_school", "bachelor", "master", "phd"]
    df["education_level"] = (
        df["education_level"]
        .str.lower()
        .replace({
            r'bach.*': 'bachelor',
            r'mas.*': 'master',
            r'hig.*': 'high_school',
            r'doc.*|phd': 'phd',
        }, regex=True)
        .fillna("unknown")
    )
    df["education_level"] = pd.Categorical(
        df["education_level"],
        categories=edu_order,
        ordered=True,
    )

    # col: job title
    df["job_title"] = (
        df["job_title"]
        .str.lower()
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace('juniour', 'junior', regex=True)
        .str.replace(r'rep\b', 'representative', regex=True)
        .str.replace(r'\bman\b', 'manager', regex=True)
        .str.replace(r'director of (.*)', r'\1 director', regex=True)
        .str.replace(r'vp of (.*)', r'\1 vp', regex=True)
    )
    df["job_title"] = pd.Categorical(
        df["job_title"],
        df["job_title"].value_counts().index,
        ordered=True,
    )

    # col: years of experience
    df["years_of_experience"] = df["years_of_experience"].astype('float32')


    # remove incorrect age-experience rows
    df = df.loc[(df["age"] - df["years_of_experience"]) >= 18]

    # drop duplicated
    df = df.drop_duplicates(ignore_index=True)


    # seniority
    df["job_seniority"] = JobSeniorityTransformer() \
                        .fit_transform(df["job_title"]).ravel()
    # group
    df["job_group"] = JobGroupTransformer() \
                        .fit_transform(df["job_title"]).ravel()
    # role
    df["job_role"] = JobRoleTransformer() \
                        .fit_transform(df["job_title"]).ravel()

    # remove prefix seniority in job_title
    df["job_title"] = (
        df["job_title"]
        .str.replace(r'\b(junior|senior)\b', "", regex=True)
        .str.strip()
    )

    return df
