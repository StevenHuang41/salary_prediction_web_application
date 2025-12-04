import os
import shutil
import pandas as pd
import numpy as np

def clean_target_col(
    df: pd.DataFrame,
    lower_bound=10000,
    upper_bound=300000,
) -> pd.DataFrame:
    # rename column
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(" ", "_")
    )
    
    # remove target nan
    df = df.dropna(subset=['salary'])

    # remove duplicated rows
    df = df.drop_duplicates(ignore_index=True)
    
    # handle outliers
    df.loc[:, 'salary'] = df['salary'].astype('float64')
    df = df[(df['salary'] >= lower_bound) & (df['salary'] <= upper_bound)]
    
    return df

def clean_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(" ", "_")
    )
    # col: age
    df.loc[:, 'age'] = df['age'].astype('int')

    # col: gender
    gen_order = ['female', 'male', 'other']
    df.loc[:, 'gender'] = (
        df['gender']
        .str.lower()
        .str.strip()
        .fillna("other")
        .replace({
            r'^fe.*': 'female',
            r'^m.*': 'male',
            r'^other$': 'other',
        }, regex=True)
    )
    
    df.loc[:, 'gender'] = pd.Categorical(
        df['gender'],
        categories=gen_order,
        ordered=True
    )

    # col: education level
    df.loc[:, 'education_level'] = (
        df['education_level']
        .str.lower()
        .replace({
            'bache.*': 'bachelor',
            'mast.*': 'master',
            'phd|doctor': 'phd',
            'high.*': 'high school',
            'other': 'unknown',
            np.nan: 'unknown',
        }, regex=True)
    )

    # col: job title
    df.loc[:, 'job_title'] = (
        df.job_title
        .str.strip()
        .str.lower()
        .str.replace('juniour', 'junior', regex=True)
        .str.replace(r'rep\b', 'representative', regex=True)
        .str.replace(r'\bman\b', 'manager', regex=True)
        .str.replace(r'director of (.*)$', r'\1 director', regex=True)
    )

    # col: years of experience
    df.loc[:, 'years_of_experience'] = df['years_of_experience'].astype('float32')

    # filter: age and years of experience validity
    df = df[(df['age'] - df['years_of_experience']) >= 18]
    
    # filter: not accept education_level having 'unknown'
    df = df[~(df.education_level == 'unknown')]

    return df

## Whole Cleansing process
def clean_data(df: pd.DataFrame, has_target_col: bool = False) -> pd.DataFrame:
    df = clean_feature_cols(df)

    if has_target_col:
        df = clean_target_col(df)

    return df


# ## remove every file in a dir
# def clean_directory(dir_path: str) -> None:
#     for file in os.listdir(dir_path):
#         file_path = os.path.join(dir_path, file)
#         try :
#             if not os.path.isdir(file_path):
#                 os.remove(file_path)
#             else :
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print(f"Failed to delete {file_path}, {e}")

