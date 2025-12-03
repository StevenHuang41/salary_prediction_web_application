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
        .str.replace(r"\s+", " ")
        .str.replace(" ", "_")
    )
    
    # remove target nan
    df = df.dropna(subset=['salary'])

    # remove duplicated rows
    df = df.drop_duplicates(ignore_index=True)
    
    # handle outliers
    df['salary'] = df['salary'].astype('float64')
    df = df[(df['salary'] > lower_bound) & (df['salary'] < upper_bound)]
    
    
    return df

def clean_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ")
        .str.replace(" ", "_")
    )
    # col: age
    df.loc[:, 'age'] = df['age'].astype('int')

    # col: gender
    gen_order = ['female', 'male', 'other']
    df.loc[:, 'gender'] = df['gender'].str.lower()
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

    return df

## Whole Cleansing process
def clean_data(df: pd.DataFrame, has_target_col: bool = False) -> pd.DataFrame:
    if has_target_col:
        df = clean_target_col(df)
        
    df = clean_feature_cols(df)

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


if __name__ == "__main__":

    # test 1: column name lowercase
    # data1 = pd.DataFrame([{
    #     'Age': 20,
    #     'gender': 'Female',
    #     'education level': 'PhD',
    #     'Job title': 'Data Engineer',
    #     'years of experience': 1,
    # }])
    # print(clean_data(data1))

    # test 2
    # data2 = pd.DataFrame({
    #     'Age': [20, 19],
    #     'gender': ['Female', 'male'],
    #     'education level': ["master's degree", 'PhD'],
    #     'Job title': ['Data Engineer', 'Data Analyst'],
    #     'years of experience': [2, 1]
    # })
    # print(clean_data(data2))

    ## test 3
    # data3 = pd.DataFrame({
    #     'Age': [20, 19, 28, 18],
    #     'gender': ['Female', 'male', 'other', 'Female'],
    #     'education level': ["master's degree", 'other', 'PhD', 'Master'],
    #     'Job title': ['Data Scientist', 'Data Engineer', 'Data Analyst', 'Accountant'],
    #     'years of experience': [2, 1, 3, 10],
    #     'salary': [9_000, 300_000, 100_000, 111_000]
    # })
    # print(clean_data(data3, has_target_col=True))
    # only row2 would pass

    ## test 4: duplicated
    data4 = pd.DataFrame({
        "Age": [20, 20],
        "gender": ["Female", "Female"],
        "education level": ["PhD", "PhD"],
        "Job title": ["Data Engineer", "Data Engineer"],
        "years of experience": [1, 1],
        "salary": [100_000, 100_000],
    })
    print(clean_data(data4, has_target_col=True))
    pass