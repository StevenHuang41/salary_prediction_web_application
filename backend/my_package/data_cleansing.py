import os
import shutil
import pandas as pd
import numpy as np

## rename column
def cleaning_rename_cols(df: pd.DataFrame) -> None:
    df.columns = [col.replace(' ', '_').lower() for col in df.columns]
    # print("Data Cleansing: rename columns - Successful")


## col: salary
def cleaning_NaN_salary(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=['salary']).reset_index(drop=True)

def cleaning_remove_salary_outlier(
    df: pd.DataFrame,
    lower_bound=10000,
    upper_bound=300000,
) -> pd.DataFrame:
    df['salary'] = df['salary'].astype('float64')
    df = df[(df['salary'] > lower_bound) &
            (df['salary'] < upper_bound)]
    return df

def cleaning_salary(df: pd.DataFrame) -> pd.DataFrame:
    df = cleaning_NaN_salary(df)
    # print("Data Cleansing: clean NAN salary value - Successful")
    df = cleaning_remove_salary_outlier(df)
    # print("Data Cleansing: clean salary outlier - Successful")
    return df

## remove nan
def cleaning_NaN(df: pd.DataFrame) -> None:
    df.dropna(inplace=True)

## remove duplicated
def cleaning_duplicated(df: pd.DataFrame) -> None:
    df.drop_duplicates(inplace=True)

## col: age
def cleaning_age(df: pd.DataFrame) -> None:
    df['age'] = df['age'].astype('int32')
    # print("Data Cleansing: cleaning age - Successful")


## col: gender
def cleaning_gender(df: pd.DataFrame) -> None:
    gender_order = ['female', 'male', 'other']
    mapping = {
        'Male': 'male',
        'Female': 'female',
        'Other': 'other'
    }
    df['gender'] = df['gender'] \
                    .map(mapping) \
                    .fillna(df['gender'])

    df['gender'] = pd.Categorical(
        df['gender'],
        categories=gender_order,
        ordered=True
    )
    # print("Data Cleansing: cleaning gender - Successful")


## col: education level
def cleaning_edu(df: pd.DataFrame) -> None:
    education_level_str = df['education_level'].str.lower()
    df['education_level'] = np.select(
        condlist=[
            education_level_str.str.contains('bachelor', na=False),
            education_level_str.str.contains('master', na=False),
            education_level_str.str.contains('phd', na=False),
            education_level_str.str.contains('high school', na=False),
        ],
        choicelist=[
            'Bachelor',
            'Master',
            'PhD',
            'High School'
        ],
        default='No Specified',
    )

    edu_order = ['No Specified', 'High School', 'Bachelor', 'Master', 'PhD']

    df['education_level'] = pd.Categorical(
        df['education_level'],
        categories=edu_order, 
        ordered=True
    )
    # print("Data Cleansing: cleaning education level - Successful")


## col: job title
# removing prefix does not improve model performance
# so this function actually does not change anything
def cleaning_job(df: pd.DataFrame) -> None:
    df['job_title'] = (
        df['job_title']
        # .str
        # .replace(r'\b(Junior|Juniour|Senior)\b\s+', '', regex=True)
        .str.strip()
        .astype('str')
    )
    # print("Data Cleansing: cleaning job title - Successful")


## col: years of experience
def cleaning_exp(df: pd.DataFrame) -> None:
    df['years_of_experience'] = df['years_of_experience'].astype('float32')
    # print("Data Cleansing: cleaning years of experience - Successful")

## check age and years of experience validity
def check_age_yearE(df: pd.DataFrame) -> None:
    return df[(df['age'] - df['years_of_experience']) >= 18] 

## Whole Cleansing process
def cleaning_data(df: pd.DataFrame,
                  has_target_columns: bool = False) -> pd.DataFrame:
    cleaning_rename_cols(df)
    if has_target_columns:
        df = cleaning_salary(df)
    else :
        cleaning_NaN(df)
        cleaning_duplicated(df)
    cleaning_age(df)
    cleaning_gender(df)
    cleaning_edu(df)
    cleaning_job(df)
    cleaning_exp(df)
    df = check_age_yearE(df)

    # print("... Finishing Cleansing Process ...", end='\n\n')

    return df


## remove every file in a dir
def clean_directory(dir_path: str) -> None:
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        try :
            if not os.path.isdir(file_path):
                os.remove(file_path)
            else :
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}, {e}")


if __name__ == "__main__":

    ## test 1
    # data1 = pd.DataFrame([{
    #     'Age': 20,
    #     'gender': 'Female',
    #     'education level': 'PhD',
    #     'Job title': 'Data Engineer',
    #     'years of experience': 1,
    # }])
    # print(cleaning_data(data1))

    # test 2
    # data2 = pd.DataFrame({
    #     'Age': [20, 19],
    #     'gender': ['Female', 'male'],
    #     'education level': ["master's degree", 'PhD'],
    #     'Job title': ['Data Engineer', 'Data Analyst'],
    #     'years of experience': [2, 1]
    # })
    # print(cleaning_data(data2))

    ## test 3
    # data3 = pd.DataFrame({
    #     'Age': [20, 19, 28],
    #     'gender': ['Female', 'male', 'other'],
    #     'education level': ["master's degree", 'other', 'PhD'],
    #     'Job title': ['Data Scientist', 'Data Engineer', 'Data Analyst'],
    #     'years of experience': [2, 1, 3],
    #     'salary': [9_000, 300_000, 100_000]
    # })
    # print(cleaning_data(data3, has_target_columns=True))

    ## test 4
    # duplicated
    # data1 = pd.DataFrame({
    #     "Age": [20, 20],
    #     "gender": ["Female", "Female"],
    #     "education level": ["PhD", "PhD"],
    #     "Job title": ["Data Engineer", "Data Engineer"],
    #     "years of experience": [1, 1],
    # })
    # print(cleaning_data(data1))
    pass