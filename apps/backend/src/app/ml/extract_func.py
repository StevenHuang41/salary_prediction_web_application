import pandas as pd
import numpy as np
import sqlite3

## sql
# def get_uniq_job_title(db: str='salary_prediction.db',
#                        table: str='salary',
#                        col: str='job_title') -> list[str]:
#     with sqlite3.connect(db) as conn:
#         c = conn.cursor()

#         c.execute(f"""
#             select distinct {col} from {table}
#             order by {col};
#         """)

#         result = [row[0] for row in c]

#     return result

## dataframe
def get_uniq_job_title(df: pd.DataFrame) -> list[str]:
    result = np.sort(df['job_title'].unique())
    return result.tolist()
    
if __name__ == "__main__":

    ## test 1
    # data = pd.DataFrame({
    #     'Age': [20, 19, 28],
    #     'gender': ['Female', 'male', 'other'],
    #     'education level': ["master's degree", 'other', 'PhD'],
    #     'Job Title': ['Data Scientist', 'Data Engineer', 'Data Analyst'],
    #     'years of experience': [2, 1, 3],
    #     'salary': [9_000, 300_000, 100_000]
    # })
    # print(get_uniq_job_title(data))

    ## test 2
    import os

    root_dir_path = os.getcwd().split('/backend')[0]
    backend_dir_path = os.path.join(root_dir_path, 'backend')
    database_dir_path = os.path.join(backend_dir_path, 'database')
    db_file_path = os.path.join(database_dir_path, 'salary_prediction.db')

    print(get_uniq_job_title(db=db_file_path))


    pass