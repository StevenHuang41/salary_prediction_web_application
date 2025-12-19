import os
from database.database import query_2_df

DB_FILE = os.path.join(os.getcwd(), "database", "salary_prediction.db")

def load_salary_df():
    return query_2_df("select * from salary", DB_FILE)

