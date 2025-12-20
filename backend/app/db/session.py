from app.core.config import DB_FILE
from database.database import query_2_df

def load_salary_df():
    return query_2_df("select * from salary", str(DB_FILE))

