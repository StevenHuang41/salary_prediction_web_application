from pathlib import Path
from database.database import init_database, create_index

DB_FILE = str(Path.cwd() / "database" / "salary_prediction.db")

def init_db():
    init_database(DB_FILE)
    create_index("job_title", "idx_job_title", db=DB_FILE)
    create_index("education_level", "idx_education_level", db=DB_FILE)
    create_index("salary", "idx_salary", db=DB_FILE)

