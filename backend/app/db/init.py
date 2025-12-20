from database.database import init_database, create_index
from app.core.config import DB_FILE


def init_db():
    init_database(str(DB_FILE))
    create_index("job_title", "idx_job_title", db=str(DB_FILE))
    create_index("education_level", "idx_education_level", db=str(DB_FILE))
    create_index("salary", "idx_salary", db=str(DB_FILE))

