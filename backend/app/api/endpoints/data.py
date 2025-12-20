from fastapi import APIRouter
import pandas as pd
from pathlib import Path

from app.schemas.salary import FullData
from app.db.session import load_salary_df
from database.database import insert_record, init_database, create_index
from my_package.data_cleansing import cleaning_data

router = APIRouter()

DB_FILE = str(Path.cwd() / "database" / "salary_prediction.db")

@router.post("/add_data")
def add_data_api(data: FullData):
    df = cleaning_data(
        pd.DataFrame([data.model_dump()]),
        has_target_columns=True,
    )
    record = df.to_dict(orient="records")[0]
    insert_record(record, "salary", DB_FILE)
    return {"status": "success"}

@router.post("/reset_model")
def reset_model_api():
    init_database(DB_FILE)
    create_index("job_title", "idx_job_title", db=DB_FILE)
    create_index("education_level", "idx_education_level", db=DB_FILE)
    create_index("salary", "idx_salary", db=DB_FILE)
    return {"status": "reset"}
