from fastapi import APIRouter
import pandas as pd

from app.schemas.salary import FullData
# from app.db.session import load_salary_df
from database.database import insert_record, init_database, create_index
from my_package.data_cleansing import cleaning_data
from app.core.config import settings

router = APIRouter()


@router.post("/add_data")
def add_data_api(data: FullData):
    df = cleaning_data(
        pd.DataFrame([data.model_dump()]),
        has_target_columns=True,
    )
    record = df.to_dict(orient="records")[0]
    insert_record(record, "salary", str(settings.DB_FILE))
    return {"status": "success"}

@router.post("/reset_model")
def reset_model_api():
    init_database(str(settings.DB_FILE))
    create_index("job_title", "idx_job_title", db=str(settings.DB_FILE))
    create_index("education_level", "idx_education_level", db=str(settings.DB_FILE))
    create_index("salary", "idx_salary", db=str(settings.DB_FILE))
    return {"status": "success"}
