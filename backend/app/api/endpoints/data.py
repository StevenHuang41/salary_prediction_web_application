from fastapi import APIRouter
import os

from database.database import query_2_df
from my_package.data_extract_func import get_uniq_job_title

router = APIRouter()

DB_FILE = os.path.join(os.getcwd(), "database", "salary_prediction.db")

@router.get("/get_uniq_job_title")
def get_uniq_job_title_api():
    df = query_2_df("select * from salary", DB_FILE)
    return {"value": get_uniq_job_title(df)}

