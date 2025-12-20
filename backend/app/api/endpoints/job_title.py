from fastapi import APIRouter
from app.db.session import get_salary_df
from my_package.data_extract_func import get_uniq_job_title

router = APIRouter()

@router.get("/get_uniq_job_title")
def get_uniq_job_title_api():
    df = get_salary_df()
    return {"value": get_uniq_job_title(df)}

