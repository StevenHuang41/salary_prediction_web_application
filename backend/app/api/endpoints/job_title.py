from fastapi import APIRouter
from app.db.repositories.salary_repository import SalaryRepository
from my_package.data_extract_func import get_uniq_job_title

router = APIRouter()
repo = SalaryRepository()

@router.get("/get_uniq_job_title")
def get_uniq_job_title_api():
    df = repo.fetch_all()
    return {"value": get_uniq_job_title(df)}

