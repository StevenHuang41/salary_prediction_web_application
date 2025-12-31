from fastapi import APIRouter, Depends
from app.db.repositories.salary_repository import SalaryRepository
from app.db.dependencies import get_salary_repository
from my_package.data_extract_func import get_uniq_job_title

router = APIRouter()

@router.get("/uniq_job_title")
def get_uniq_job_title_api(
    repo: SalaryRepository = Depends(get_salary_repository)
):
    df = repo.fetch_all()
    return {"value": get_uniq_job_title(df)}

