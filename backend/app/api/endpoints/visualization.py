from fastapi import APIRouter, Response, Depends

from app.schemas.salary import SalaryPrediction
from app.db.repositories.salary_repository import SalaryRepository
from app.db.dependencies import get_salary_repository
from my_package.data_visualization import salary_hist_image, salary_box_image

router = APIRouter()

@router.post("/salary_avxline_plot")
async def salary_hist_api(
    data: SalaryPrediction,
    repo: SalaryRepository = Depends(get_salary_repository)
):
    df = repo.fetch_all()
    img = salary_hist_image(data.salary, df)
    return Response(content=img, media_type="image/png")

@router.post("/salary_boxplot")
async def salary_boxplot_api(
    data: SalaryPrediction,
    repo: SalaryRepository = Depends(get_salary_repository)
):
    df = repo.fetch_all()
    img = salary_box_image(data.salary, df)
    return Response(content=img, media_type="image/png")

