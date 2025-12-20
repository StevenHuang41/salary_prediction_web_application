from fastapi import APIRouter, Response

from app.schemas.salary import SalaryInput
from app.db.repositories.salary_repository import SalaryRepository
from my_package.data_visualization import salary_hist_image, salary_box_image

router = APIRouter()
repo = SalaryRepository()

@router.post("/salary_avxline_plot")
async def salary_hist_api(data: SalaryInput):
    df = repo.fetch_all()
    img = salary_hist_image(data.salary, df)
    return Response(content=img, media_type="image/png")

@router.post("/salary_boxplot")
async def salary_boxplot_api(data: SalaryInput):
    df = repo.fetch_all()
    img = salary_box_image(data.salary, df)
    return Response(content=img, media_type="image/png")

