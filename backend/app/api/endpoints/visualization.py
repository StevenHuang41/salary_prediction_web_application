from fastapi import APIRouter, Response

from app.schemas.salary import SalaryInput
from app.db.session import get_salary_df
from my_package.data_visualization import salary_hist_image, salary_box_image

router = APIRouter()

@router.post("/salary_avxline_plot")
async def salary_hist_api(data: SalaryInput):
    df = get_salary_df()
    img = salary_hist_image(data.salary, df)
    return Response(content=img, media_type="image/png")

@router.post("/salary_boxplot")
async def salary_boxplot_api(data: SalaryInput):
    df = get_salary_df()
    img = salary_box_image(data.salary, df)
    return Response(content=img, media_type="image/png")

