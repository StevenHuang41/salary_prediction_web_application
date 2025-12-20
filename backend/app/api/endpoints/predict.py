from fastapi import APIRouter, Depends
import pandas as pd

from app.core.config import settings
from app.schemas.salary import RowData
from app.db.repositories.salary_repository import SalaryRepository
from app.db.dependencies import get_salary_repository
from my_package.data_cleansing import cleaning_data
from my_package.data_predict import predict_salary

router = APIRouter()


@router.post("/predict")
async def predict_salary_api(
    data: RowData,
    repo: SalaryRepository = Depends(get_salary_repository)
):
    df = repo.fetch_all()
    input_df = cleaning_data(pd.DataFrame([data.model_dump()]))

    return predict_salary(input_df, df, str(settings.MODEL_DIR))
