from fastapi import APIRouter
import pandas as pd

from app.core.config import settings
from app.schemas.salary import RowData
from app.db.session import load_salary_df
from my_package.data_cleansing import cleaning_data
from my_package.data_predict import predict_salary

router = APIRouter()


@router.post("/predict")
async def predict_salary_api(data: RowData):
    df = load_salary_df()
    input_df = cleaning_data(pd.DataFrame([data.model_dump()]))

    return predict_salary(input_df, df, str(settings.MODEL_DIR))
