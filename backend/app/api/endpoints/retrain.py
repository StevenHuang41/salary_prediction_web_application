from fastapi import APIRouter
import pandas as pd

from app.core.config import settings
from app.schemas.salary import RowData
from app.db.repositories.salary_repository import SalaryRepository
from my_package.data_cleansing import cleaning_data
from my_package.data_predict import predict_salary

router = APIRouter()
repo = SalaryRepository()


@router.post("/retrain_model")
async def retrain_model_api(data: RowData):
    df = repo.fetch_all()
    input_df = cleaning_data(pd.DataFrame([data.model_dump()]))

    return {
        "status": "success",
        "result": predict_salary(
            input_df,
            df,
            str(settings.MODEL_DIR),
            restart=True,
        ),
    }

