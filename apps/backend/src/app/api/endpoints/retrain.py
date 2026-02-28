from fastapi import APIRouter, Depends
import pandas as pd

from app.core.config import settings
from app.schemas.salary import RowData
from app.db.dependencies import get_salary_repository
from app.db.repositories.salary_repository import SalaryRepository
from my_package.data_cleansing import cleaning_data
from my_package.data_predict import predict_salary

router = APIRouter()


@router.post("/retrain_model")
async def retrain_model_api(
    data: RowData,
    repo: SalaryRepository = Depends(get_salary_repository)
):
    df = repo.fetch_all()
    input_df = cleaning_data(pd.DataFrame([data.model_dump()]))

    return {
        "status": "success",
        "result": predict_salary(
            input_df,
            df,
            str(settings.ARTIFACTS_DIR),
            restart=True,
        ),
    }

