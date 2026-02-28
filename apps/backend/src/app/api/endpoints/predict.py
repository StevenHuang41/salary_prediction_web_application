from fastapi import APIRouter, Depends
import pandas as pd

from app.core.config import settings
from app.schemas.salary import RowData, SalaryPrediction
from app.db.repositories.salary_repository import SalaryRepository
from app.db.dependencies import get_salary_repository
from my_package.data_cleansing import cleaning_data
from my_package.data_predict import predict_salary

router = APIRouter()


@router.post("/predict", response_model=SalaryPrediction)
async def predict_salary_api(
    data: RowData,
    repo: SalaryRepository = Depends(get_salary_repository)
):
    df = repo.fetch_all()
    input_df = cleaning_data(pd.DataFrame([data.model_dump()]))

    result = predict_salary(input_df, df, str(settings.ARTIFACTS_DIR))
    return {
        "model_name": result['model_name'],
        "use_polynomial": result['use_polynomial'],
        "value": result['value'],
        "num_train_dataset": result['num_train_dataset'],
        "num_test_dataset": result['num_test_dataset'],
        "params": result['params'],
    }

