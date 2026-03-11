from pydantic import BaseModel

from app.schemas.features import SalaryFeatures


class PredictRequest(SalaryFeatures):
    pass


class PredictResponse(BaseModel):
    salary: float

    model_name: str

    mse: float
    mae: float
    rmse: float

    n_train: int
    n_test: int

    created_at: str 
    duration: str
