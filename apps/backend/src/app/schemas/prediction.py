from pydantic import BaseModel

class PredictRequest(BaseModel):
    age: int
    gender: str
    education_level: str
    job_title: str
    years_of_experience: float

class PredictResponse(BaseModel):
    salary: float
    model_name: str
    mse: float
    mae: float
    rmse: float
    n_train: int
    n_test: int

# class FullData(RowData):
#     salary: float
#
# class SalaryPrediction(BaseModel):
#     model_name: str
#     use_polynomial: bool
#     value: float
#     params: dict
#
#
#
