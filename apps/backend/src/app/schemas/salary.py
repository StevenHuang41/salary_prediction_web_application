from pydantic import BaseModel

class RowData(BaseModel):
    age: int
    gender: str
    education_level: str
    job_title: str
    years_of_experience: float

class FullData(RowData):
    salary: float

class SalaryPrediction(BaseModel):
    model_name: str
    use_polynomial: bool 
    value: float
    num_train_dataset: int
    num_test_dataset: int
    params: dict


class SalaryValue(BaseModel):
    salary: float

