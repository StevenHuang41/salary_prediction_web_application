from pydantic import BaseModel

class RowData(BaseModel):
    age: int
    gender: str
    education_level: str
    job_title: str
    years_of_experience: float

class FullData(RowData):
    salary: float

class SalaryInput(BaseModel):
    salary: float

