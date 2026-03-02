from pydantic import BaseModel


class AddRecordRequest(BaseModel):
    age: int
    gender: str
    education_level: str
    job_title: str
    years_of_experience: float
    salary: float

class AddRecordResponse(BaseModel):
    status: str

