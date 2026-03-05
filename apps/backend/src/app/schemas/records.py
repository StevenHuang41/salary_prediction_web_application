from pydantic import BaseModel

from app.schemas.features import SalaryFeatures


class AddRecordRequest(SalaryFeatures):
    salary: float

class AddRecordResponse(BaseModel):
    status: str

