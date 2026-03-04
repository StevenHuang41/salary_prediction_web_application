from fastapi import APIRouter

from app.schemas.records import AddRecordRequest, AddRecordResponse
from app.services import data_service

router = APIRouter()

@router.post("/records", response_model=AddRecordResponse)
def add_data_api(data: AddRecordRequest):
    n_df_size = data_service.add_record(data.model_dump())
    return AddRecordResponse(
        status="success",
    )
