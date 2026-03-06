from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.deps import get_db
from app.schemas.records import AddRecordRequest, AddRecordResponse
from app.services import data_service

router = APIRouter()

@router.post("/records", response_model=AddRecordResponse)
def add_data_api(
    data: AddRecordRequest,
    db: Session = Depends(get_db)
):
    id = data_service.add_record(db, data.model_dump())
    return AddRecordResponse(
        status="success",
        id=id
    )
