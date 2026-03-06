from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.deps import get_db
from app.services.data_service import data_service

router = APIRouter()

@router.get("/job_titles")
def get_job_titles(db: Session = Depends(get_db)):
    return data_service.get_job_titles(db)
