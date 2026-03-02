from fastapi import APIRouter

from app.services.data_service import data_service

router = APIRouter()

@router.get("/job_titles")
def get_job_titles():
    return data_service.get_job_titles()
