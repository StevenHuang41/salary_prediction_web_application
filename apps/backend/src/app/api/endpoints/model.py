from fastapi import APIRouter, BackgroundTasks

from app.services import data_service
from app.services.model_service import model_service


router = APIRouter()

@router.put("/model/training")
async def model_retrain(background_tasks: BackgroundTasks):
    if model_service.get_status():
        return {
            "status": "already_training"
        }
    background_tasks.add_task(model_service.retrain, data_service.df)
    return {
        "status": "training_started"
    }

@router.put("/model/initial")
async def model_reset(background_tasks: BackgroundTasks):
    data_service.reset()
    background_tasks.add_task(model_service.retrain)
    return {
        "status": "initialization_started"
    }

@router.get("/model/status")
async def model_status():
    return {
        "is_training": model_service.get_status()
    }
