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
    background_tasks.add_task(model_service.train, data_service.df)
    return {
        "status": "training_started"
    }

@router.put("/model/initial")
async def model_reset():
    data_size = data_service.reset()
    return {
        "status": "initialized",
        "data_size": data_size
    }

@router.get("/model/status")
async def model_status():
    return {
        "is_training": model_service.get_status()
    }

