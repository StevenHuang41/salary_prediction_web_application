from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.db.deps import get_db
from app.services import data_service
from app.services.model_service import model_service


router = APIRouter()

@router.put("/model/training")
async def model_retrain(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    if model_service.is_training:
        return {
            "status": "already_training"
        }
    background_tasks.add_task(model_service.train, db)
    return {
        "status": "training_started"
    }

@router.put("/model/initial")
async def model_reset(db: Session = Depends(get_db)):
    data_size = data_service.reset_to_default(db)
    return {
        "status": "initialized",
        "data_size": data_size
    }

@router.get("/model/status")
async def model_status():
    return {
        "is_training": model_service.model_is_training()
    }

@router.put("/model/data-sync")
async def model_sync():
    model_service.load_artifacts()
    model_service.is_training = False
    return {
        "status": "ok",
    }
