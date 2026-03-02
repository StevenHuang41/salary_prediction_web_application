from fastapi import APIRouter

from app.api.endpoints import (
    job_title,
    prediction,
    model,
    data,
    visualization,
)

router = APIRouter()

router.include_router(job_title.router)
router.include_router(prediction.router)
router.include_router(model.router)
router.include_router(visualization.router)
router.include_router(data.router)
