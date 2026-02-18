from fastapi import APIRouter

from app.api.endpoints.job_title import router as job_title_router
from app.api.endpoints.predict import router as predict_router
from app.api.endpoints.retrain import router as retrain_router
from app.api.endpoints.data import router as data_router
from app.api.endpoints.visualization import router as viz_router

router = APIRouter(prefix="/api")

router.include_router(job_title_router)
router.include_router(predict_router)
router.include_router(retrain_router)
router.include_router(data_router)
router.include_router(viz_router)

