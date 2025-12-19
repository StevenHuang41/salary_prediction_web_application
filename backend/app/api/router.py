from fastapi import APIRouter
from app.api.endpoints import data

router = APIRouter(prefix="/api")

router.include_router(data.router)

