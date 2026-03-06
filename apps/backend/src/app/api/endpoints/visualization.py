from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.deps import get_db
from app.schemas.plot import PlotRequest
from app.services import data_service

router = APIRouter()

@router.post("/images/histogram")
async def post_histogram(
    request: PlotRequest,
    db: Session = Depends(get_db)
):
    img = data_service.plot_histogram(db, request.salary)
    return StreamingResponse(
        img,
        media_type="image/png"
    )

@router.post("/images/boxplot")
async def post_boxplot(
    request: PlotRequest,
    db: Session = Depends(get_db)
):
    img = data_service.plot_box(db, request.salary)
    return StreamingResponse(
        img,
        media_type="image/png"
    )
