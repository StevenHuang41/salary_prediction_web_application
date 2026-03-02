from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.plot import PlotRequest
from app.services import data_service

router = APIRouter()

@router.post("/images/histogram")
async def post_histogram(request: PlotRequest):
    img = data_service.plot_histogram(request.salary)
    return StreamingResponse(
        img,
        media_type="image/png"
    )

@router.post("/images/boxplot")
async def post_boxplot(request: PlotRequest):
    img = data_service.plot_box(request.salary)
    return StreamingResponse(
        img,
        media_type="image/png"
    )


