import pandas as pd
from fastapi import APIRouter

from app.ml.data.cleaning import clean_data
from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.model_service import model_service


router = APIRouter()

@router.post("/predictions", response_model=PredictResponse)
async def predict(request: PredictRequest):
    df = pd.DataFrame([request.model_dump()])
    df = clean_data(df)

    salary = model_service.predict(df)[0]
    metadata = model_service.metadata

    return PredictResponse(
        salary=float(salary),
        model_name=str(metadata["model_name"]),
        mse=float(metadata["mse"]),
        mae=float(metadata["mae"]),
        rmse=float(metadata["rmse"]),
        n_train=int(metadata["n_train"]),
        n_test=int(metadata["n_test"]),
        created_at=str(metadata["created_at"]),
        duration=str(metadata["duration"]),
    )
