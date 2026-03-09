from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.router import router as api_router
from app.db import init_db
from app.core.config import settings
from app.db.session import SessionLocal
from app.repositories.salary_repository import SalaryRepository
from app.services.data_service import data_service
from app.services.model_service import model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    try :
        init_db()
        db = SessionLocal()
        repo = SalaryRepository()

        if repo.count(db) == 0:
            data_service.seed(db)

        model_service.load(db)
        data_service.load(db)

        db.close()
    except Exception as e:
        print("Startup DB init skipped:", e)

    yield

def create_app() -> FastAPI:
    app = FastAPI(
        title="Salary Prediction API",
        lifespan=lifespan,
    )

    print("Frontend Origins:", settings.frontend_origins_list)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            *(settings.frontend_origins_list or []),
            "https://storage.googleapis.com",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()

@app.get("/health")
def health_check():
    return {"status": "ok"}

