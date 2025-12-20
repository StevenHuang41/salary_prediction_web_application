from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.router import router as api_router
from app.db.init import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

def create_app() -> FastAPI:
    app = FastAPI(
        title="Salary Prediction API",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app


app = create_app()

