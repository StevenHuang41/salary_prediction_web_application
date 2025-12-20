from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    BASE_DIR: Path = Path("/backend")

    DATABASE_DIR: Path = BASE_DIR / "database"

    MODEL_DIR: Path = BASE_DIR / "best_performance"

    DB_FILE: Path = DATABASE_DIR / "salary_prediction.db"

    CSV_FILE: Path = DATABASE_DIR / "salary_data.csv"

    FRONTEND_ORIGIN: str = "http://localhost:3000"

    class Config:
        env_file = ".env"

settings = Settings()
