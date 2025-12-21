from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    BASE_DIR: Path = Path(__file__).resolve().parents[2]

    DATABASE_DIR: Path = BASE_DIR / "app" / "db"

    MODEL_DIR: Path = BASE_DIR / "ml" / "best_performance"

    DB_FILE: Path = DATABASE_DIR / "salary_prediction.db"

    CSV_FILE: Path = DATABASE_DIR / "salary_data.csv"

    FRONTEND_ORIGINS: str = "http://localhost:3000"

    class Config:
        env_file = (".env", ".env.local")
        extra = "ignore"

    @property
    def frontend_origins_list(self) -> list[str]:
        return [o.strip() for o in self.FRONTEND_ORIGINS.split(",")]

settings = Settings()
