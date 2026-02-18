from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # -------------------------
    # Project Root
    # -------------------------
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[5]

    # -------------------------
    # Data Directories (project-level)
    # -------------------------
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # -------------------------
    # ML Artifacts (backend-level)
    # -------------------------
    BACKEND_ROOT: Path = Path(__file__).resolve().parents[3]

    MODEL_DIR: Path = BACKEND_ROOT / "ml" / "artifacts"
    MODEL_FILE: Path = MODEL_DIR / "model.joblib"

    # -------------------------
    # Database
    # -------------------------
    DATABASE_URL: str | None = None

    SQLITE_FILE: Path = BACKEND_ROOT / "db" / "app.db"

    # -------------------------
    # CORS
    # -------------------------
    FRONTEND_ORIGINS: str = "http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        extra="ignore",
    )

    @property
    def frontend_origins_list(self) -> list[str]:
        return [o.strip() for o in self.FRONTEND_ORIGINS.split(",")]


settings = Settings()

