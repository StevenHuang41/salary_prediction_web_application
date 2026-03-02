from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # -------------------------
    # Project Root
    # -------------------------
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[5]
    BACKEND_DIR: Path = PROJECT_ROOT / "apps" / "backend"

    # -------------------------
    # Data Directories (project-level)
    # -------------------------
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    RAW_DATA_FILE: Path = RAW_DATA_DIR / "salary_data.csv"

    # -------------------------
    # Artifacts (backend-level)
    # -------------------------
    ARTIFACTS_DIR: Path = BACKEND_DIR / "artifacts"
    MODEL_FILE: Path = ARTIFACTS_DIR / "model.joblib"
    METADATA_FILE: Path = ARTIFACTS_DIR / "metadata.json"

    # -------------------------
    # Database
    # -------------------------
    DATABASE_URL: str | None = None

    SQLITE_FILE: Path = BACKEND_DIR / "app" / "db" / "app.db"

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

