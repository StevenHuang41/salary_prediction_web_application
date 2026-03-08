import os
from pathlib import Path
from pydantic.fields import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    backend_dir: Path = Path(__file__).resolve().parents[3]

    @property
    def data_dir(self) -> Path:
        return self.backend_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def raw_data_file(self) -> Path:
        return self.raw_data_dir / "salary_data.csv"


    # -------------------------
    # Artifacts (backend-level)
    # -------------------------
    @property
    def artifacts_dir(self) -> Path:
        return self.backend_dir / "artifacts"

    @property
    def model_file(self) -> Path:
        return self.artifacts_dir / "model.joblib"

    @property
    def metadata_file(self) -> Path:
        return self.artifacts_dir / "metadata.json"

    # -------------------------
    # Database
    # -------------------------
    postgres_user:      str = "postgres"
    postgres_password:  str = "postgres"
    postgres_db:        str = "salarydb"
    database_url:       str | None = None

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}"
            f":{self.postgres_password}@db:5432/{self.postgres_db}"
        )

    # -------------------------
    # CORS
    # -------------------------
    frontend_origins: str = "http://localhost:3000"

    @property
    def frontend_origins_list(self) -> list[str]:
        return [o.strip() for o in self.frontend_origins.split(",")]


settings = Settings()
