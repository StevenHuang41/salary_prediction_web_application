from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    backend_dir: Path = Path(__file__).resolve().parents[3]

    # -------------------------
    # Data Directories
    # -------------------------
    data_dir: Path = Path("./data")
    
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
    artifacts_dir: Path = backend_dir / "artifacts"
    model_file: Path = artifacts_dir / "model.joblib"
    metadata_file: Path = artifacts_dir / "metadata.json"

    # -------------------------
    # Database
    # -------------------------
    database_url: str | None = None

    # -------------------------
    # CORS
    # -------------------------
    frontend_origins: str = "http://localhost:3000"

    @property
    def frontend_origins_list(self) -> list[str]:
        return [o.strip() for o in self.frontend_origins.split(",")]


settings = Settings()

