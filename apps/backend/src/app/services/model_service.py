import json
from typing import Literal
import joblib
from sqlalchemy.orm import Session
import requests
from google.cloud import storage
import google.auth as gauth
import google.auth.transport.requests as gar

from app.db.session import SessionLocal
from app.core.config import settings
from app.ml.train import Trainer
from app.repositories.salary_repository import SalaryRepository


class ModelService:
    def __init__(self):
        self.repo = SalaryRepository()
        self.model = None
        self.metadata: dict = {}

        self.is_training: bool = False
        self.trainer = Trainer()


    def _download_cloud_artifacts(
        self,
        target: Literal["model", "metadata", "status", "all"] = "all",
    ):
        client = storage.Client()
        bucket = client.bucket(settings.model_bucket)

        model_blob = bucket.blob("model.joblib")
        metadata_blob = bucket.blob("metadata.json")
        status_blob = bucket.blob("status.json")

        if not model_blob.exists() \
            or not metadata_blob.exists() \
            or not status_blob.exists():
            raise FileNotFoundError("Model artifacts not found in cloud storage")

        if target == "model":
            model_blob.download_to_filename(settings.model_file)
        elif target == "metadata":
            metadata_blob.download_to_filename(settings.metadata_file)
        elif target == "status":
            status_blob.download_to_filename(settings.status_file)
        else :
            model_blob.download_to_filename(settings.model_file)
            metadata_blob.download_to_filename(settings.metadata_file)


    def _upload_cloud_artifacts(
        self,
        target: Literal["model", "metadata", "status", "all"] = "all",
    ):
        if not settings.use_cloud:
            return

        if settings.model_bucket is None:
            raise ValueError("MODEL_BUCKET is not set in .env")

        client = storage.Client()
        bucket = client.bucket(settings.model_bucket)

        if target == "model":
            bucket.blob("model.joblib").upload_from_filename(settings.model_file)
        elif target == "metadata":
            bucket.blob("metadata.json").upload_from_filename(settings.metadata_file)
        elif target == "status":
            bucket.blob("status.json").upload_from_filename(settings.status_file)
        else :
            bucket.blob("model.joblib").upload_from_filename(settings.model_file)
            bucket.blob("metadata.json").upload_from_filename(settings.metadata_file)


    def model_is_training(self) -> bool:
        if not settings.use_cloud:
            return self.is_training

        self._download_cloud_artifacts("status")

        with open(settings.status_file, "r") as f:
            status_f = json.load(f)

        return status_f["status"] == "training"


    def load_artifacts(self):
        if settings.use_cloud:
            self._download_cloud_artifacts()

        if not settings.model_file.exists() \
            or not settings.metadata_file.exists():
            raise FileNotFoundError("Model files does not exists in artifacts/")

        self.model = joblib.load(settings.model_file)

        with open(settings.metadata_file, "r") as f:
            self.metadata = json.load(f)


    def predict(self, df):
        if self.model is None:
            self.load_artifacts()

        return self.model.predict(df) # type: ignore


    def _set_training_status(self, status: str):
        self.is_training = True if status == "training" else False

        with open(settings.status_file, "w") as f:
            json.dump({"status": status}, f, indent=4)

        if settings.use_cloud:
            self._upload_cloud_artifacts("status")


    def _run_training_job(self, db: Session):
        df = self.repo.get_dataframe(db)

        self.trainer.load_data(df)
        self.trainer.run()

        self._upload_cloud_artifacts()

        self._set_training_status("ready")


    def _call_training_api(self):
        url = (
            f"https://run.googleapis.com/v2/projects/"
            f"{settings.gcp_project}/locations/asia-east1/jobs/"
            f"{settings.training_job_name}:run"
        )

        credentials, _ = gauth.default()
        auth_req = gar.Request()
        credentials.refresh(auth_req)

        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, headers=headers)

        if response.status_code not in (200, 201):
            raise RuntimeError(response.text)


    def train(self, db: Session):
        self._set_training_status("training")
        
        if not settings.use_cloud or settings.i_am == "jobrun":
            self._run_training_job(db)
        else :
            self._call_training_api()


model_service = ModelService()

if __name__ == "__main__":
    from app.db.session import SessionLocal

    db = SessionLocal()
    try :
        model_service.train(db)
    finally:
        db.close()



