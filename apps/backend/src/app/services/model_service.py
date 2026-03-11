import json
import joblib
from sqlalchemy.orm import Session
from google.cloud import storage
import requests
import google.auth as gauth
import google.auth.transport.requests as gar

from app.core.config import settings
from app.ml.train import Trainer
from app.repositories.salary_repository import SalaryRepository


class ModelService:
    def __init__(self):
        self.repo = SalaryRepository()
        self.model = None
        self.metadata = None
        self._is_training = False


    def _download_cloud_artifacts(self):
        client = storage.Client()
        bucket = client.bucket(settings.model_bucket)

        model_blob = bucket.blob("model.joblib")
        metadata_blob = bucket.blob("metadata.json")

        if not model_blob.exists() or not metadata_blob.exists():
            raise FileNotFoundError("Model artifacts not found in cloud storage")

        model_blob.download_to_filename(settings.model_file)
        metadata_blob.download_to_filename(settings.metadata_file)


    def _upload_cloud_artifacts(self):
        if not settings.use_cloud:
            return

        if settings.model_bucket is None:
            raise ValueError("MODEL_BUCKET is not set in .env")

        client = storage.Client()
        bucket = client.bucket(settings.model_bucket)

        bucket.blob("model.joblib").upload_from_filename(settings.model_file)
        bucket.blob("metadata.json").upload_from_filename(settings.metadata_file)


    def load(self):
        if settings.use_cloud:
            self._download_cloud_artifacts()

        if not settings.model_file.exists() or not settings.metadata_file.exists():
            raise FileNotFoundError("Model files does not exists in artifacts/")

        self.model = joblib.load(settings.model_file)

        with open(settings.metadata_file, "r") as f:
            self.metadata = json.load(f)


    def _check(self):
        if self.model is None or self.metadata is None:
            self.load()

    def _clear_old_model(self):
        self.model = None
        self.metadata = None


    def get_metadata(self) -> dict:
        self._check()

        return self.metadata # type: ignore


    def predict(self, df):
        self._check()

        return self.model.predict(df) # type: ignore


    def _set_training_status(self, status: str):
        self._is_training = True if status == "training" else False

        if not settings.use_cloud:
            return

        client = storage.Client()
        bucket = client.bucket(settings.model_bucket)

        bucket.blob("status.json").upload_from_string(
            json.dumps({"status": status}),
            content_type="application/json"
        )


    def _run_training_job(self, db: Session):
        df = self.repo.get_dataframe(db)

        trainer = Trainer(df)
        trainer.run()

        self._upload_cloud_artifacts()

        self._set_training_status("ready")


    def _call_training_api(self):
        self._set_training_status("training")

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

        if settings.i_am == "jobrun":
            self._clear_old_model()
            self._run_training_job(db)

        else :
            self._call_training_api()


    def model_is_training(self) -> bool:
        if not settings.use_cloud:
            return self._is_training

        client = storage.Client()
        bucket = client.bucket(settings.model_bucket)

        blob = bucket.blob("status.json")

        if not blob.exists():
            print("Warning: status.json does not exist !!!")
            return False

        status_f = json.loads(blob.download_as_string())
        return status_f["status"] == "training"




model_service = ModelService()

if __name__ == "__main__":
    from app.db.session import SessionLocal

    db = SessionLocal()
    try :
        model_service.train(db)
    finally:
        db.close()



