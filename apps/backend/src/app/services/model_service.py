import json
from google.cloud.storage import bucket
import joblib
from sqlalchemy.orm import Session
from google.cloud import storage

from app.core.config import settings
from app.ml.train import Trainer
from app.repositories.salary_repository import SalaryRepository


class ModelService:
    def __init__(self):
        self.repo = SalaryRepository()
        self.model = None
        self.metadata = {}
        self._is_training = False

    def load(self, db: Session):
        if not settings.model_file.exists() or not settings.metadata_file.exists():
            print("Model files missing, start training ...")
            self.train(db)

        self.model = joblib.load(settings.model_file)

        with open(settings.metadata_file, "r") as f:
            self.metadata = json.load(f)

    def upload_artifacts(self):
        if not settings.use_cloud:
            return

        if settings.model_bucket is None:
            raise ValueError("MODEL_BUCKET is not set in .env")

        client = storage.Client()
        bucket = client.bucket(settings.model_bucket)

        bucket.blob("model.joblib").upload_from_filename(settings.model_file)
        bucket.blob("metadata.json").upload_from_filename(settings.metadata_file)

    def set_training_status(self, status: str):
        self._is_training = True if status == "training" else False

        if not settings.use_cloud:
            return

        client = storage.Client()
        bucket = client.bucket(settings.model_bucket)

        bucket.blob("status.json").upload_from_string(
            json.dumps({"status": status}),
            content_type="application/json"
        )

    def train(self, db: Session):
        self.set_training_status("training")

        df = self.repo.get_dataframe(db)

        trainer = Trainer(df)
        trainer.run()

        self.upload_artifacts()

        self.load(db)

        self.set_training_status("ready")

    def check(self, db):
        if self.model is None or self.metadata == {}:
            self.train(db)

    def predict(self, db: Session, df):
        self.check(db)

        return self.model.predict(df) # type: ignore

    def get_metadata(self, db: Session) -> dict:
        self.check(db)

        return self.metadata

    def get_status(self) -> bool:
        if not settings.use_cloud:
            return self._is_training

        client = storage.Client()
        bucket = client.bucket(settings.model_bucket)

        blob = bucket.blob("status.json")

        if not blob.exists():
            print("Warning: status.json does not exist !!!")
            return True

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



