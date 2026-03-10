import json
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

    def train(self, db: Session):
        self._is_training = True

        df = self.repo.get_dataframe(db)

        trainer = Trainer(df)
        trainer.run()

        self.upload_artifacts()

        self.load(db)

        self._is_training = False

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
        return self._is_training


model_service = ModelService()

if __name__ == "__main__":
    from app.db.session import SessionLocal

    db = SessionLocal()
    try :
        model_service.train(db)
    finally:
        db.close()



