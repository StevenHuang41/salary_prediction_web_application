import json
import joblib
from app.core.config import settings
from app.ml.train import Trainer


class ModelService:
    def __init__(self):
        self.model = None
        self.metadata = {}
        self._is_training = False

    def load(self):
        self.model = joblib.load(settings.MODEL_FILE)

        with open(settings.METADATA_FILE, "r") as f:
            self.metadata = json.load(f)

    def retrain(self, df=None):
        self._is_training = True
        trainer = Trainer(df)
        trainer.run()
        self.load()
        self._is_training = False

    def check(self):
        if self.model is None or self.metadata == {}:
            self.retrain()

    def predict(self, df):
        self.check()

        return self.model.predict(df) # type: ignore

    def get_metadata(self) -> dict:
        self.check()

        return self.metadata

    def get_status(self) -> bool:
        return self._is_training


model_service = ModelService()
