import json
import joblib
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_squared_error,
)

from app.core.config import settings
from app.ml.data import clean_data, split_data
from app.ml.models import MODEL_REGISTRY
from app.ml.train import compare_models
from app.ml.tune import tune_models


class Trainer:
    def __init__(self):
        self.X_train: pd.DataFrame
        self.y_train: pd.Series
        self.X_test: pd.DataFrame
        self.y_test: pd.Series

        self.model = None,
        self.best_model_name: str = ""
        self.best_params: dict = {}
        self.mse: float = 0
        self.mae: float = 0
        self.rmse: float = 0
        self.n_train: int = 0
        self.n_test: int = 0


    def load_data(self, df: pd.DataFrame | None = None):
        if df is None or df.empty:
            df = pd.read_csv(settings.raw_data_file)

        df = clean_data(df, has_target_col=True)
        train_df, test_df = split_data(df)

        self.X_train = train_df.drop("salary", axis=1)
        self.y_train = train_df["salary"] # type: ignore
        self.n_train = len(self.X_train)

        self.X_test = test_df.drop("salary", axis=1)
        self.y_test = test_df["salary"] # type: ignore
        self.n_test = len(self.X_test)

        return (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        )


    def train(self, n_trial: int = 10):
        if self.X_train is None:
            self.load_data()

        # compare
        self.best_model_name, _ = compare_models(
            MODEL_REGISTRY,
            self.X_train,
            self.y_train,
        )

        # tune
        self.best_params = tune_models(
            self.best_model_name,
            self.X_train,
            self.y_train,
            n_trials=n_trial,
        )

        # build best model
        self.model = MODEL_REGISTRY[self.best_model_name](
            **self.best_params
        )

        # train best model
        self.model.fit(self.X_train, self.y_train)

        # evaluate
        y_pred = self.model.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, y_pred)
        self.mae = mean_absolute_error(self.y_test, y_pred)
        self.rmse = root_mean_squared_error(self.y_test, y_pred)


    def save(self, duration: str = "No data"):
        joblib.dump(self.model, settings.model_file)

        metadata = {
            "model_name": self.best_model_name,
            "params": self.best_params,
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "data_size": self.n_train + self.n_test,
            "created_at": datetime.now(ZoneInfo(settings.time_zone)).strftime("%d/%B/%Y %a %I:%M %p"),
            "duration": duration,
        }
        with open(settings.metadata_file, "w") as f:
            json.dump(metadata, f, indent=4, default=str)

    def run(self):
        start = datetime.now(ZoneInfo(settings.time_zone))
        self.train()
        duration = f"{(datetime.now(ZoneInfo(settings.time_zone)) - start).total_seconds():.2f} s"
        self.save(duration)
