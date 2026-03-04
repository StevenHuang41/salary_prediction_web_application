import json
import joblib
import pandas as pd
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
    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df if df is not None else pd.DataFrame()
        self.model = None
        self.best_model_name = None
        self.best_params = None
        self.mse = None
        self.mae = None
        self.rmse = None
        self.n_train = None
        self.n_test = None

    def load_data(self):
        if self.df.empty:
            self.df = pd.read_csv(settings.raw_data_file)

        self.df = clean_data(self.df, has_target_col=True)
        train_df, test_df = split_data(self.df)

        X_train = train_df.drop("salary", axis=1)
        y_train = train_df["salary"]
        self.n_train = len(X_train)

        X_test = test_df.drop("salary", axis=1)
        y_test = test_df["salary"]
        self.n_test = len(X_test)

        return X_train, y_train, X_test, y_test

    def train(self, n_trial: int = 10):
        X_train, y_train, X_test, y_test = self.load_data()

        # compare
        self.best_model_name, _ = compare_models(
            MODEL_REGISTRY,
            X_train,
            y_train,
        )

        # tune
        self.best_params = tune_models(
            self.best_model_name,
            X_train,
            y_train,
            n_trials=n_trial,
        )

        # build best model
        self.model = MODEL_REGISTRY[self.best_model_name](
            **self.best_params
        )

        # train best model
        self.model.fit(X_train, y_train)

        # evaluate
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.rmse = root_mean_squared_error(y_test, y_pred)


    def save(self):
        joblib.dump(self.model, settings.model_file)

        metadata = {
            "model_name": self.best_model_name,
            "params": self.best_params,
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "n_train": self.n_train,  # type: ignore
            "n_test": self.n_test,  # type: ignore
            "data_size": len(self.df)

        }
        with open(settings.metadata_file, "w") as f:
            json.dump(metadata, f, indent=4, default=str)

    def run(self):
        self.train()
        self.save()
