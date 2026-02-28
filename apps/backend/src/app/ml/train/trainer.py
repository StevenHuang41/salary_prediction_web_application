import joblib
import pandas as pd
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
)

from app.core.config import settings
from app.ml.data import clean_data, split_data
from app.ml.models import MODEL_REGISTRY
from app.ml.train import compare_models
from app.ml.tune import tune_models


class Trainer:
    def __init__(self, df=None):
        self.df = df
        self.model = None
        self.best_model_name = None
        self.best_params = None
        self.mse = None
        self.rmse = None

    def load_data(self):
        if self.df is None:
            data_path = settings.RAW_DATA_FILE
            df = pd.read_csv(data_path)

        df = clean_data(df, has_target_col=True)
        train_df, test_df = split_data(df)

        X_train = train_df.drop("salary", axis=1)
        y_train = train_df["salary"]

        X_test = test_df.drop("salary", axis=1)
        y_test = test_df["salary"]

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
        self.rmse = root_mean_squared_error(y_test, y_pred)

        return self.mse, self.rmse

    def save(self):
        joblib.dump(self.model, settings.MODEL_FILE)

    def run(self):
        mse, rmse = self.train()
        self.save()

        return {
            "best_model": self.best_model_name,
            "best_params": self.best_params,
            "mse": mse,
            "rmse": rmse,
        }
