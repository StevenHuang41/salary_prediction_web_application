from pathlib import Path
import json
import pandas as pd
import pytest
import warnings
from sklearn.exceptions import ConvergenceWarning

from app.ml.train.trainer import Trainer


@pytest.fixture
def sample_X():
    df = pd.DataFrame({
        "age": [25, 32, 24, 31, 29, 33]*2,
        "gender": ["female", "male", "female", "male", "female", "male"]*2,
        "education_level": ["high_school", "phd", "master", "bachelor", "master", "phd"]*2,
        "job_title": [
            "ai engineer",
            "data scientist",
            "ai engineer",
            "senior data scientist",
            "senior ai engineer",
            "senior data engineer"
        ]*2,
        "years_of_experience": [3, 5, 2, 3, 7, 5]*2,
        "salary": [120000, 210000, 110000, 100000, 220000, 150000]*2,
    })
    return df


def test_trainer_run(monkeypatch, sample_X, tmp_path):
    # mock compare
    monkeypatch.setattr(
        "app.ml.train.trainer.compare_models",
        lambda models, X, y: ("linear", 100)
    )

    # mock tune
    monkeypatch.setattr(
        "app.ml.train.trainer.tune_models",
        lambda model_name, X, y, n_trials: {}
    )

    # mock model
    class DummyModel:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return [0] * len(X)

    monkeypatch.setattr(
        "app.ml.train.trainer.MODEL_REGISTRY",
        {"linear": lambda **best_params: DummyModel()}
    )

    monkeypatch.setattr(
        "app.core.config.settings.RAW_DATA_FILE",
        "path"
    )

    monkeypatch.setattr(
        "app.ml.train.trainer.pd.read_csv",
        lambda path: sample_X
    )

    monkeypatch.setattr(
        "app.ml.train.trainer.settings.MODEL_FILE",
        tmp_path / "model.pkl"
    )

    monkeypatch.setattr(
        "app.ml.train.trainer.settings.METADATA_FILE",
        tmp_path / "metadata.json"
    )

    monkeypatch.setattr(
        "app.ml.train.trainer.joblib.dump",
        lambda model, path: Path(tmp_path / "model.pkl").touch()
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        trainer = Trainer()
        trainer.run()

    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "metadata.json").exists()

    with open(Path(tmp_path / "metadata.json"), "r") as f:
        metadata = json.load(f)

    assert "model_name" in metadata
    assert "params" in metadata
    assert "mse" in metadata
    assert "mae" in metadata
    assert "rmse" in metadata
    assert "n_train" in metadata
    assert "n_test" in metadata
