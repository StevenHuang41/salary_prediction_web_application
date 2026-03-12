from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import pandas as pd
import numpy as np
import pytest
import warnings
from sklearn.exceptions import ConvergenceWarning

from app.ml.train.trainer import Trainer


def test_trainer_load(sample_df):
    trainer = Trainer()

    X_tra, y_tra, X_tes, y_tes = trainer.load_data(sample_df)

    assert not X_tra.empty
    assert "salary" not in X_tes.columns
    assert trainer.n_train > 0


@patch("app.ml.train.trainer.MODEL_REGISTRY")
@patch("app.ml.train.trainer.tune_models")
@patch("app.ml.train.trainer.compare_models")
def test_trainer_train(mock_compare, mock_tune, mock_registry, sample_df: pd.DataFrame):

    mock_compare.return_value = ("linear", {})
    mock_tune.return_value = {"param": True}

    mock_model_instance = MagicMock()
    mock_model_instance.predict.return_value = [100_000] * int(np.ceil(len(sample_df) * 0.2))

    mock_registry.__getitem__.return_value = MagicMock(return_value=mock_model_instance)

    trainer = Trainer()
    trainer.load_data(sample_df)
    trainer.train()

    assert trainer.best_model_name == "linear"
    assert trainer.model is mock_model_instance
    mock_model_instance.fit.asser_called_once()
    assert trainer.mse is not None


@patch("app.ml.train.trainer.joblib.dump")
def test_trainer_save(mock_dump, tmp_path):
    mock_model_path = tmp_path / "model.joblib"
    mock_metadata_path = tmp_path / "metadata.json"

    with patch("app.ml.train.trainer.settings", autospec=False) as mock_settings:
        mock_settings.model_file = mock_model_path
        mock_settings.metadata_file = mock_metadata_path
        mock_settings.time_zone = "Asia/Taipei"

        mock_dump

        trainer = Trainer()

        trainer.model = MagicMock()
        trainer.best_model_name = "nn"
        trainer.best_params = {"params": "good"}
        trainer.mse = 1000
        trainer.mae = 2000
        trainer.rmse = 3000
        trainer.n_train = 2
        trainer.n_test = 1

        trainer.save()

        mock_dump.assert_called_once_with(trainer.model, mock_model_path)
        assert mock_metadata_path.exists()

        with open(mock_metadata_path, "r") as f:
            meta = json.load(f)
            assert meta["model_name"] == "nn"
            assert meta["data_size"] == 3


@patch("app.ml.train.trainer.pd.read_csv")
def test_trainer_run_integration(mock_read, sample_df, tmp_path):
    mock_read.return_value = sample_df

    with (
        patch.object(Trainer, 'train') as mock_train,
        patch.object(Trainer, 'save') as mock_save,
        patch("app.ml.train.trainer.settings", autospec=False) as mock_settings
    ):
        mock_settings.raw_data_file = "fake.csv"
        mock_settings.time_zone = "Asia/Taipei"

        trainer = Trainer()
        trainer.run()

        mock_train.assert_called_once()
        mock_save.assert_called_once()
