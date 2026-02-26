import optuna
from sklearn.model_selection import cross_val_score

from app.ml.models.tree import (
    build_HGBR,
    build_xgb,
    build_xgbrf,
)


def tune_tree_family(X, y, cv=3, n_trials=100):

    def objective(trial):
        model_name = trial.suggest_categorical(
            "model",
            ["HGBR", "xgb", "xgbrf"],
        )

        if model_name == "HGBR":
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_iter": trial.suggest_int(
                    "max_iter", 100, 500
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 3, 10
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 5, 40
                ),
                "l2_regularization": trial.suggest_float(
                    "l2_regularization", 0.0, 5.0
                ),
            }
            model = build_HGBR(**params)

        elif model_name == "xgb":
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 3, 12
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 500
                ),
                "subsample": trial.suggest_float(
                    "subsample", 0.6, 1.0
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 0.0, 5.0
                ),
            }
            model = build_xgb(**params)

        elif model_name == "xgbrf":
            params = {
                "max_depth": trial.suggest_int(
                    "max_depth", 3, 12
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 500
                ),
                "subsample": trial.suggest_float(
                    "subsample", 0.6, 1.0
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 0.0, 5.0
                ),
            }
            model = build_xgbrf(**params)

        scores = -cross_val_score(
            model,
            X=X, y=y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        ).mean()
        return scores

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_model = best_params.pop("model")

    return best_model, best_params
