import optuna
from sklearn.model_selection import cross_val_score

from app.ml.models.linear import (
    build_linear,
    build_ridge,
    build_lasso,
    build_elasticNet,
)

def tune_linear_family(X, y, cv=3, n_trials=100):

    def objective(trial):
        model_name = trial.suggest_categorical(
            "model",
            [
                "linear",
                "ridge",
                "lasso",
                "elasticNet",
            ],
        )

        if model_name == "linear":
            model = build_linear()

        elif model_name == "ridge":
            params = {
                "alpha": trial.suggest_float(
                    "alpha", 1e-3, 1e+2, log=True
                ),
            }
            model = build_ridge(**params)

        elif model_name == "lasso":
            params = {
                "alpha": trial.suggest_float(
                    "alpha", 1e-3, 1e+2, log=True
                ),
                "selection": trial.suggest_categorical(
                    "selection", ['cyclic', 'random']
                ),
            }
            model = build_lasso(**params)

        elif model_name == "elasticNet":
            params = {
                "alpha": trial.suggest_float(
                    "alpha", 1e-3, 1e+2, log=True
                ),
                "l1_ratio": trial.suggest_float(
                    "l1_ratio", 0.01, 0.99
                ),
            }
            model = build_elasticNet(**params)

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

