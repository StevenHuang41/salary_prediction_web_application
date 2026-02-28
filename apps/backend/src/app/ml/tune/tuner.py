import optuna
from sklearn.model_selection import cross_val_score

from app.ml.models import MODEL_REGISTRY
from app.ml.tune.search_spaces import SEARCH_SPACE_REGISTRY


def tune_models(
    model_name,
    X, y,
    cv=3,
    n_trials=100
):
    def objective(trial):
        params = SEARCH_SPACE_REGISTRY[model_name](trial)
        model = MODEL_REGISTRY[model_name](**params)

        scores = -cross_val_score(
            model,
            X=X, y=y,
            cv=cv,
            scoring="neg_mean_squared_error",
        ).mean()
        return scores

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()

    return best_params

