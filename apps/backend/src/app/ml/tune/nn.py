import optuna
from sklearn.model_selection import cross_val_score

from app.ml.models.nn import build_MLP


def tune_nn_family(X, y, cv=3, n_trials=100):

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(
                trial.suggest_int(f"n_node_{i}", 8, 128)
            )

        params = {
            "hidden_layer_sizes": tuple(hidden_layers),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["adaptive", "constant"]
            ),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 0.1, log=True
            ),
            "max_iter": trial.suggest_int(
                "max_iter", 800, 3000
            ),
            "alpha": trial.suggest_float(
                "alpha", 1e-6, 0.1, log=True
            ),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "tanh"]
            ),
            "batch_size": trial.suggest_int(
                "batch_size", 32, 128
            ),
            "early_stopping": True,
        }

        model = build_MLP(**params)

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

    return "nn", best_params

