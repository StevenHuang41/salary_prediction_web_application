import optuna


def space_linear(trial):
    return {}

def space_ridge(trial):
    return {
        "alpha": trial.suggest_float(
            "alpha", 1e-3, 1e+2, log=True
        ),
    }

def space_lasso(trial):
    return {
        "alpha": trial.suggest_float(
            "alpha", 1e-3, 1e+2, log=True
        ),
        "selection": trial.suggest_categorical(
            "selection", ['cyclic', 'random']
        ),
    }

def space_elasticNet(trial):
    return {
        "alpha": trial.suggest_float(
            "alpha", 1e-3, 1e+2, log=True
        ),
        "l1_ratio": trial.suggest_float(
            "l1_ratio", 0.01, 0.99
        ),
    }


def space_HGBR(trial):
    return {
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

def space_xgb(trial):
    return {
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

def space_xgbrf(trial):
    return {
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


def space_nn(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)

    hidden_layers = []
    for i in range(n_layers):
        hidden_layers.append(
            trial.suggest_int(f"n_node_{i}", 8, 128)
        )

    return {
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


SEARCH_SPACE_REGISTRY = {
    "linear": space_linear,
    "ridge": space_ridge,
    "lasso": space_lasso,
    "elasticNet": space_elasticNet,
    "xgbrf": space_xgbrf,
    "xgb": space_xgb,
    "HGBR": space_HGBR,
    "nn": space_nn,
}
