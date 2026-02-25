from sklearn.model_selection import cross_val_score, cross_validate


def compare_models(models: dict, X, y, cv: int = 3):
# cleaned, split data (X and y)
    results: dict[str, float] = {}

    for model_name, build_model in models.items():
        model = build_model()

        rmse = -cross_val_score(
            model,
            X, y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        ).mean()

        results[model_name] = rmse

    return max(results, key=lambda k: results[k])



