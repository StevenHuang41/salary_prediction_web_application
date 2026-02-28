from sklearn.model_selection import cross_val_score


def compare_models(models: dict, X, y, cv: int = 3):
    results: dict[str, float] = {}

    for model_name, build_model in models.items():
        model = build_model()

        score = -cross_val_score(
            model,
            X, y,
            cv=cv,
            scoring="neg_mean_squared_error",
        ).mean()

        results[model_name] = score

        best_model = min(results, key=lambda k: results[k])

    return best_model, results[best_model]
