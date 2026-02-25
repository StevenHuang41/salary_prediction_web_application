from .data_preprocessing import preprocess_data
from .data_cleansing import clean_directory

import os
import shutil
import pandas as pd
import numpy as np
import joblib
import json
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
)
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import keras
import keras_tuner as kt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Activation, LeakyReLU
from tensorflow.keras.activations import swish
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mse, mae
from tensorflow.keras.models import load_model
from tqdm import tqdm
from itertools import product

import warnings
warnings.filterwarnings(
    "ignore",
    message="Skipping variable loading for optimizer",
)
warnings.filterwarnings(
    "ignore",
    message="The objective has been evaluated at point",
)


SGD_search_space = [
    {
        'penalty': Categorical(['elasticnet']),
        'alpha': Real(1e-5, 1e+2, prior='log-uniform'),
        'l1_ratio': Real(0.01, 0.99),
        'learning_rate': Categorical(['adaptive', 'constant', 'invscaling']),
        'eta0': Real(1e-5, 1, prior='log-uniform'),
    },
    {
        'penalty': Categorical(['l1']),
        'alpha': Real(1e-5, 1e+2, prior='log-uniform'),
        'learning_rate': Categorical(['adaptive', 'constant', 'invscaling']),
        'eta0': Real(1e-5, 1, prior='log-uniform'),
    },
    {
        'penalty': Categorical(['l2']),
        'alpha': Real(1e-5, 1e+2, prior='log-uniform'),
        'learning_rate': Categorical(['adaptive', 'constant', 'invscaling']),
        'eta0': Real(1e-5, 1, prior='log-uniform'),
    },
]

model_configurations = [
    {
        'name': 'linear_model',
        'estimator': LinearRegression(),
        'search_space': {
            'fit_intercept': Categorical([True]),
        },
        'iter': 1,
    },
    {
        'name': 'ridge_model',
        'estimator': Ridge(),
        'search_space': {
            'alpha': Real(low=1e-3, high=1e+2, prior='log-uniform'),
        },
        'iter': 5,
    },
    {
        'name': 'lasso_model',
        'estimator': Lasso(max_iter=int(1e+6)),
        'search_space': {
            'alpha': Real(low=1e-3, high=1e+2, prior='log-uniform'),
            'selection': Categorical(['cyclic', 'random']),
        },
        'iter': 8,
    },
    {
        'name': 'elasticNet',
        'estimator': ElasticNet(max_iter=int(1e+6)),
        'search_space': {
            'alpha': Real(low=1e-3, high=1e+2, prior='log-uniform'),
            'l1_ratio': Real(low=0.01, high=0.99),
            'selection': Categorical(['cyclic', 'random']),
        },
        'iter': 8,
    },
    {
        'name': 'SGD_model',
        'estimator': SGDRegressor(max_iter=int(1e+6), early_stopping=True),
        'search_space': SGD_search_space,
        'iter': 8,
    },
    {
        'name': 'randomForestRegressor_model',
        'estimator': RandomForestRegressor(n_jobs=-1),
        'search_space': {
            'n_estimators': Integer(20, 80),
            'criterion': Categorical(['squared_error', 'absolute_error']),
            'max_depth': Integer(5, 15),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 10),
        },
        'iter': 8,
    },
    {
        'name': 'XGBRFRegressor_model',
        'estimator': xgb.XGBRFRegressor(),
        'search_space': {
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(10, 30),
            'subsample': Real(0.9, 1),
            'colsample_bytree': Real(0.9, 1),
            'reg_lambda': Real(0, 10),
            'reg_alpha': Real(0, 10),
            'gamma': Integer(0, 5),
        },
        'iter': 8,
    },
    {
        'name': 'LGBMRegressor_model',
        'estimator': lgb.LGBMRegressor(verbose=-1),
        'search_space': {
            'boosting_type': Categorical(['gbdt', 'dart', 'rf']),
            'num_leaves': Integer(10, 40),
            'max_depth': Integer(10, 30),
            'learning_rate': Real(1e-2, 1e+2),
            'n_estimators': Integer(50, 120),
            'subsample': Real(0.9, 1),
            'subsample_freq': Integer(0, 7),
            'colsample_bytree': Real(0.9, 1),
            'reg_alpha': Real(0, 10),
            'reg_lambda': Real(0, 10),
            'bagging_freq': Integer(1, 7),
            'bagging_fraction': Real(0.5, 0.99),
            'feature_fraction': Real(0.5, 0.99),
        },
        'iter': 8,
    },
    {
        'name': 'NN_model',
        'iter': 8,
    },
]

def NN_model_training(
    X_train,
    y_train,
    *,
    n_layers: int = 2,
    n_iter: int = 2
) -> tuple[Model, dict]:

    class BTuner(kt.BayesianOptimization):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters

            batch_size = hp.Choice('batch_size', [32, 64, 128])
            epochs = hp.Int('epochs', 10, 30, step=10)

            return super().run_trial(
                trial,
                *args,
                batch_size=batch_size,
                epochs=epochs,
                **kwargs,
            )

    def build_model(hp):
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))

        for i in range(1, n_layers + 1):
            hp_units = hp.Choice(f'unit_{i}', [32, 64, 125, 256])
            hp_actif = hp.Choice(f'acti_{i}',
                                 ['relu', 'tanh', 'leaky_relu', 'swish'])
            model.add(Dense(units=hp_units))
            if hp_actif == 'leaky_relu':
                model.add(LeakyReLU(
                    negative_slope=hp.Float(f'neg_slope_{i}', 0.1, 0.3)
                ))
            elif hp_actif == 'swish':
                model.add(Activation(swish))
            else :
                model.add(Activation(hp_actif))

        model.add(Dense(units=1,
                        activation=hp.Choice('acti_output',
                                             ['linear', 'relu'])))

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae'],
        )

        return model

    tuner = BTuner(
        hypermodel=build_model,
        objective=kt.Objective('val_loss', direction='min'),
        max_trials=n_iter,
        project_name='bayesianOptimization_NN_model',
        overwrite=True,
    )

    tuner.search(
        X_train, y_train,
        validation_split=0.2,
        shuffle=True,
        callbacks=[keras.callbacks.EarlyStopping(patience=5)],
        verbose=0,
    )

    ## get best model from tuner
    NN_model = tuner.get_best_models(num_models=1)[0]

    ## get best parameters from tuner
    NN_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0].values

    return (NN_model, NN_hyperparams)

class NumpyEncoder(json.JSONEncoder):
    def default(self, object):
        if isinstance(object, np.integer):
            return int(object)
        elif isinstance(object, np.floating):
            return float(object)
        elif isinstance(object, np.ndarray):
            return object.tolist()
        elif isinstance(object, np.bool_):
            return bool(object)

        return super().default(object)

def model_select(X_train, y_train, X_test, y_test, store_file) -> str:
    """
    receive: splited data
    store the best model and its data into best_performance dir

    create best_performance dir for storing model and its data.
    create {model_name}.json file, storing best models data
    """

    models_storage = {}

    X_train_, X_test_ = preprocess_data(
        X_train, y_train, X_test, use_polynomial=False
    )
    X_train_poly, X_test_poly = preprocess_data(
        X_train, y_train, X_test, use_polynomial=True
    )
    for use_poly, config in tqdm(
        product([False, True], model_configurations),
        total=2 * len(model_configurations),
        ncols=100,
        desc='Model Training'
    ):
        train_X, test_X = (X_train_poly, X_test_poly) \
                            if use_poly \
                            else (X_train_, X_test_)

        tqdm.write(f"Running {config['name']}"
                   f"{' with polynominal features' if use_poly else ''}")

        # sklearn model
        if 'NN' not in config['name']:
            model = BayesSearchCV(
                estimator=config['estimator'],
                search_spaces=config['search_space'],
                n_iter=config['iter'],
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=132,
                verbose=0,
            )
            model.fit(train_X, y_train)
            y_pred = model.predict(test_X)
            
            model_name = f"{config['name']}{'_poly' if use_poly else ''}"
            params = dict(getattr(model, "best_params_", {}))

        # neural model
        else : 
            model, NN_params = NN_model_training(
                train_X, y_train, n_iter=config['iter'],
            )
            y_pred = model.predict(test_X, verbose=-1)

            model_name = f"NN_model{'_poly' if use_poly else ''}"
            params = dict(NN_params)

        models_storage[model_name] = {
            'model': model,
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'params': params,
        }

    ## get the model with the minimum mae|mse
    best_model_name = min(models_storage.keys(),
                          key=lambda x: models_storage[x]['mae'])

    # if best_performance dir is not empty, clear it
    best_performance_dir = os.path.join(os.getcwd(), f'{store_file}')
    try :
        if os.listdir(best_performance_dir) != []:
            # shutil.rmtree(store_file)
            clean_directory(store_file)
    except :
        pass 

    # create best_performance dir
    os.makedirs(name=f'{store_file}', mode=0o755, exist_ok=True)

    ## store best model into best_performance dir
    if 'NN' in best_model_name:
        models_storage[best_model_name]['model'] \
            .save(f'{best_performance_dir}/{best_model_name}.keras')
    else :
        joblib.dump(models_storage[best_model_name]['model'],
                    f'{best_performance_dir}/{best_model_name}.joblib')

    ## store best model params into best_performance dir
    best_model = models_storage[best_model_name].pop('model', None)
    with open(f'{best_performance_dir}/{best_model_name}.json', 'w') as f:
        json.dump(models_storage[best_model_name], f, indent=4)

    return best_model_name, best_model


if __name__ == "__main__":
    
    from data_cleansing import cleaning_data
    from data_spliting import spliting_data

    ## load csv 
    p_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    FILE_PATH = os.path.join(p_dir, "database/Salary_Data.csv")
    df = pd.read_csv(FILE_PATH, delimiter=',')
    df = cleaning_data(df, has_target_columns=True)

    X_train, X_test, y_train, y_test = spliting_data(df)

    # test 
    store_file_name = "best"

    model_name, model = \
        model_select(X_train, y_train, X_test, y_test, store_file_name)
    print(model_name, model)

    shutil.rmtree(store_file_name)