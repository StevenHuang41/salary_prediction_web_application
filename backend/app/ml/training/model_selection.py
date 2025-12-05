# import os
# import shutil
# import pandas as pd
# import numpy as np
# import joblib
# import json
# from skopt import BayesSearchCV
# from skopt.space import Real, Integer, Categorical
# import keras
# import keras_tuner as kt
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Input, Activation, LeakyReLU
# from tensorflow.keras.activations import swish
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import mse, mae
# from tensorflow.keras.models import load_model
# from tqdm import tqdm
# from itertools import product

# import warnings
# warnings.filterwarnings(
#     "ignore",
#     message="Skipping variable loading for optimizer",
# )
# warnings.filterwarnings(
#     "ignore",
#     message="The objective has been evaluated at point",
# )
import optuna
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from xgboost import XGBRegressor, XGBRFRegressor
from lightgbm import LGBMRegressor
import keras
from keras import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping





def model_selection(X_train, y_train):

    cv = KFold(n_splits=3, shuffle=True, random_state=48)

    def eval_sklearn_model(model, X, y):
        score = cross_val_score(
            model,
            X, y,
            cv=cv,
            scoring="neg_mean_squared_error",
            # n_jobs=-1,
        ).mean()
        
        return -score

    early_stpping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    def build_nn_model(trial, n_features):
        model = Sequential([Input(shape=(n_features,))])
        lr= trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)

        n_layers = trial.suggest_int("n_layers", 1, 3)
        for i in range(n_layers):
            nodes = trial.suggest_int(f"n{i + 1}", 16, 64)
            acti = trial.suggest_categorical(f"a{i + 1}", ['relu', 'tanh', 'sigmoid'])
            model.add(Dense(nodes, activation=acti))
            
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='mse',
            metrics=['mae'],
        )

        return model

    model_list = ['linear', 'ridge', 'lasso', 'elastic', 'svr', 'knn',
                  'decisionT', 'randomF', 'xgrf', 'gb', 'xggb', 'lgb', 'nn']

    def objective(trial):
        model_name = trial.suggest_categorical('model', model_list)
        
        if model_name == 'nn':
            fold_losses = []
            
            for tra_idx, val_idx in cv.split(X_train):
                X_tra = X_train[tra_idx]
                y_tra = y_train[tra_idx]
                X_val = X_train[val_idx]
                y_val = y_train[val_idx]
                
                model = build_nn_model(trial, X_train.shape[1])
                model.fit(
                    X_tra, y_tra,
                    validation_data=(X_val, y_val),
                    epochs=trial.suggest_int("epochs", 30, 100),
                    batch_size=trial.suggest_int("batch_size", 32, 256),
                    callbacks=[early_stpping],
                    verbose=0,
                )
                y_pred = model.predict(X_val, verbose=0)
                fold_losses.append(mean_squared_error(y_val, y_pred))
            
            return np.mean(fold_losses)

        elif model_name == 'linear':
            params = {}
            model = LinearRegression(**params)
        elif model_name == 'ridge':
            params = {'alpha': trial.suggest_float('alpha', 0.1, 100, log=True)}
            model = Ridge(**params)
        elif model_name == 'lasso':
            params = {'alpha': trial.suggest_float('alpha', 1e-3, 1, log=True)}
            model = Lasso(**params)
        elif model_name == 'elastic':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-4, 1),
                'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
            }
            model = ElasticNet(**params)
        elif model_name == 'svr':
            params = {
                'degree': trial.suggest_int('degree', 1, 5),
                'C': trial.suggest_float('C', 0.1, 1000, log=True),
                'epsilon': trial.suggest_float('epsilon', 1e-3, 0.1),
            }
            model = SVR(**params)
        elif model_name == 'knn':
            params = {'n_neighbors': trial.suggest_int('n_neighbors', 3, 50)}
            model = KNeighborsRegressor(**params)
        elif model_name == 'decisionT':
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            model = DecisionTreeRegressor(**params)
        elif model_name == 'randomF':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            model = RandomForestRegressor(**params)
        elif model_name == 'xgrf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'subsample': trial.suggest_float('subsample', 0.1, 1),
            }
            model = XGBRFRegressor(**params)
        elif model_name == 'gb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 1e-2, 0.3, log=True), 
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            model = GradientBoostingRegressor(**params)
        elif model_name == 'xggb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 0.1, log=True), 
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True), 
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True), 

            }
            model = XGBRegressor(**params)
        elif model_name == 'lgb':
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 10, 40),
                'max_depth': trial.suggest_categorical('max_depth', [-1, 3, 6, 12]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 120),
                'subsample': trial.suggest_float('subsample', 0.9, 1)
            }
            model = LGBMRegressor(**params)

        return eval_sklearn_model(model, X_train, y_train)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    
    return study.best_params
####################################################################################################

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, object):
#         if isinstance(object, np.integer):
#             return int(object)
#         elif isinstance(object, np.floating):
#             return float(object)
#         elif isinstance(object, np.ndarray):
#             return object.tolist()
#         elif isinstance(object, np.bool_):
#             return bool(object)

#         return super().default(object)
