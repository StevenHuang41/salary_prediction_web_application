from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

from app.ml.preprocess.nn import build as build_nn_pre

def build_MLP(**kws):
    default_params = {
        "hidden_layer_sizes": (100,),
        "learning_rate": "adaptive",
        "max_iter": 1000,
    }
    default_params.update(kws)

    return Pipeline([
        ('preprocess', build_nn_pre()),
        ('model', MLPRegressor(**default_params)),
    ])
