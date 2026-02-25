from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

from app.ml.preprocess.nn import build as build_nn_pre

def build_MLP(
    hidden_layer_sizes: tuple[int, ...] = (100,),
    max_iter: int = 1000,
):
    return Pipeline([
        ('preprocess', build_nn_pre()),
        ('model', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate="adaptive",
            max_iter=max_iter,
        )),
    ])
