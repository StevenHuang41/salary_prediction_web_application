from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.pipeline import Pipeline

from app.ml.preprocess.linear import build as build_linear_pre


def build_linear(**kws):
    return Pipeline([
        ('preprocess', build_linear_pre()),
        ('model', LinearRegression(**kws)),
    ])

def build_ridge(**kws):
    return Pipeline([
        ('preprocess', build_linear_pre()),
        ('model', Ridge(**kws)),
    ])

def build_lasso(**kws):
    return Pipeline([
        ('preprocess', build_linear_pre()),
        ('model', Lasso(**kws)),
    ])

def build_elasticNet(**kws):
    return Pipeline([
        ('preprocess', build_linear_pre()),
        ('model', ElasticNet(**kws)),
    ])

