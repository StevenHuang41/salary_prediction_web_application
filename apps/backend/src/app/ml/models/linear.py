from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.pipeline import Pipeline

from app.ml.preprocess.linear import build as build_linear_pre

def build_linear():
    return Pipeline([
        ('preprocess', build_linear_pre()),
        ('model', LinearRegression()),
    ])

def build_ridge():
    return Pipeline([
        ('preprocess', build_linear_pre()),
        ('model', Ridge()),
    ])

def build_lasso():
    return Pipeline([
        ('preprocess', build_linear_pre()),
        ('model', Lasso()),
    ])

def build_elasticNet():
    return Pipeline([
        ('preprocess', build_linear_pre()),
        ('model', ElasticNet()),
    ])

