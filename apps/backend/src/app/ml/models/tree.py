from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.pipeline import Pipeline

from app.ml.preprocess.tree import build as build_tree_pre

def build_HGBR():
    return Pipeline([
        ('preprocess', build_tree_pre()),
        ('model', HistGradientBoostingRegressor()),
    ])

def build_xgb():
    return Pipeline([
        ('preprocess', build_tree_pre()),
        ('model', XGBRegressor()),
    ])

def build_xgbrf():
    return Pipeline([
        ('preprocess', build_tree_pre()),
        ('model', XGBRFRegressor()),
    ])

