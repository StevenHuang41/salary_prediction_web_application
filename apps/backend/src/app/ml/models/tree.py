from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.pipeline import Pipeline

from app.ml.preprocess.tree import build as build_tree_pre

def build_HGBR(**kws):
    return Pipeline([
        ('preprocess', build_tree_pre()),
        ('model', HistGradientBoostingRegressor(**kws)),
    ])

def build_xgb(**kws):
    return Pipeline([
        ('preprocess', build_tree_pre()),
        ('model', XGBRegressor(**kws)),
    ])

def build_xgbrf(**kws):
    return Pipeline([
        ('preprocess', build_tree_pre()),
        ('model', XGBRFRegressor(**kws)),
    ])

