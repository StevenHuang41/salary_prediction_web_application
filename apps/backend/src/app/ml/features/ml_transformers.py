from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

class JobSeniorityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.Series | pd.DataFrame | np.ndarray):
        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                col = pd.Series(X).astype(str)
            else :
                col = pd.Series(X[:, 0]).astype(str)
        elif isinstance(X, pd.Series):
            col = X.astype(str)
        else :
            col = X.iloc[:, 0].astype(str)

        out = np.select(
            [
                col.str.contains(r'.*junior|assist|asso.*', regex=True),
                col.str.contains(r'.*senior.*', regex=True),
                col.str.contains(r'.*director|vp.*', regex=True),
                col.str.contains(r'.*ceo|cfo|chief.*', regex=True),
            ],
            [
                "junior",
                "senior",
                "director_vp",
                "c_level",
            ],
            default='mid',
        )
        return out.reshape(-1, 1)


    def get_feature_names_out(self, input_features=None):
        return np.array(["seniority"])


class JobGroupTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                col = pd.Series(X).astype(str)
            else :
                col = pd.Series(X[:, 0]).astype(str)
        elif isinstance(X, pd.Series):
            col = X.astype(str)
        else :
            col = X.iloc[:, 0].astype(str)

        out = np.select(
            [
                col.str.contains(r'project manager'),
                col.str.contains(r'data'),
                col.str.contains(r'software|engineer|developer'),
                col.str.contains(r'design|content|creative'),

                col.str.contains(r'market'),
                col.str.contains(r'product|quali|supply'),
                col.str.contains(r'financ'),
                col.str.contains(r'hr|human|recru|training'),

                col.str.contains(r'sales'),
                col.str.contains(r'operation'),
                col.str.contains(r'research'),
                col.str.contains(r'busin'),

                col.str.contains(r'custom|driver'),
                col.str.contains(r'account'),
                col.str.contains(r'social media|public re'),
                col.str.contains(r'recep|admin|desk|event|office|coordi|consulta|writer'),

                col.str.contains(r'it\b|tech'),
                col.str.contains(r'scienti'),
            ],
            [
                'project_manager', 'data', 'software_engineer_developer', 'design_creative',
                'marketing', 'product_quality_supply', 'finance', 'hr',
                'sales', 'operation', 'research', 'business',
                'custome_labor', 'accounting', 'social_media_pr', 'office',
                'it', 'science',
            ],
            default='other',
        )

        return out.reshape(-1, 1)


    def get_feature_names_out(self, input_features=None):
        return np.array(["group"])


class TextExistTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text: str, *, case: bool = False):
        self.case = case
        self.text = text

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        col = np.asarray(X).flatten().astype(str)

        col = col if self.case else np.char.lower(col)
        search_text = self.text if self.case else self.text.lower()

        out = (np.char.find(col, search_text) != -1).astype(int)

        return out.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"is_{self.text}"])


class MathTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        method: Literal['log', 'sqrt', '1/x', 'square', 'cube', 'exp'],
        suffix=None
    ):
        self.method = method
        self.suffix = suffix
        self.origin_colname = None
        self.out_suffix = None

    def fit(self, X, y=None):
        valid_methods = {
            'log': 'log',
            'sqrt': 'square_root',
            '1/x': 'reciprocal',
            'square': 'square',
            'cube': 'cube',
            'exp': 'exp',
        }

        if self.method not in valid_methods:
            raise ValueError(f"Unknown method: {self.method}")

        # set suffix
        self.out_suffix = self.suffix or valid_methods[self.method]

        # get column name
        if isinstance(X, pd.DataFrame):
            self.origin_colname = X.columns[0]
        elif isinstance(X, pd.Series):
            self.origin_colname = X.name or "mathT"
        else :
            self.origin_colname = "mathT"

        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        if self.method == 'log':
            X = np.where(X <= 0, 0, np.log1p(X))
        elif self.method == 'sqrt':
            X = np.where(X < 0, 0, np.sqrt(X))
        elif self.method == '1/x':
            X = np.where(X == 0, 0, np.reciprocal(X))
        elif self.method == 'square':
            X = np.square(X)
        elif self.method == 'cube':
            X = np.power(X, 3)
        elif self.method == 'exp':
            X = np.exp(X)

        return X.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.origin_colname}_{self.out_suffix}"])


class MultiLabelWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, sep=" "):
        self.sep = sep
        self.vectorizer = MultiLabelBinarizer()

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                col = pd.Series(X).astype(str)
            else :
                col = pd.Series(X[:, 0]).astype(str)
        elif isinstance(X, pd.Series):
            col = X.astype(str)
        else :
            col = X.iloc[:, 0].astype(str)

        col = col.str.split(self.sep)
        self.vectorizer.fit(col)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                col = pd.Series(X).astype(str)
            else :
                col = pd.Series(X[:, 0]).astype(str)
        elif isinstance(X, pd.Series):
            col = X.astype(str)
        else :
            col = X.iloc[:, 0].astype(str)

        col = col.str.split(self.sep)
        return self.vectorizer.transform(col)

    def get_feature_names_out(self, input_features=None):
        return self.vectorizer.classes_

