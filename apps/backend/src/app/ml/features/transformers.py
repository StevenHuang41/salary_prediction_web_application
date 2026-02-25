from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

class JobSeniorityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray | pd.Series | pd.DataFrame | list):
        if isinstance(X, pd.Series):
            col = X.astype(str)
        elif isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0].astype(str)
        elif isinstance(X, np.ndarray):
            col = pd.Series(X.ravel()).astype(str)
        else :
            col = pd.Series(X).astype(str)

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
        return np.array(["job_seniority"])


class JobGroupTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray | pd.Series | pd.DataFrame | list):
        if isinstance(X, pd.Series):
            col = X.astype(str)
        elif isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0].astype(str)
        elif isinstance(X, np.ndarray):
            col = pd.Series(X.ravel()).astype(str)
        else :
            col = pd.Series(X).astype(str)

        out = np.select(
            [
                col.str.contains(r'software'),
                col.str.contains(r'back'),
                col.str.contains(r'front'),
                col.str.contains(r'full stack'),
                col.str.contains(r'web'),

                col.str.contains(r'data'),
                col.str.contains(r'project'),
                col.str.contains(r'marketing'),
                col.str.contains(r'product'),
                col.str.contains(r'finan|accou'),

                col.str.contains(r'hr|human|recrui'),
                col.str.contains(r'sale'),
                col.str.contains(r'operation'),
                col.str.contains(r'research'),
                col.str.contains(r'graphic'),

                col.str.contains(r'busin'),
                col.str.contains(r'media|adverti'),
                col.str.contains(r'custom|reception'),
                col.str.contains(r'admin|offic'),

                col.str.contains(r'ux\b'),
                col.str.contains(r'engin'),
                col.str.contains(r'it\b|tech'),
                col.str.contains(r'scien'),
                col.str.contains(r'develo'),

                col.str.contains(r'content|creat|director'),
                col.str.contains(r'manage|ceo'),
                col.str.contains(r'driver|supply'),
            ],
            [
                "software", "backend", "frontend", "full_stack", "web",
                "data", "project", "marketing", "product", "finance",
                "hr", "sales", "operation", "research", "graphic",
                "business", "media", "customer_service", "admin",
                "ux", "engineering", "it", "science", "developer",
                "creative", "manage", "labor"
            ],
            default='other',
        )

        return out.reshape(-1, 1)


    def get_feature_names_out(self, input_features=None):
        return np.array(["job_group"])


class JobRoleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray | pd.Series | pd.DataFrame | list):
        if isinstance(X, pd.Series):
            col = X.astype(str)
        elif isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0].astype(str)
        elif isinstance(X, np.ndarray):
            col = pd.Series(X.ravel()).astype(str)
        else :
            col = pd.Series(X).astype(str)

        out = np.select(
            [
                col.str.contains(r'devel.*', regex=True),
                col.str.contains(r'engine.*', regex=True),
                col.str.contains(r'analy.*', regex=True),
                col.str.contains(r'manag.*', regex=True),

                col.str.contains(r'direc.*', regex=True),
                col.str.contains(r'asso.*', regex=True),
                col.str.contains(r'coordina.*', regex=True),
                col.str.contains(r'scient.*', regex=True),

                col.str.contains(r'\bvp', regex=True),
                col.str.contains(r'represen', regex=True),
                col.str.contains(r'support', regex=True),
                col.str.contains(r'executive', regex=True),

                col.str.contains(r'designer', regex=True),
                col.str.contains(r'ceo', regex=True),
                col.str.contains(r'accou', regex=True),
                col.str.contains(r'recep', regex=True),

                col.str.contains(r'specialist', regex=True),
                col.str.contains(r'recru', regex=True),
                col.str.contains(r'general', regex=True),

                col.str.contains(r'resear', regex=True),
                col.str.contains(r'advi', regex=True),
                col.str.contains(r'clerk', regex=True),
                col.str.contains(r'offi', regex=True),

                col.str.contains(r'assi', regex=True),
                col.str.contains(r'consul', regex=True),
                col.str.contains(r'writer', regex=True),

                col.str.contains(r'archi', regex=True),
                col.str.contains(r'producer', regex=True),
                col.str.contains(r'driver', regex=True),
            ],
            [
                "developer", "engineer", "analyst", "manager",
                "director", "associate", "coordinator", "scientist",
                "vp", "representative", "support", "execu",
                "designer", "ceo", "accountant", "reception",
                "specialist", "recruiter", "generalist",
                "researcher", "advisor", "clerk", "officer",
                "assistant", "consultant", "writer",
                "architect", "producer", "driver"

            ],
            default='unknown',
        )
        return out.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["job_role"])


class TextExistTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text: str, *, case: bool = False):
        self.case = case
        self.text = text

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray | pd.Series | pd.DataFrame | list):
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

    def transform(self, X: pd.DataFrame | pd.Series | np.ndarray | list):
        if isinstance(X, pd.Series):
            col = np.asarray(X, dtype=float)
        elif isinstance(X, pd.DataFrame):
            col = np.asarray(X.iloc[:, 0], dtype=float)
        elif isinstance(X, list):
            col = np.asarray(X, dtype=float)
        else :
            col = X.ravel().astype('float')

        if self.method == 'log':
            out = np.where(col <= 0, 0, np.log1p(col))
        elif self.method == 'sqrt':
            out = np.where(col < 0, 0, np.sqrt(col))
        elif self.method == '1/x':
            out = np.where(col == 0, 0, np.reciprocal(col))
        elif self.method == 'square':
            out = np.square(col)
        elif self.method == 'cube':
            out = np.power(col, 3)
        elif self.method == 'exp':
            out = np.exp(col)

        return out.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.origin_colname}_{self.out_suffix}"])


class MultiLabelWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, sep=" "):
        self.sep = sep
        self.vectorizer = MultiLabelBinarizer()

    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            col = X.astype(str)
        elif isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0].astype(str)
        elif isinstance(X, np.ndarray):
            col = pd.Series(X.ravel()).astype(str)
        else :
            col = pd.Series(X).astype(str)

        col = col.str.split(self.sep)
        self.vectorizer.fit(col)
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray | pd.Series | pd.DataFrame | list):
        if isinstance(X, pd.Series):
            col = X.astype(str)
        elif isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0].astype(str)
        elif isinstance(X, np.ndarray):
            col = pd.Series(X.ravel()).astype(str)
        else :
            col = pd.Series(X).astype(str)

        col = col.str.split(self.sep)
        return self.vectorizer.transform(col)

    def get_feature_names_out(self, input_features=None):
        return self.vectorizer.classes_

