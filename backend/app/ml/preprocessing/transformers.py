from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

class TfidfWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=64):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
    
    def _ensure_series(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].astype(str)

        elif isinstance(X, pd.Series):
            return X.astype(str)

        elif isinstance(X, str):
            return pd.Series([X])

    def fit(self, X, y=None):
        X_series = self._ensure_series(X)
        self.vectorizer.fit(X_series)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X_series = self._ensure_series(X)
        return self.vectorizer.transform(X_series).toarray()
    
    def get_feature_names_out(self, input_features=None):
        return self.vectorizer.get_feature_names_out()

class MathTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method=None, suffix=None):
        self.method = method
        self.suffix = suffix
    
    def fit(self, X: pd.DataFrame, y=None):
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

        self.suffix_ = self.suffix if self.suffix is not None else valid_methods[self.method]
        self.origin_colname = X.columns[0]
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=float)
        else :
            X = np.asanyarray(X, dtype=float)
            
        if self.method == 'log':
            X = np.where(X <= 0, 0, np.log1p(X))
            suffix_ = 'log'
        elif self.method == 'sqrt':
            X = np.where(X < 0, 0, np.sqrt(X))
            suffix_ = 'sqrt'
        elif self.method == '1/x':
            X = np.where(X == 0, 0, np.reciprocal(X))
            suffix_ = 'reciprocal'
        elif self.method == 'square':
            X = np.square(X)
            suffix_ = 'square'
        elif self.method == 'cube':
            X = np.power(X, 3)
            suffix_ = 'cube'
        elif self.method == 'exp':
            X = np.exp(X)
            suffix_ = 'exp'
        else :
            raise ValueError(f"Unknown method: {self.method}")
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.origin_colname}_{self.suffix_}"])
    
class TextEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, word, case=False, regex=False):
        self.word = word
        self.case = case
        self.regex = regex
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].astype(str)
        else :
            X = pd.Series(X[:, 0]).astype(str)
        
        new_col = (
            X
            .str.contains(self.word,
                          case=self.case,
                          regex=self.regex,
                          na=False)
            .astype(int)
        )
        
        return new_col.to_numpy().reshape(-1, 1)
    
    def get_feature_names_out(self, input_features=None):
        return np.array([f"is_{self.word}"])

class FeatureCreation(BaseEstimator, TransformerMixin):
    def __init__(self, method=None, case=False, regex=True):
        self.method = method
        self.case = case
        self.regex = regex
        
    def fit(self, X: pd.DataFrame, y=None):
        if self.method in ['seniority', 'group']:
            self.feature_name = f"job_{self.method}"
        else :
            name = str(self.method).replace(' ', "_")
            self.feature_name = f'is_{name}'
            
        self.input_column_names_ = X.columns
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        col = X.loc[:, 'job_title'].astype(str)

        if self.method == 'seniority':
            X_out = np.select(
                [
                    col.str.contains(r'vp|vice president|ceo|chief|principal', regex=True),
                    col.str.contains(r'junior|associate|entry|assistant', regex=True),
                    col.str.contains(r'senior', regex=True),
                    col.str.contains(r'director', regex=True),
                ],
                ['vp_clevel_principal', 'junior', 'senior', 'director'],
                default='mid',
            )
            
        elif self.method == 'group':
            X_out = np.select(
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
                default='other'
            )
        
        else :
            X_out = col.str.contains(rf'{self.method}',
                                     case=self.case,
                                     regex=self.regex).astype(int)
            
        X.loc[:, self.feature_name] = X_out
        return X
            
    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.input_column_names_) + [self.feature_name])