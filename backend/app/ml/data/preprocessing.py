import pandas as pd

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import TargetEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config

class TfidfWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=64):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
    
    def _ensure_series(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].astype(str)
        elif isinstance(X, pd.Series):
            return X.astype(str)
        elif isinstance(X, (list, tuple)):
            return pd.Series(X).astype(str)
        elif isinstance(X, str):
            return pd.Series([X])
        else:  # numpy array
            return pd.Series(X).astype(str)

    def fit(self, X, y=None):
        X_series = self._ensure_series(X)
        # X_series = X.squeeze().astype(str)
        self.vectorizer.fit(X_series)
        return self

    def transform(self, X):
        X_series = self._ensure_series(X)
        # X_series = X.squeeze()
        return self.vectorizer.transform(X_series).toarray()
    
    def get_feature_names_out(self, input_features=None):
        return self.vectorizer.get_feature_names_out()


## preprocess
def preprocess_data(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
    data: pd.DataFrame,
    *,
    use_polynomial: bool = False,
) -> tuple[pd.DataFrame, ...] | pd.DataFrame:

    numeric_cols = ['age', 'years_of_experience']
    onehot_cols = ['gender']
    ordinal_cols = ['education_level']
    ordinal_order = ['unknown',
                     'high school',
                     'bachelor',
                     'master',
                     'phd']

    target_cols = ['job_title']

    set_config(transform_output='pandas')

    numeric_pipe = Pipeline([
        ('scaler', MinMaxScaler()),
    ])

    onehot_pipe = Pipeline([
        ('one_hot_encoder', OneHotEncoder(sparse_output=False,
                                  handle_unknown='ignore'))
    ])

    ordinal_pipe = Pipeline([
        ('ordinal_encoder', OrdinalEncoder(categories=[ordinal_order]))
    ])

    target_pipe = Pipeline([
        ('target_encoder', TargetEncoder()),
        ('target_scaler', MinMaxScaler()),
    ])
    
    title_pipe = Pipeline([
        ('job', TfidfWrapper(max_features=64)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, numeric_cols),
            ('cat_1', onehot_pipe, onehot_cols),
            ('cat_2', ordinal_pipe, ordinal_cols),
            ('target', target_pipe, target_cols),
            ('job', title_pipe, ['job_title']),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
        sparse_threshold=0,
    )

    X_train_ = preprocessor.fit_transform(X_train, y_train)

    data_ = preprocessor.transform(data)

    if use_polynomial:
        poly_cols = X_train_.columns.difference(['gender_female',
                                                 'gender_male',
                                                 'gender_other'])

        poly_pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', MinMaxScaler()),
        ])

        poly_transformer = ColumnTransformer(
            transformers=[
                ('num_poly', poly_pipe, poly_cols),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False,
            sparse_threshold=0,
        )

        X_train_ = poly_transformer.fit_transform(X_train_, y_train)
        data_ = poly_transformer.transform(data_)

    return X_train_, data_


if __name__ == "__main__":
    from data_cleansing import cleaning_data
    from data_spliting import spliting_data

    ## load csv 
    FILE_NAME = "../database/Salary_Data.csv"
    df = pd.read_csv(FILE_NAME, delimiter=',')
    df = cleaning_data(df, has_target_columns=True)
    X_train, X_test, y_train, y_test = spliting_data(df)
    print(X_train)
    print(X_test)

    # # test 1
    # X_train_, X_test_ = preprocess_data(X_train, y_train, X_test,
    #                                     use_polynomial=True)
    # print(X_train_)
    # print(X_test_)
    # test 2
    X_train_, X_test_ = preprocess_data(X_train, y_train, X_test, use_polynomial=False)
    print(X_train_)
    print(X_test_)

    # exam = pd.DataFrame([{
    #     'age': 20,
    #     'gender': 'female',
    #     'education_level': 'PhD',
    #     'job_title': 'Data Engineer',
    #     'years_of_experience': 1,
    # }])
    # # test 3
    # _, exam_ = preprocess_data(X_train, y_train, exam, use_polynomial=True)
    # print(exam_)
    # # test 4
    # _, exam_ = preprocess_data(X_train, y_train, exam, use_polynomial=False)
    # print(exam_)
    pass