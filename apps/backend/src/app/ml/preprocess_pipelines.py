import pandas as pd

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
)
from category_encoders import TargetEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn import set_config


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
    ordinal_order = ['No Specified',
                     'High School',
                     'Bachelor',
                     'Master',
                     'PhD']

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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, numeric_cols),
            ('cat_1', onehot_pipe, onehot_cols),
            ('cat_2', ordinal_pipe, ordinal_cols),
            ('target', target_pipe, target_cols),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
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
        )

        X_train_ = poly_transformer.fit_transform(X_train_, y_train)
        data_ = poly_transformer.transform(data_)

    return X_train_, data_


if __name__ == "__main__":
    from data_cleansing import cleaning_data
    from data_spliting import spliting_data

    ## load csv 
    FILE_NAME = "../Salary_Data.csv"
    df = pd.read_csv(FILE_NAME, delimiter=',')
    df = cleaning_data(df, has_target_columns=True)
    X_train, X_test, y_train, y_test = spliting_data(df)
    # print(X_train)
    # print(X_test)

    # test 1
    X_train_, X_test_ = preprocess_data(X_train, y_train, X_test,
                                        use_polynomial=True)
    print(X_train_)
    print(X_test_)
    # test 2
    X_train_, X_test_ = preprocess_data(X_train, y_train, X_test,
                                        use_polynomial=False)
    print(X_train_)
    print(X_test_)

    exam = pd.DataFrame([{
        'age': 20,
        'gender': 'female',
        'education_level': 'PhD',
        'job_title': 'Data Engineer',
        'years_of_experience': 1,
    }])
    # test 3
    _, exam_ = preprocess_data(X_train, y_train, exam, use_polynomial=True)
    print(exam_)
    # test 4
    _, exam_ = preprocess_data(X_train, y_train, exam, use_polynomial=False)
    print(exam_)