import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
)
from app.ml.preprocessing.transformers import (
    TfidfWrapper,
    MathTransformer,
    TextEncoder,
    FeatureCreation,
)
from category_encoders import TargetEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from typing import Literal


#     X_train: pd.DataFrame,
#     y_train: pd.DataFrame | pd.Series,
#     data: pd.DataFrame,
#     *,
#     use_polynomial: bool = False,
# ) -> tuple[pd.DataFrame, ...] | pd.DataFrame:
    # set_config(transform_output='pandas')
    

edu_order = ['high school', 'bachelor', 'master', 'phd']
seniority_order = ['junior', 'mid', 'senior', 'director', 'vp_clevel_principal']

def scaler_wrapper(
    trans_type: Literal['math', 'OHE', 'ordinal', 'tfidf', 'target'],
    math_method: Literal['log', 'sqrt', '1/x', 'square', 'cube', 'exp'] | None = None,
    order_list: list[str] | None = None,
    *,
    use_scaler: bool,
    scaler: Literal['standard', 'minmax'],
):
    steps = []
    if trans_type == 'math':
        steps.append(('mathT', MathTransformer(method=math_method)))
    elif trans_type == 'OHE':
        steps.append(('OHE', OneHotEncoder(drop='first', sparse_output=True)))
        return Pipeline(steps)
    elif trans_type == 'ordinal':
        steps.append(('ordinal', OrdinalEncoder(categories=[order_list])))
    elif trans_type == 'tfidf':
        steps.append(('tfidf', TfidfWrapper(max_features=64)))
        return Pipeline(steps)
    elif trans_type == 'target':
        steps.append(('target', TargetEncoder()))
        return Pipeline(steps)

    if use_scaler:
        if scaler == 'standard':
            scaler_object = StandardScaler()
        else :
            scaler_object = MinMaxScaler()
            
        steps.append(('scaler', scaler_object))

    if not steps:
        return 'passthrough'
    
    return Pipeline(steps)

def build_preprocessor(
    *,
    use_scaler: bool,
    scaler: Literal["standard", "minmax"],
):
    
    age_pipe = scaler_wrapper(
        trans_type='math',
        math_method='1/x',
        use_scaler=use_scaler,
        scaler=scaler,
    )
    
    gen_pipe = scaler_wrapper(
        trans_type='OHE',
    )
    
    edu_pipe = scaler_wrapper(
        trans_type='ordinal',
        order_list=edu_order,
        use_scaler=use_scaler,
        scaler=scaler,
    )
    
    job_pipe = scaler_wrapper(
        scaler=scaler,
    )

    seniority_pipe = scaler_wrapper(
        trans_type='ordinal',
        order_list=seniority_order,
        use_scaler=use_scaler,
        scaler=scaler,
    )
    
    group_pipe = scaler_wrapper(
        trans_type='target',
    )   

    year_pipe = scaler_wrapper(
        trans_type='math',
        math_method='sqrt',
        use_scaler=use_scaler,
        scaler=scaler,
    )

    preprocess_trans_pipe = ColumnTransformer([
        ('age', age_pipe, ['age']),

        ('gender', gen_pipe, ['gender']),

        ('edu_ordiE', edu_pipe, ['education_level']),

        ('job', job_pipe, ['job_title']),

        ('seniority', seniority_pipe, ['job_seniority']),

        ('group', group_pipe, ['job_group']),
        
        ('year_mathT', year_pipe, ['years_of_experience']),

    ], verbose_feature_names_out=False, sparse_threshold=0)


    full_pipe = Pipeline([
        ('f_seniority', FeatureCreation('seniority')),
        ('f_group', FeatureCreation('group')),
        ('preprocess', preprocess_trans_pipe),
    ])

    return full_pipe
