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
    
def get_numeric_pipeline(
    *,
    use_scaler: bool,
    scaler: Literal['standard', 'minmax'],
    math_method: Literal['log', 'sqrt', '1/x', 'square', 'cube', 'exp'] | None,
):
    steps = []

    if math_method is not None:
        steps.append(('math', MathTransformer(method=math_method)))
        
    if use_scaler:
        if scaler == 'standard':
            scaler_object = StandardScaler()
        else :
            scaler_object = MinMaxScaler()
            
        steps.append(('scaler', scaler_object))
        
    if not steps:
        return 'passthrough'
    
    return Pipeline(steps)

edu_order = ['high school', 'bachelor', 'master', 'phd']
seniority_order = ['junior', 'mid', 'senior', 'director', 'vp_clevel_principal']

def get_ordinal_pipeline(
    *,
    use_scaler: bool,
    scaler: Literal['standard', 'minmax'],
    order_list: list[str],
):
    steps = [('ordiE', OrdinalEncoder(categories=[order_list]))]
    if use_scaler:
        if scaler == 'standard':
            scaler_object = StandardScaler()
        else :
            scaler_object = MinMaxScaler()
            
        steps.append(('scaler', scaler_object))
    
    return Pipeline(steps)

def scaler_wrapper(
    transformer,
    *,
    use_scaler: bool,
    scaler: Literal['standard', 'minmax'],
    # math_method: Literal['log', 'sqrt', '1/x', 'square', 'cube', 'exp'] | None,
    # order_list: list[str],
):
    steps = [('transformer', transformer)]










    if use_scaler:
        if scaler == 'standard':
            scaler_object = StandardScaler()
        else :
            scaler_object = MinMaxScaler()
            
        steps.append(('scaler', scaler_object))
    



## preprocess
def build_preprocessor(
    *,
    use_scaler: bool = True,
    scaler: Literal["standard", "minmax"] = "standard",
    # math_method: Literal['log', 'sqrt', '1/x', 'square', 'cube', 'exp'] | None = None,
):
    
    age_pipe = get_numeric_pipeline(
        use_scaler=use_scaler,
        scaler=scaler,
        math_method='1/x',
    )
    
    gen_pipe = Pipeline([
        ('gen_OHE', OneHotEncoder(drop='first', sparse_output=False)),
    ])
    
    edu_pipe = get_ordinal_pipeline(
        use_scaler=use_scaler,
        scaler=scaler,
        order_list=edu_order,
    )
    
    job_pipe = Pipeline([
        ('job_encode', TfidfWrapper(max_features=64)),
    ])

    edu_pipe = get_ordinal_pipeline(
        use_scaler=use_scaler,
        scaler=scaler,
        order_list=seniority_order,
    )
    
    group_pipe = Pipeline([
        ('group', TargetEncoder()),
    ])
    
    year_pipe = get_numeric_pipeline(
        use_scaler=use_scaler,
        scaler=scaler,
        math_method='sqrt',
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
