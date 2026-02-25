from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)

def build():
    linear_pre = ColumnTransformer([
        # age drop

        # gender
        ('gender', Pipeline([
            ('ohe', OneHotEncoder(drop='first', sparse_output=False)),
        ]), ['gender']),

        # education_level
        ('edu_level', Pipeline([
            ('ordinal', OrdinalEncoder(
                categories=[[ # type: ignore
                    'unknown',
                    'high_school',
                    'bachelor',
                    'master',
                    'phd'
                ]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
            ('scaler', StandardScaler()),
        ]), ['education_level']),

        # job v1
        # ('job_multi_hot', Pipeline([
        #     ('encode', MultiLabelWrapper()),
        # ]), ['job_title']),

        # job v2
        ('job_title', Pipeline([
            ('target', TargetEncoder(target_type='continuous')),
            ('scaler', StandardScaler()),
        ]), ['job_title']),

        # seniority
        ('job_seniority', Pipeline([
            
            ('ordinal', OrdinalEncoder(
                categories=[[ # type: ignore
                    'junior',
                    'mid',
                    'senior',
                    'director_vp',
                    'c_level'
                ]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
            ('scaler', StandardScaler()),
        ]), ['job_seniority']),

        # group
        ('job_group', Pipeline([
            ('target', TargetEncoder(target_type='continuous')),
            ('scaler', StandardScaler()),
        ]), ['job_group']),

        # role
        ('job_role', Pipeline([
            ('target', TargetEncoder(target_type='continuous')),
            ('scaler', StandardScaler()),
        ]), ['job_role']),

        # year
        ('year', Pipeline([
            ('scaler', StandardScaler()),
        ]), ['years_of_experience']),
    ], verbose_feature_names_out=False)
    return linear_pre


