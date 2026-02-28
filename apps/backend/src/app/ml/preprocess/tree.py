from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)


def build():
    tree_pre = ColumnTransformer([
        # age
        ('age', 'passthrough', ['age']),

        # gender
        ('gender', Pipeline([
            ('ohe', OneHotEncoder(sparse_output=False)),
        ]), ['gender']),

        # education_level
        ('education_level', Pipeline([
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
        ]), ['education_level']),

        # job v1
        # ('job_multi_hot', Pipeline([
        #     ('encode', MultiLabelWrapper()),
        # ]), ['job_title']),

        # job v2
        # ('job_title', Pipeline([
        #     ('target', TargetEncoder(
        #         target_type='continuous',
        #     )),
        # ]), ['job_title']),

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
        ]), ['job_seniority']),

        # group
        ('job_group', Pipeline([
            ('target', TargetEncoder(target_type='continuous')),
        ]), ['job_group']),

        # role
        ('job_role', Pipeline([
            ('target', TargetEncoder(target_type='continuous')),
        ]), ['job_role']),

        # year
        ('year', 'passthrough', ['years_of_experience']),
    ], verbose_feature_names_out=False)
    return tree_pre



