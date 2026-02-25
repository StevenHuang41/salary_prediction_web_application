from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
)
def build():
    numeric_features = ["age", "years_of_experience"]

    categorical_features = [
        "gender",
        "education_level",
        "job_seniority",
        "job_title",
        "job_group",
        "job_role"
    ]

	# # method 1
	# tdf = df.copy()
	# std_scaler = StandardScaler()
	# tdf[numeric_features] = std_scaler.fit_transform(tdf[numeric_features])

	# le_encoders = {}
	# for col in categorical_features:
	#     le = LabelEncoder()
	#     tdf[col] = le.fit_transform(tdf[col])
	#     le_encoders[col] = le
	# tdf

	# method 2
    nn_pre = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ), categorical_features),
    ], verbose_feature_names_out=False)
    return nn_pre



