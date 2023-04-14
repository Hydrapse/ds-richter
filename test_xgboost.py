import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

# DATA_DIR = "/home/gangda/workspace/ds-richter/data"
DATA_DIR = "/Users/synapse/Desktop/Repository/pycharm-workspace/ds-richter/data"


X = pd.read_csv(DATA_DIR + '/train_values.csv', index_col='building_id')
y = pd.read_csv(DATA_DIR + '/train_labels.csv', index_col='building_id')
y -= 1

# Identify categorical and numerical columns
categorical_columns = [
    "land_surface_condition", "foundation_type", "roof_type",
    "ground_floor_type", "other_floor_type", "position",
    "plan_configuration", "legal_ownership_status",
]
numerical_columns = [
    "geo_level_1_id", "geo_level_2_id", "geo_level_3_id",
    "count_floors_pre_eq", "age", "area_percentage",
    "height_percentage", "count_families"
]
other_columns = [
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber",
    "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
    "has_superstructure_other",
    "has_secondary_use",
    "has_secondary_use_agriculture",
    "has_secondary_use_hotel",
    "has_secondary_use_rental",
    "has_secondary_use_institution",
    "has_secondary_use_school",
    "has_secondary_use_industry",
    "has_secondary_use_health_post",
    "has_secondary_use_gov_office",
    "has_secondary_use_use_police",
    "has_secondary_use_other",
]


# Preprocessing pipeline
# preprocessor = ColumnTransformer(transformers=[
#     ("num", StandardScaler(), numerical_columns),
#     ("cat", OrdinalEncoder(), categorical_columns)
# ])
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_columns),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_columns),
    ("other", FunctionTransformer(validate=False), other_columns),
])

# Create a pipeline with RandomForestClassifier
model_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Create a pipeline with XGBClassifier
model_xgb = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train and evaluate the RandomForestClassifier
# model_rf.fit(X_train, y_train)
# y_pred_rf = model_rf.predict(X_test)
# print("RandomForestClassifier classification report:")
# print(classification_report(y_test, y_pred_rf))

# Train and evaluate the XGBClassifier
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
print("XGBClassifier classification report:")
print(classification_report(y_test, y_pred_xgb))

# Choose the model with the best F1-score
# f1_rf = f1_score(y_test, y_pred_rf, average='micro')
f1_xgb = f1_score(y_test, y_pred_xgb, average='micro')
print("Score on test set")
print("\n ========================================================")
# print('f1_rf', f1_rf)
print('f1_xgb', f1_xgb)


# # Preprocess the data
# X_processed = preprocessor.fit_transform(X_train)
#
# # Set up the parameter grid for XGBoost
# param_grid = {
#     "learning_rate": [0.01, 0.1, 0.2],
#     "max_depth": [3, 6, 9],
#     "min_child_weight": [1, 5, 10],
#     "subsample": [0.5, 0.7, 1.0],
#     "colsample_bytree": [0.5, 0.7, 1.0],
#     "n_estimators": [100, 200, 500],
# }
#
# # Perform Grid Search
# xgb = XGBClassifier(random_state=42, objective="multi:softmax", num_class=3, eval_metric="mlogloss")
# grid_search = GridSearchCV(
#     estimator=xgb, param_grid=param_grid, cv=5, scoring="f1_micro", verbose=1, n_jobs=-1
# )
# grid_search.fit(X_processed, y_train)
#
# # Get the best parameters
# best_params = grid_search.best_params_
# print("Best parameters found for XGBoost:")
# print(best_params)
#
# # Train and evaluate the XGBClassifier with the best parameters
# model_xgb_best = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("classifier", XGBClassifier(random_state=42, **best_params))
# ])
# model_xgb_best.fit(X_train, y_train)
# y_pred_xgb_best = model_xgb_best.predict(X_test)
# print("XGBClassifier (best parameters) classification report:")
# print(classification_report(y_test, y_pred_xgb_best))
