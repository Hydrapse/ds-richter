import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import f1_score

DATA_DIR = "/home/gangda/workspace/ds-richter/data"

X = pd.read_csv(DATA_DIR + '/train_values.csv', index_col='building_id')
y = pd.read_csv(DATA_DIR + '/train_labels.csv', index_col='building_id')
X_test = pd.read_csv(DATA_DIR + '/test_values.csv', index_col='building_id')

# preprocess
non_numeric_columns = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
                            'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
X = pd.get_dummies(X, columns=non_numeric_columns)
y = y - 1

# split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

cat_cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'count_floors_pre_eq', 'has_superstructure_adobe_mud',
            'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
            'has_superstructure_cement_mortar_stone',
            'has_superstructure_mud_mortar_brick',
            'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
            'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
            'has_superstructure_rc_engineered', 'has_superstructure_other', 'has_secondary_use',
            'has_secondary_use_agriculture', 'has_secondary_use_hotel',
            'has_secondary_use_rental', 'has_secondary_use_institution',
            'has_secondary_use_school', 'has_secondary_use_industry',
            'has_secondary_use_health_post', 'has_secondary_use_gov_office',
            'has_secondary_use_use_police', 'has_secondary_use_other',
            'land_surface_condition_n', 'land_surface_condition_o',
            'land_surface_condition_t', 'foundation_type_h', 'foundation_type_i',
            'foundation_type_r', 'foundation_type_u', 'foundation_type_w',
            'roof_type_n', 'roof_type_q', 'roof_type_x', 'ground_floor_type_f',
            'ground_floor_type_m', 'ground_floor_type_v', 'ground_floor_type_x',
            'ground_floor_type_z', 'other_floor_type_j', 'other_floor_type_q',
            'other_floor_type_s', 'other_floor_type_x', 'position_j', 'position_o',
            'position_s', 'position_t', 'plan_configuration_a',
            'plan_configuration_c', 'plan_configuration_d', 'plan_configuration_f',
            'plan_configuration_m', 'plan_configuration_n', 'plan_configuration_o',
            'plan_configuration_q', 'plan_configuration_s', 'plan_configuration_u',
            'legal_ownership_status_a', 'legal_ownership_status_r',
            'legal_ownership_status_v', 'legal_ownership_status_w']


def evaluate_microF1_lgb(preds, train_data):
    labels = train_data.get_label()
    num_classes = len(np.unique(labels))
    preds = preds.reshape(num_classes, -1).argmax(axis=0)
    f1 = f1_score(labels, preds, average='micro')
    return 'microF1', f1, True


params = {
    'objective': 'multiclass',
    'num_class': 3,
    'boosting': 'gbdt',
    'metric': 'None',
    # "num_leaves": 1280,
    # "learning_rate": 0.05,
    # "feature_fraction": 0.85,
    # "reg_lambda": 2,
}

d_training = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
d_test = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, free_raw_data=False)

best_model = lgb.train(params, train_set=d_training, valid_sets=[d_training, d_test], feval=evaluate_microF1_lgb,
                       verbose_eval=20, early_stopping_rounds=100, num_boost_round=1000)

val_preds = best_model.predict(X_val)
val_preds = val_preds.argmax(axis=1)
accuracy = f1_score(y_val, val_preds, average='micro')
print("Score on test set")
print("\n ========================================================")
print(accuracy)
