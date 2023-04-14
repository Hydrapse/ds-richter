import logging
import sys
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import f1_score

DATA_DIR = "/home/gangda/workspace/ds-richter/data"

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "lightgbm_v1"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

X = pd.read_csv(DATA_DIR + '/train_values.csv', index_col='building_id')
y = pd.read_csv(DATA_DIR + '/train_labels.csv', index_col='building_id')
X_test = pd.read_csv(DATA_DIR + '/test_values.csv', index_col='building_id')

# preprocess
non_numeric_columns = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
                            'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
X = pd.get_dummies(X, columns=non_numeric_columns)
y = y - 1

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


def objective(trial):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    d_training = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    d_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, free_raw_data=False)

    param = {
        'objective': 'multiclass',
        'num_class': 3,
        'boosting': 'gbdt',
        'metric': 'None',
        'seed': 42,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        # "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, d_training, feval=evaluate_microF1_lgb, valid_sets=[d_training, d_val],
                    verbose_eval=False, num_boost_round=2000, early_stopping_rounds=200)
    preds = gbm.predict(X_val).argmax(axis=1)
    accuracy = f1_score(y_val, preds, average='micro')
    return accuracy


study = optuna.create_study(study_name=study_name, storage=storage_name,
                            sampler=TPESampler(), direction="maximize", load_if_exists=True)
study.optimize(objective, n_trials=1000)
print("Number of completed trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial

print("\tBest Score: {}".format(trial.value))
print("\tBest Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
# study.optimize(objective, n_trials=3)
