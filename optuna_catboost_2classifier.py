import logging
import sys
import pandas as pd
import os.path as osp
from catboost import CatBoostClassifier
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import optuna
from optuna.integration import CatBoostPruningCallback

DATA_DIR = "/home/gangda/workspace/ds-richter/data"
split_set = "1_2"

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "catboost_v3_{}".format(split_set)  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

X = pd.read_csv(osp.join(DATA_DIR, 'train_{}_values.csv').format(split_set), index_col='building_id')
y = pd.read_csv(osp.join(DATA_DIR, 'train_{}_labels.csv').format(split_set), index_col='building_id')

cat_cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'land_surface_condition', 'foundation_type',
            'roof_type',
            'ground_floor_type', 'other_floor_type', 'position',
            'plan_configuration', 'count_floors_pre_eq', 'has_superstructure_adobe_mud',
            'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
            'has_superstructure_cement_mortar_stone',
            'has_superstructure_mud_mortar_brick',
            'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
            'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
            'has_superstructure_rc_engineered', 'has_superstructure_other',
            'legal_ownership_status', 'has_secondary_use',
            'has_secondary_use_agriculture', 'has_secondary_use_hotel',
            'has_secondary_use_rental', 'has_secondary_use_institution',
            'has_secondary_use_school', 'has_secondary_use_industry',
            'has_secondary_use_health_post', 'has_secondary_use_gov_office',
            'has_secondary_use_use_police', 'has_secondary_use_other']


def objective(trial):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'n_estimators': 5000,
        'depth': trial.suggest_int("depth", 4, 10),
        'learning_rate': trial.suggest_categorical("learning_rate", [0.045, 0.05, 0.055, 0.06, 0.065, 0.07]),
        'border_count': trial.suggest_categorical('border_count', [11, 13, 15, 17, 254]),
        'random_strength': trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        'eval_metric': 'F1',
        'random_seed': 42,
    }

    model = CatBoostClassifier(**params, cat_features=cat_cols)
    pruning_callback = CatBoostPruningCallback(trial, "F1")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=1000,
        verbose=False,
        callbacks=[pruning_callback],
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    y_pred = model.predict(X_val)
    acc = f1_score(y_val, y_pred, average='micro')

    return acc


study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    sampler=TPESampler(),
    direction="maximize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=1000)

print("Number of completed trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial

print("\tBest Score: {}".format(trial.value))
print("\tBest Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
