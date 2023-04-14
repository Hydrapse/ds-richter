import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

DATA_DIR = "/home/gangda/workspace/ds-richter/data"

train_values = pd.read_csv(DATA_DIR + '/train_values.csv', index_col='building_id')
train_labels = pd.read_csv(DATA_DIR + '/train_labels.csv', index_col='building_id')
test_values = pd.read_csv(DATA_DIR + '/test_values.csv', index_col='building_id')

print("Resample the train test split & try these parameters set")
X_train = train_values
y_train = train_labels
X_test = test_values

cat_cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'land_surface_condition', 'foundation_type',
            'roof_type',
            'ground_floor_type', 'other_floor_type', 'position',
            'plan_configuration', 'legal_ownership_status', 'count_floors_pre_eq', 'has_superstructure_adobe_mud',
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

best_params = {
    'depth': 8,
    'iterations': 5000,
    'learning_rate': 0.06,
    # 'l2_leaf_reg': 9,
    # 'border_count': 15,
}
model = CatBoostClassifier(eval_metric='TotalF1', task_type="GPU", cat_features=cat_cols,
                           random_seed=42, silent=False, devices=[2], **best_params)
model.fit(X_train, y_train)
preds = model.predict(X_test)

submission_format = pd.read_csv(DATA_DIR + '/submission_format.csv', index_col='building_id')
my_submission = pd.DataFrame(data=preds,
                             columns=submission_format.columns,
                             index=submission_format.index)
my_submission['damage_grade'] = my_submission['damage_grade'].astype(int)
my_submission.to_csv(DATA_DIR + '/submission/catboost_full_v1.csv')
