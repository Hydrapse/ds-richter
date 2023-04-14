import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling
import cufflinks as cf
import plotly.offline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
pd.set_option('display.max_columns', 38)

DATA_DIR = "/home/gangda/workspace/ds-richter/data"

train_values = pd.read_csv(DATA_DIR + '/train_values.csv', index_col='building_id')
train_labels = pd.read_csv(DATA_DIR + '/train_labels.csv', index_col='building_id')
test_values = pd.read_csv(DATA_DIR + '/test_values.csv', index_col='building_id')

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

print(format('How to find optimal parameters for CatBoost using GridSearchCV for Classification', '*^82'))
# Split the training data set
X = train_values
y = train_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print('Data loaded')

# Initialise the catboost classifier, if you have no GPU on your machine you can remove task_type="GPU"
model = CatBoostClassifier(eval_metric='TotalF1', task_type="GPU", cat_features=cat_cols, devices=[3])

# Choose parameters to test here
parameters = {'depth': [[2, 4, 6, 8, 10, 12]],
              'iterations': [5000, 6000, 7000, 8000],
              'learning_rate': [0.02, 0.05, 0.06, 0.07],
              'l2_leaf_reg': [3, 5, 7, 9],
              'border_count': [11, 13, 15, 17]}
print('Parameters defined')

# Initialise the Gridsearch, cv is set to 2 for speed.
randm = GridSearchCV(estimator=model, param_grid=parameters, cv=2, n_jobs=1)
randm.fit(X_train, y_train)

# Results from Random Search
print("\n========================================================")
print(" Results from Random Search ")
print("========================================================")

print("\n The best estimator across ALL searched params:\n",
      randm.best_estimator_)

print("\n The best score across ALL searched params:\n",
      randm.best_score_)

print("\n The best parameters across ALL searched params:\n",
      randm.best_params_)

print("\n ========================================================")
