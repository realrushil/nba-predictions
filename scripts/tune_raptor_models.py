import pandas as pd
import numpy as np
import json
import os
import glob
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time

# Make sure models directory exists
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Load RAPTOR ratings
raptor_df = pd.read_csv('data/raptor/modern_RAPTOR_by_player.csv')

# Clean player_id and season to use as a key
raptor_df['player_season'] = raptor_df['player_id'] + '_' + raptor_df['season'].astype(str)

# Get target variables
raptor_targets = raptor_df[['player_season', 'raptor_offense', 'raptor_defense']]

# Define stats that should NOT be used as features
excluded_features = {
    # Identifiers and team info
    "EntityId", "TeamId", "Name", "ShortName", "RowId", "TeamAbbreviation", 
    # Game-specific metrics that shouldn't be used as features
    "SecondsPlayed", "Minutes", "TotalPoss", "OffPoss", "DefPoss",
    "PenaltyOffPoss", "PenaltyDefPoss", "SecondChanceOffPoss",
}

print("Loading and preparing data...")
# Load normalized player statistics and combine them with RAPTOR ratings
all_player_data = []

for json_file in glob.glob("data/normalized/totals_*.json"):
    season = os.path.basename(json_file).replace('totals_', '').replace('.json', '')
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for player in data:
        player_id = player.get('RowId', '')
        if not player_id:
            continue  # Skip if no player ID
            
        player_season = f"{player_id}_{season}"
        
        # Create feature dict excluding non-feature stats
        player_features = {k: v for k, v in player.items() if k not in excluded_features and isinstance(v, (int, float))}
        
        # Add player_season key to join with RAPTOR data
        player_features['player_season'] = player_season
        all_player_data.append(player_features)

# Convert to DataFrame
player_df = pd.DataFrame(all_player_data)

# Merge with RAPTOR targets
merged_df = pd.merge(player_df, raptor_targets, on='player_season', how='inner')
print(f"Total matched player-seasons: {len(merged_df)}")

# Drop player_season from features
X = merged_df.drop(['player_season', 'raptor_offense', 'raptor_defense'], axis=1)
y_offense = merged_df['raptor_offense']
y_defense = merged_df['raptor_defense']

# Save the feature names for later use
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.pkl')

# Split data
X_train, X_test, y_off_train, y_off_test = train_test_split(X, y_offense, test_size=0.2, random_state=42)
_, _, y_def_train, y_def_test = train_test_split(X, y_defense, test_size=0.2, random_state=42)

print(f"Number of features: {X.shape[1]}")

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

# Function to tune model and return best parameters
def tune_model(X, y, param_grid, model_type):
    print(f"\nTuning {model_type} RAPTOR model...")
    start_time = time.time()
    
    # Create base model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Create cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    print(f"Time elapsed: {(time.time() - start_time) / 60:.2f} minutes")
    
    return grid_search.best_params_, grid_search.best_estimator_

# Uncomment to run full hyperparameter tuning (warning: can be time-consuming)
# Comment out these lines to use default parameters instead
"""
# Tune offensive RAPTOR model
off_best_params, off_best_model = tune_model(X_train, y_off_train, param_grid, "Offensive")

# Tune defensive RAPTOR model
def_best_params, def_best_model = tune_model(X_train, y_def_train, param_grid, "Defensive")
"""

# Use reasonable default parameters instead of full tuning
print("\nTraining models with default parameters...")
off_best_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

def_best_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train models
off_best_model.fit(X_train, y_off_train, eval_set=[(X_test, y_off_test)], early_stopping_rounds=10, verbose=False)
def_best_model.fit(X_train, y_def_train, eval_set=[(X_test, y_def_test)], early_stopping_rounds=10, verbose=False)

# Evaluate models
off_pred = off_best_model.predict(X_test)
def_pred = def_best_model.predict(X_test)

print("\nOffensive RAPTOR Model Evaluation:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_off_test, off_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_off_test, off_pred):.4f}")
print(f"R²: {r2_score(y_off_test, off_pred):.4f}")

print("\nDefensive RAPTOR Model Evaluation:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_def_test, def_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_def_test, def_pred):.4f}")
print(f"R²: {r2_score(y_def_test, def_pred):.4f}")

# Plot feature importance
def plot_feature_importance(model, title, filename):
    # Get feature importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[-15:]  # Top 15 features
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    
plot_feature_importance(off_best_model, 'Top 15 Features for Offensive RAPTOR Prediction', 'models/offensive_feature_importance.png')
plot_feature_importance(def_best_model, 'Top 15 Features for Defensive RAPTOR Prediction', 'models/defensive_feature_importance.png')

# Save models
off_best_model.save_model('models/offensive_raptor_model.json')
def_best_model.save_model('models/defensive_raptor_model.json')

print("\nModels and feature importance plots saved to the models directory")

# Create a prediction script for future use
prediction_script = """
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

def predict_raptor(player_stats):
    # Load models
    off_model = xgb.XGBoost()
    off_model.load_model('models/offensive_raptor_model.json')
    
    def_model = xgb.XGBoost()
    def_model.load_model('models/defensive_raptor_model.json')
    
    # Load feature names
    feature_names = joblib.load('models/feature_names.pkl')
    
    # Prepare features
    features = pd.DataFrame([player_stats])
    
    # Align features with what model expects
    missing_cols = set(feature_names) - set(features.columns)
    for col in missing_cols:
        features[col] = 0
    
    features = features[feature_names]
    
    # Make predictions
    offensive_raptor = off_model.predict(features)[0]
    defensive_raptor = def_model.predict(features)[0]
    
    return {
        "offensive_raptor": offensive_raptor,
        "defensive_raptor": defensive_raptor,
        "total_raptor": offensive_raptor + defensive_raptor
    }
"""

with open('scripts/predict_raptor.py', 'w') as f:
    f.write(prediction_script)

print("\nPrediction script created at 'scripts/predict_raptor.py'")
print("\nHyperparameter Tuning Recommendations:")
print("""
To fully tune these models, uncomment the hyperparameter tuning section in this script.
This will perform a grid search over the following parameters:

1. n_estimators: [50, 100, 200] - Controls number of boosting rounds
2. learning_rate: [0.01, 0.05, 0.1, 0.2] - Step size shrinkage
3. max_depth: [3, 5, 7] - Maximum depth of trees
4. subsample: [0.6, 0.8, 1.0] - Subsample ratio of training instances
5. colsample_bytree: [0.6, 0.8, 1.0] - Subsample ratio of columns when building trees
6. min_child_weight: [1, 3, 5] - Minimum sum of instance weight needed in a child

Note: Full hyperparameter tuning can take several hours depending on your hardware.
You may want to run this overnight or on a machine with significant computing power.
""") 