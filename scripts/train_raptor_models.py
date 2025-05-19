import pandas as pd
import numpy as np
import json
import os
import glob
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Create the models directory if it doesn't exist
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

# Load normalized player statistics and combine them with RAPTOR ratings
all_player_data = []

for json_file in glob.glob("data/normalized/totals_*.json"):
    season = os.path.basename(json_file).replace('totals_', '').replace('.json', '')
    
    print(f"Processing {json_file}...")
    
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

# Split data
X_train, X_test, y_off_train, y_off_test = train_test_split(X, y_offense, test_size=0.2, random_state=42)
_, _, y_def_train, y_def_test = train_test_split(X, y_defense, test_size=0.2, random_state=42)

# Print feature counts
print(f"Number of features: {X.shape[1]}")
print(f"Top 10 features: {X.columns[:10].tolist()}")

# Train offensive RAPTOR model
print("\nTraining Offensive RAPTOR model...")
xgb_off_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_off_model.fit(X_train, y_off_train)

# Train defensive RAPTOR model
print("\nTraining Defensive RAPTOR model...")
xgb_def_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_def_model.fit(X_train, y_def_train)

# Evaluate models
off_pred = xgb_off_model.predict(X_test)
def_pred = xgb_def_model.predict(X_test)

print("\nOffensive RAPTOR Model Evaluation:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_off_test, off_pred)):.4f}")
print(f"R²: {r2_score(y_off_test, off_pred):.4f}")

print("\nDefensive RAPTOR Model Evaluation:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_def_test, def_pred)):.4f}")
print(f"R²: {r2_score(y_def_test, def_pred):.4f}")

# Get feature importance
plt.figure(figsize=(10, 8))
plt.barh(range(10), xgb_off_model.feature_importances_[:10])
plt.yticks(range(10), [X.columns[i] for i in np.argsort(xgb_off_model.feature_importances_)[-10:]])
plt.title('Top 10 Features for Offensive RAPTOR Prediction')
plt.tight_layout()
plt.savefig('models/offensive_feature_importance.png')

plt.figure(figsize=(10, 8))
plt.barh(range(10), xgb_def_model.feature_importances_[:10])
plt.yticks(range(10), [X.columns[i] for i in np.argsort(xgb_def_model.feature_importances_)[-10:]])
plt.title('Top 10 Features for Defensive RAPTOR Prediction')
plt.tight_layout()
plt.savefig('models/defensive_feature_importance.png')

# Save models
xgb_off_model.save_model('models/offensive_raptor_model.json')
xgb_def_model.save_model('models/defensive_raptor_model.json')

print("\nModels saved to the models directory")
print("\nHyperparameter Recommendations:")
print("""
Key hyperparameters for XGBoost that can be tuned:
1. n_estimators: Number of trees (default: 100)
2. learning_rate: Step size shrinkage (default: 0.1)
3. max_depth: Maximum tree depth (default: 5)
4. subsample: Subsample ratio of training instances (default: 0.8)
5. colsample_bytree: Subsample ratio of columns (default: 0.8)
6. min_child_weight: Minimum sum of instance weight needed in a child (default: 1)
7. gamma: Minimum loss reduction required for splitting (default: 0)

To optimize these parameters, you can use GridSearchCV or Bayesian optimization.
""") 