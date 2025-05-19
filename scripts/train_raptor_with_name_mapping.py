import pandas as pd
import numpy as np
import json
import os
import glob
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from fuzzywuzzy import process, fuzz
import joblib
import time

print("NBA RAPTOR Prediction Model with Hyperparameter Tuning")
print("="*70)

# Create the models directory if it doesn't exist
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Define stats that should NOT be used as features
excluded_features = {
    # Identifiers and team info
    "EntityId", "TeamId", "Name", "ShortName", "RowId", "TeamAbbreviation", 
    # Game-specific metrics that shouldn't be used as features
    "SecondsPlayed", "Minutes", "TotalPoss", "OffPoss", "DefPoss",
    "PenaltyOffPoss", "PenaltyDefPoss", "SecondChanceOffPoss",
}

# Load RAPTOR ratings
print("Loading RAPTOR data...")
raptor_df = pd.read_csv('data/raptor/modern_RAPTOR_by_player.csv')
print(f"RAPTOR dataset has {len(raptor_df)} rows with {len(raptor_df['player_id'].unique())} unique players")

# Prepare season strings to match formats
raptor_df['season_str'] = raptor_df['season'].astype(str)

# Function to find the best RAPTOR match for a player name
def find_raptor_match(nba_name, raptor_season_df, threshold=85):
    """Find the best match for an NBA player name in the RAPTOR dataset for a specific season."""
    raptor_names = raptor_season_df['player_name'].tolist()
    if not raptor_names:
        return None, 0
        
    best_match, score = process.extractOne(nba_name, raptor_names, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        # Get the RAPTOR row for this match
        matched_row = raptor_season_df[raptor_season_df['player_name'] == best_match].iloc[0]
        return matched_row, score
    return None, 0

# Load all players from all seasons
print("\nProcessing player data...")
all_seasons_data = []

for json_file in glob.glob("data/normalized/totals_*.json"):
    file_season = os.path.basename(json_file).replace('totals_', '').replace('.json', '')
    # Fix the season mapping: add 1 to the year for proper RAPTOR matching
    # totals_2013.json should map to RAPTOR season "14"
    raptor_season = str(int(file_season) + 1)
    
    print(f"Processing {json_file}... (File season: {file_season}, RAPTOR season: {raptor_season})")
    
    with open(json_file, 'r') as f:
        nba_data = json.load(f)
    
    # Filter RAPTOR data for the corrected season
    raptor_season_df = raptor_df[raptor_df['season_str'] == raptor_season]
    
    if len(raptor_season_df) == 0:
        print(f"  No RAPTOR data available for season {raptor_season}, skipping")
        continue
    
    print(f"  Found {len(raptor_season_df)} RAPTOR players for season {raptor_season}")
    
    matched_count = 0
    total_players = len(nba_data)
    
    # Process each NBA player for this season
    for player in nba_data:
        nba_name = player.get('Name', '')
        if not nba_name:
            continue
            
        # Find the matching RAPTOR player
        raptor_player, score = find_raptor_match(nba_name, raptor_season_df)
        
        if raptor_player is not None:
            # Create feature dict excluding non-feature stats
            player_features = {k: v for k, v in player.items() 
                              if k not in excluded_features and isinstance(v, (int, float))}
            
            # Add RAPTOR targets
            player_features['raptor_offense'] = raptor_player['raptor_offense']
            player_features['raptor_defense'] = raptor_player['raptor_defense']
            player_features['raptor_total'] = raptor_player['raptor_total']
            player_features['player_name'] = nba_name
            player_features['raptor_name'] = raptor_player['player_name']
            player_features['season'] = file_season
            player_features['raptor_season'] = raptor_season
            player_features['match_score'] = score
            
            all_seasons_data.append(player_features)
            matched_count += 1
    
    print(f"  Matched {matched_count} of {total_players} players ({matched_count/total_players*100:.1f}%)")

# Convert to DataFrame
print("\nCreating combined dataset...")
combined_df = pd.DataFrame(all_seasons_data)
print(f"Combined dataset has {len(combined_df)} player-seasons")

# Filter only high-confidence matches
high_confidence_df = combined_df[combined_df['match_score'] >= 90]
print(f"High confidence matches (score >= 90): {len(high_confidence_df)} player-seasons")

# Use high confidence dataset for modeling
modeling_df = high_confidence_df.copy()

# Drop non-feature columns
non_features = ['raptor_offense', 'raptor_defense', 'raptor_total', 
               'player_name', 'raptor_name', 'season', 'raptor_season', 'match_score']
X = modeling_df.drop(non_features, axis=1)
y_offense = modeling_df['raptor_offense']
y_defense = modeling_df['raptor_defense']

# Save the feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.pkl')

print(f"Final dataset has {len(X)} samples with {len(feature_names)} features")

# Create three-way split: train, validation, test (70%, 15%, 15%)
X_train, X_temp, y_off_train, y_off_temp, y_def_train, y_def_temp = train_test_split(
    X, y_offense, y_defense, test_size=0.3, random_state=42)

X_val, X_test, y_off_val, y_off_test, y_def_val, y_def_test = train_test_split(
    X_temp, y_off_temp, y_def_temp, test_size=0.5, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

# Use RandomizedSearchCV for more efficient tuning
def tune_model(X_train, y_train, X_val, y_val, param_grid, model_type):
    print(f"\nTuning {model_type} RAPTOR model...")
    start_time = time.time()
    
    # First step: RandomizedSearchCV
    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # RandomizedSearchCV for more efficient tuning
    grid_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=20,  # Try 20 parameter combinations
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit on training data without early stopping
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    # Second step: Train final model with best parameters (without early stopping)
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **best_params
    )
    
    # Train on training data
    eval_set = [(X_val, y_val)]
    final_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    print(f"Time elapsed: {(time.time() - start_time) / 60:.2f} minutes")
    
    return final_model, best_params

# Tune models
print("\nStarting hyperparameter tuning (this may take some time)...")
off_model, off_params = tune_model(X_train, y_off_train, X_val, y_off_val, param_grid, "Offensive")
def_model, def_params = tune_model(X_train, y_def_train, X_val, y_def_val, param_grid, "Defensive")

# Evaluate on validation set
off_val_pred = off_model.predict(X_val)
def_val_pred = def_model.predict(X_val)

print("\nValidation Set Performance:")
print("Offensive RAPTOR Model:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_off_val, off_val_pred)):.4f}")
print(f"  MAE: {mean_absolute_error(y_off_val, off_val_pred):.4f}")
print(f"  R²: {r2_score(y_off_val, off_val_pred):.4f}")

print("Defensive RAPTOR Model:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_def_val, def_val_pred)):.4f}")
print(f"  MAE: {mean_absolute_error(y_def_val, def_val_pred):.4f}")
print(f"  R²: {r2_score(y_def_val, def_val_pred):.4f}")

# Evaluate on test set (final evaluation)
off_test_pred = off_model.predict(X_test)
def_test_pred = def_model.predict(X_test)

print("\nTest Set Performance (Final Evaluation):")
print("Offensive RAPTOR Model:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_off_test, off_test_pred)):.4f}")
print(f"  MAE: {mean_absolute_error(y_off_test, off_test_pred):.4f}")
print(f"  R²: {r2_score(y_off_test, off_test_pred):.4f}")

print("Defensive RAPTOR Model:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_def_test, def_test_pred)):.4f}")
print(f"  MAE: {mean_absolute_error(y_def_test, def_test_pred):.4f}")
print(f"  R²: {r2_score(y_def_test, def_test_pred):.4f}")

# Plot feature importance
def plot_importance(model, title, filename, top_n=15):
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)

# Plot top features
plot_importance(off_model, 'Top Features for Offensive RAPTOR Prediction', 
               'models/offensive_feature_importance.png')
               
plot_importance(def_model, 'Top Features for Defensive RAPTOR Prediction',
               'models/defensive_feature_importance.png')

# Save models
off_model.save_model('models/offensive_raptor_model.json')
def_model.save_model('models/defensive_raptor_model.json')

# Save model hyperparameters for reference
model_params = {
    'offensive_model': off_params,
    'defensive_model': def_params
}

with open('models/model_hyperparameters.json', 'w') as f:
    json.dump(model_params, f, indent=2, default=str)

# Save a sample of the predictions for inspection
sample_size = min(20, len(X_test))
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)

# Get actual test indices in the original dataframe
test_indices = modeling_df.index[X_test.index]
sample_test_indices = test_indices[sample_indices]

sample_df = pd.DataFrame({
    'player_name': modeling_df.iloc[sample_test_indices]['player_name'].values,
    'season': modeling_df.iloc[sample_test_indices]['season'].values,
    'true_off_raptor': y_off_test.iloc[sample_indices].values,
    'pred_off_raptor': off_test_pred[sample_indices],
    'true_def_raptor': y_def_test.iloc[sample_indices].values,
    'pred_def_raptor': def_test_pred[sample_indices]
})

sample_df['off_raptor_error'] = abs(sample_df['true_off_raptor'] - sample_df['pred_off_raptor'])
sample_df['def_raptor_error'] = abs(sample_df['true_def_raptor'] - sample_df['pred_def_raptor'])

print("\nSample predictions on test set:")
print(sample_df.sort_values('off_raptor_error').head(10))

# Save sample predictions to CSV for inspection
sample_df.to_csv('models/sample_predictions.csv', index=False)

print("\nModels, hyperparameters, and visualizations saved to the models directory")

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