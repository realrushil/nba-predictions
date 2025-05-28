import pandas as pd
import numpy as np
import json
import os
import glob
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time
from scipy.stats import randint, uniform
from fuzzywuzzy import fuzz, process

# Define stats that should NOT be used as features
excluded_features = {
    # Identifiers and team info
    "EntityId", "TeamId", "Name", "ShortName", "RowId", "TeamAbbreviation", 
    # Game-specific metrics that shouldn't be used as features
    "SecondsPlayed", "Minutes", "TotalPoss", "OffPoss", "DefPoss",
    "PenaltyOffPoss", "PenaltyDefPoss", "SecondChanceOffPoss",
}

def select_features(X, y, n_features=30):
    """
    Select the most important features using XGBoost's feature importance.
    
    Args:
        X: Feature DataFrame
        y: Target variable
        n_features: Number of features to select
        
    Returns:
        list: Selected feature names
    """
    print("\nPerforming feature selection...")
    # Train a simple XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X, y)
    
    # Get feature importance scores
    importance_scores = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_scores
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Select top features
    selected_features = feature_importance['feature'].head(n_features).tolist()
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance_scores)), feature_importance['importance'])
    plt.xticks(range(len(importance_scores)), feature_importance['feature'], rotation=90)
    plt.title('Feature Importance Scores')
    plt.tight_layout()
    plt.savefig('models/initial_feature_importance.png')
    plt.close()
    
    print(f"\nTop {n_features} features selected:")
    for i, (feature, importance) in enumerate(feature_importance.head(n_features).values, 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    return selected_features

def create_param_distributions():
    """
    Create parameter distributions for randomized search.
    Using distributions instead of fixed values allows for more thorough exploration.
    """
    param_dist = {
        'n_estimators': randint(50, 500),  # Wider range for number of trees
        'learning_rate': uniform(0.01, 0.3),  # Continuous distribution for learning rate
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 7),
        'subsample': uniform(0.6, 0.4),  # Range from 0.6 to 1.0
        'colsample_bytree': uniform(0.6, 0.4),  # Range from 0.6 to 1.0
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),  # L1 regularization
        'reg_lambda': uniform(0, 1)   # L2 regularization
    }
    return param_dist

def evaluate_model(model, X, y, prefix=""):
    """
    Evaluate model performance with multiple metrics
    """
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    
    print(f"{prefix} Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

def find_best_match(name, choices, min_score=85):
    """Find the best matching name from choices using fuzzy matching"""
    if not choices:
        return None
    best_match = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    if best_match and best_match[1] >= min_score:
        return best_match[0]
    return None

def tune_model(X_train, y_train, X_val, y_val, model_type="Offensive"):
    """
    Tune XGBoost model using randomized search with cross-validation
    """
    print(f"\nTuning {model_type} RAPTOR model...")
    start_time = time.time()
    
    # Create base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        early_stopping_rounds=20  # Add early stopping here
    )
    
    # Get parameter distributions
    param_dist = create_param_distributions()
    
    # Create TimeSeriesSplit for temporal cross-validation
    # This is more appropriate for sports data where we want to predict future performance
    tscv = TimeSeriesSplit(n_splits=5, test_size=len(X_train) // 5)
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=100,  # Number of parameter settings sampled
        scoring='neg_mean_squared_error',
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )
    
    # Fit RandomizedSearchCV
    random_search.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Get best parameters and scores
    best_params = random_search.best_params_
    cv_results = random_search.cv_results_
    
    # Print detailed results
    print(f"\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print("\nCross-validation results:")
    print(f"Mean test score: {-random_search.best_score_:.4f} MSE")
    print(f"Standard deviation: {cv_results['std_test_score'][random_search.best_index_]:.4f}")
    
    # Train final model with best parameters
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        early_stopping_rounds=20,  # Add early stopping here too
        **best_params
    )
    
    # Add early stopping to final model
    final_model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate final model
    train_metrics = evaluate_model(final_model, X_train, y_train, "Training")
    val_metrics = evaluate_model(final_model, X_val, y_val, "Validation")
    
    print(f"\nTime elapsed: {(time.time() - start_time) / 60:.2f} minutes")
    
    # Create feature importance plot
    plt.figure(figsize=(12, 6))
    xgb.plot_importance(final_model, max_num_features=20)
    plt.title(f'{model_type} RAPTOR - Feature Importance')
    plt.tight_layout()
    plt.savefig(f'models/{model_type.lower()}_feature_importance.png')
    plt.close()
    
    return final_model, best_params, train_metrics, val_metrics

def main():
    # Load and prepare data
    print("Loading data...")
    
    # Create the models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load RAPTOR ratings
    raptor_df = pd.read_csv('data/raptor/modern_RAPTOR_by_player.csv')
    
    # Create season-specific player name lists for matching
    raptor_players_by_season = {}
    for season in raptor_df['season'].unique():
        season_players = raptor_df[raptor_df['season'] == season]['player_name'].tolist()
        raptor_players_by_season[str(season)] = season_players
    
    # Load normalized player statistics
    all_player_data = []
    match_stats = {'total': 0, 'matched': 0}
    
    for json_file in glob.glob("data/normalized/totals_*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Extract season from filename (e.g., 2013 from totals_2013.json)
        file_season = os.path.basename(json_file).replace('totals_', '').replace('.json', '')
        # Convert to RAPTOR season format (e.g., 2013 -> 2014)
        raptor_season = str(int(file_season) + 1)
        
        # Get RAPTOR players for this season
        raptor_players = raptor_players_by_season.get(raptor_season, [])
        if not raptor_players:
            print(f"No RAPTOR data for season {raptor_season}, skipping {file_season}")
            continue
            
        print(f"Processing season {file_season} (RAPTOR season {raptor_season})")
        
        for player in data:
            match_stats['total'] += 1
            player_name = player.get('Name', '')
            if not player_name:
                continue
                
            # Find matching RAPTOR player name
            raptor_name = find_best_match(player_name, raptor_players)
            if not raptor_name:
                continue
                
            # Get RAPTOR ratings for this player-season
            raptor_data = raptor_df[
                (raptor_df['player_name'] == raptor_name) & 
                (raptor_df['season'] == int(raptor_season))
            ].iloc[0]
            
            # Create feature dict excluding non-feature stats
            player_features = {k: v for k, v in player.items() 
                             if k not in excluded_features and isinstance(v, (int, float))}
            
            # Add RAPTOR ratings
            player_features['raptor_offense'] = raptor_data['raptor_offense']
            player_features['raptor_defense'] = raptor_data['raptor_defense']
            player_features['season'] = int(file_season)
            
            all_player_data.append(player_features)
            match_stats['matched'] += 1
    
    print(f"\nMatched {match_stats['matched']} of {match_stats['total']} player-seasons "
          f"({match_stats['matched']/match_stats['total']*100:.1f}%)")
    
    if not all_player_data:
        print("Error: No matched player data found!")
        return
    
    # Convert to DataFrame
    player_df = pd.DataFrame(all_player_data)
    
    # Define features and targets
    X = player_df.drop(['raptor_offense', 'raptor_defense', 'season'], axis=1)
    y_offense = player_df['raptor_offense']
    y_defense = player_df['raptor_defense']
    
    # Perform feature selection
    print("\nSelecting features for offensive model...")
    offensive_features = select_features(X, y_offense)
    print("\nSelecting features for defensive model...")
    defensive_features = select_features(X, y_defense)
    
    # Save selected features
    feature_selection = {
        'offensive_features': offensive_features,
        'defensive_features': defensive_features
    }
    with open('models/selected_features.json', 'w') as f:
        json.dump(feature_selection, f, indent=2)
    
    # Use selected features
    X_off = X[offensive_features]
    X_def = X[defensive_features]
    
    # Create train/validation/test split for offensive model
    X_off_train, X_off_temp, y_off_train, y_off_temp = train_test_split(
        X_off, y_offense, test_size=0.3, random_state=42)
    X_off_val, X_off_test, y_off_val, y_off_test = train_test_split(
        X_off_temp, y_off_temp, test_size=0.5, random_state=42)
    
    # Create train/validation/test split for defensive model
    X_def_train, X_def_temp, y_def_train, y_def_temp = train_test_split(
        X_def, y_defense, test_size=0.3, random_state=42)
    X_def_val, X_def_test, y_def_val, y_def_test = train_test_split(
        X_def_temp, y_def_temp, test_size=0.5, random_state=42)
    
    print(f"\nTraining set size: {len(X_off_train)}")
    print(f"Validation set size: {len(X_off_val)}")
    print(f"Test set size: {len(X_off_test)}")
    
    # Tune and train models with selected features
    off_model, off_params, off_train_metrics, off_val_metrics = tune_model(
        X_off_train, y_off_train, X_off_val, y_off_val, "Offensive")
    
    def_model, def_params, def_train_metrics, def_val_metrics = tune_model(
        X_def_train, y_def_train, X_def_val, y_def_val, "Defensive")
    
    # Final evaluation on test set
    print("\nFinal Test Set Evaluation:")
    off_test_metrics = evaluate_model(off_model, X_off_test, y_off_test, "Offensive Test")
    def_test_metrics = evaluate_model(def_model, X_def_test, y_def_test, "Defensive Test")
    
    # Save models and parameters
    off_model.save_model('models/offensive_raptor_model.json')
    def_model.save_model('models/defensive_raptor_model.json')
    
    # Save hyperparameters and metrics
    results = {
        'offensive_model': {
            'parameters': off_params,
            'metrics': {
                'train': off_train_metrics,
                'validation': off_val_metrics,
                'test': off_test_metrics
            }
        },
        'defensive_model': {
            'parameters': def_params,
            'metrics': {
                'train': def_train_metrics,
                'validation': def_val_metrics,
                'test': def_test_metrics
            }
        }
    }
    
    with open('models/tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nModels, parameters, and results saved to the models directory")

if __name__ == "__main__":
    main() 