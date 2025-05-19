import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import argparse
import os
from pathlib import Path

def load_player_stats(stats_file):
    """
    Load player stats from a JSON file.
    """
    with open(stats_file, 'r') as f:
        return json.load(f)
    
def load_models():
    """
    Load the trained XGBoost models for RAPTOR prediction.
    """
    off_model = xgb.XGBRegressor()
    off_model.load_model('models/offensive_raptor_model.json')
    
    def_model = xgb.XGBRegressor()
    def_model.load_model('models/defensive_raptor_model.json')
    
    feature_names = joblib.load('models/feature_names.pkl')
    
    return off_model, def_model, feature_names

def prepare_player_features(player_stats, feature_names):
    """
    Prepare player features for prediction.
    """
    # Filter out identifiers and non-feature stats
    excluded_features = {
        "EntityId", "TeamId", "Name", "ShortName", "RowId", "TeamAbbreviation", 
        "SecondsPlayed", "Minutes", "TotalPoss", "OffPoss", "DefPoss",
        "PenaltyOffPoss", "PenaltyDefPoss", "SecondChanceOffPoss",
    }
    
    features = {}
    for player in player_stats:
        # Skip any non-dict items
        if not isinstance(player, dict):
            continue
            
        player_name = player.get('Name', 'Unknown Player')
        
        # Extract features
        player_features = {k: v for k, v in player.items() 
                          if k not in excluded_features and isinstance(v, (int, float))}
        
        # Create a row with all needed features, filling missing features with 0
        feature_row = {feat: 0 for feat in feature_names}
        feature_row.update(player_features)
        
        features[player_name] = feature_row
        
    return features

def predict_raptor(player_features, off_model, def_model, feature_names):
    """
    Predict RAPTOR ratings for players.
    """
    results = []
    
    for player_name, features in player_features.items():
        # Create DataFrame with features in correct order
        df = pd.DataFrame([features])
        df = df[feature_names]  # Ensure features are in correct order
        
        # Predict
        off_raptor = off_model.predict(df)[0]
        def_raptor = def_model.predict(df)[0]
        total_raptor = off_raptor + def_raptor
        
        results.append({
            'player_name': player_name,
            'offensive_raptor': float(off_raptor),
            'defensive_raptor': float(def_raptor),
            'total_raptor': float(total_raptor)
        })
    
    # Sort by total RAPTOR
    results.sort(key=lambda x: x['total_raptor'], reverse=True)
    return results

def main():
    parser = argparse.ArgumentParser(description='Predict RAPTOR ratings from player stats')
    parser.add_argument('--input', type=str, required=True, help='Path to player stats JSON file')
    parser.add_argument('--output', type=str, help='Path to save predictions (optional)')
    
    args = parser.parse_args()
    
    # Check if models exist
    if not os.path.exists('models/offensive_raptor_model.json') or \
       not os.path.exists('models/defensive_raptor_model.json'):
        print("Error: Model files not found. Please train the models first.")
        return
        
    # Load models
    print("Loading models...")
    off_model, def_model, feature_names = load_models()
    
    # Load player stats
    print(f"Loading player stats from {args.input}...")
    player_stats = load_player_stats(args.input)
    
    # Extract season information from filename for reference
    try:
        file_season = os.path.basename(args.input).replace('totals_', '').replace('.json', '')
        raptor_season = str(int(file_season) + 1)
        print(f"File corresponds to season {file_season} (RAPTOR season: {raptor_season})")
    except:
        print("Could not extract season information from filename.")
    
    # Prepare features
    print("Preparing player features...")
    player_features = prepare_player_features(player_stats, feature_names)
    
    if not player_features:
        print("Error: No valid player data found in the input file.")
        return
        
    # Make predictions
    print("Predicting RAPTOR ratings...")
    predictions = predict_raptor(player_features, off_model, def_model, feature_names)
    
    # Display results
    print("\nPredicted RAPTOR Ratings:")
    print("-" * 70)
    print(f"{'Player':<25} {'Offensive':<12} {'Defensive':<12} {'Total':<12}")
    print("-" * 70)
    
    for pred in predictions:
        print(f"{pred['player_name']:<25} {pred['offensive_raptor']:< 12.2f} {pred['defensive_raptor']:< 12.2f} {pred['total_raptor']:< 12.2f}")
    
    # Save predictions if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"\nPredictions saved to {args.output}")

if __name__ == "__main__":
    main() 