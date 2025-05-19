# NBA RAPTOR Rating Prediction

This project processes NBA player statistics and uses machine learning to predict player RAPTOR ratings (developed by FiveThirtyEight) based on their performance metrics. The system converts raw stats to per-100-possession values and then normalizes them relative to league averages.

## Project Structure

- `data/`: Contains all data files
  - `raw/`: Raw player statistics by season (totals_YYYY.json)
  - `processed/`: Statistics converted to per-100-possession values
  - `normalized/`: Statistics normalized relative to league averages
  - `raptor/`: RAPTOR ratings from FiveThirtyEight

- `models/`: Contains trained models and visualizations
  - `offensive_raptor_model.json`: XGBoost model for predicting offensive RAPTOR
  - `defensive_raptor_model.json`: XGBoost model for predicting defensive RAPTOR
  - `offensive_feature_importance.png`: Visualization of features most important for offensive RAPTOR
  - `defensive_feature_importance.png`: Visualization of features most important for defensive RAPTOR

- `scripts/`: Contains all processing and modeling scripts
  - `scale_stats_to_per_100_possessions.py`: Converts raw stats to per-100 possession values
  - `normalize_to_league_average.py`: Normalizes stats relative to league averages
  - `train_raptor_with_name_mapping.py`: Trains RAPTOR prediction models
  - `predict_raptor_from_stats.py`: Makes predictions using trained models

## Data Preparation Pipeline

1. **Raw Statistics**: Player statistics from NBA.com or other sources
2. **Per-100 Possessions**: Normalize all counting stats to a per-100-possession basis
3. **League Average Normalization**: Express each stat as a deviation from league average
4. **Player Matching**: Match players with their RAPTOR ratings using fuzzy name matching
5. **Model Training**: Train XGBoost models to predict offensive and defensive RAPTOR

## Model Performance

The models achieved the following performance metrics:

- **Offensive RAPTOR Model**:
  - RMSE: 2.19
  - MAE: 1.56
  - R²: 0.35

- **Defensive RAPTOR Model**:
  - RMSE: 2.47
  - MAE: 1.54
  - R²: 0.16

## Usage

### Processing Raw Data

```bash
# Convert raw stats to per-100 possessions
python scripts/scale_stats_to_per_100_possessions.py

# Normalize stats relative to league average
python scripts/normalize_to_league_average.py
```

### Training Models

```bash
# Train both offensive and defensive RAPTOR prediction models
python scripts/train_raptor_with_name_mapping.py
```

### Making Predictions

```bash
# Predict RAPTOR ratings for players in a specific season
python scripts/predict_raptor_from_stats.py --input "data/normalized/totals_2022.json" --output "models/predictions_2022.json"
```

## Hyperparameter Tuning

The XGBoost models use default hyperparameters, but can be tuned for better performance:

1. n_estimators: Number of trees (default: 100)
2. learning_rate: Step size shrinkage (default: 0.1)
3. max_depth: Maximum tree depth (default: 5)
4. subsample: Subsample ratio of training instances (default: 0.8)
5. colsample_bytree: Subsample ratio of columns (default: 0.8)
6. min_child_weight: Minimum sum of instance weight needed in a child (default: 1)
7. gamma: Minimum loss reduction required for splitting (default: 0)

To implement full hyperparameter tuning, edit `scripts/tune_raptor_models.py` and uncomment the tuning section.

## Requirements

- Python 3.6+
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- fuzzywuzzy
- python-Levenshtein
- joblib

Install the required packages:

```bash
pip install pandas numpy xgboost scikit-learn matplotlib fuzzywuzzy python-Levenshtein joblib
``` 