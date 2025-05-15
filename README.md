# NBA Player Minutes & Lineup Predictor

A machine learning system to predict NBA player minutes, rotations, and lineups for upcoming games. This project utilizes historical game data to predict:

- Player minutes per game
- Starting lineups
- Rotation patterns throughout the game
- Depth charts by position

## Project Structure

```
nba-predictions/
├── data/
│   ├── processed/         # Processed data (depth charts, predictions)
│   └── raw/               # Raw data from NBA API
├── modules/               # Core functionality modules
│   ├── data_collection.py # Data collection from NBA API
│   ├── minutes_processor.py # Process minutes data and generate depth charts
│   ├── minutes_predictor.py # Predict minutes and lineups
│   └── injury_tracker.py  # Track player injuries
├── scripts/               # Command-line scripts
│   ├── generate_depth_charts.py # Generate team depth charts
│   ├── predict_minutes.py       # Predict minutes for upcoming games
│   ├── analyze_lineups.py       # Analyze lineup data
│   └── manage_injuries.py       # Manage player injuries
├── tests/                 # Unit tests
│   ├── test_data_collection.py
│   ├── test_minutes_processor.py
│   ├── test_minutes_predictor.py
│   ├── test_injury_tracker.py
│   └── run_tests.py       # Test runner script
└── README.md              # Project documentation
```

## Requirements

- Python 3.7+
- pandas
- numpy
- nba_api

Install required packages:

```bash
pip install pandas numpy nba_api
```

## Usage

### 1. Generate Depth Charts

First, generate team depth charts based on recent player minutes:

```bash
python scripts/generate_depth_charts.py --season 2023-24 --days 30 --window 5
```

Arguments:
- `--season`: NBA season in format YYYY-YY (default: "2023-24")
- `--days`: Number of days of game data to fetch (default: 30)
- `--window`: Window size for rolling minutes calculation (default: 5)
- `--output`: Output file path for depth charts (default: "data/processed/depth_charts.csv")

### 2. Predict Player Minutes

Predict minutes for players on a specific team for an upcoming game:

```bash
python scripts/predict_minutes.py --team GSW --opponent LAL --date 2024-03-15
```

Arguments:
- `--season`: NBA season in format YYYY-YY (default: "2023-24")
- `--days`: Number of days of historical data to use (default: 30)
- `--team`: Team abbreviation (required, e.g., "GSW")
- `--opponent`: Opponent team abbreviation (optional)
- `--date`: Game date in format YYYY-MM-DD (default: tomorrow)
- `--depth-chart`: Path to depth chart CSV file (default: "data/processed/depth_charts.csv")
- `--b2b`: Flag for back-to-back game
- `--injuries`: Path to injury list JSON file
- `--output-dir`: Output directory for predictions (default: "data/processed/predictions")

### 3. Analyze Lineup Data

Collect and analyze NBA lineup data:

```bash
python scripts/analyze_lineups.py --team GSW --min-minutes 50
```

Arguments:
- `--season`: NBA season in format YYYY-YY (default: "2023-24")
- `--team`: Team abbreviation to filter (e.g., "GSW")
- `--min-minutes`: Minimum minutes played to include in analysis (default: 50)
- `--output-dir`: Output directory for lineup data (default: "data/processed/lineups")

### 4. Manage Injuries

Track player injuries to improve predictions:

```bash
# Add an injured player
python scripts/manage_injuries.py add --team GSW --player-id 201939 --player-name "Stephen Curry" --injury "Right ankle sprain" --return-date 2024-03-20

# List injuries for a team
python scripts/manage_injuries.py list --team GSW

# Remove a player from the injury list
python scripts/manage_injuries.py remove --team GSW --player-id 201939

# Export injury report to CSV
python scripts/manage_injuries.py export --output data/processed/injury_report.csv
```

## Testing

The project includes a comprehensive test suite to ensure all components work as expected. To run the tests:

```bash
python tests/run_tests.py
```

This will run all unit tests and report any failures.

## Key Features

1. **Minutes Prediction**: Uses rolling window averages of recent games to predict player minutes.
2. **Depth Chart Estimation**: Constructs team depth charts by position based on minutes played.
3. **Lineup Analysis**: Analyzes which lineups play together and their effectiveness.
4. **Rotation Patterns**: Predicts how minutes are distributed throughout quarters.
5. **Injury Adjustments**: Adjusts predictions based on player injuries.

## Data Sources

This project uses the official NBA API (via the `nba_api` package) to collect:
- Player game logs (minutes, stats, etc.)
- Player information (positions, teams)
- Lineup data (combinations, minutes, ratings)

## Model Approach

The prediction system uses a rule-based approach with the following factors:
- Recent minutes played (rolling 5 and 10 game averages)
- Depth chart position (starter vs bench)
- Injury status
- Back-to-back game effects
- Team-specific rotation patterns

## Future Improvements

- Add machine learning models for more accurate predictions
- Incorporate opponent matchup data
- Add play-by-play analysis for more detailed rotation patterns
- Create a web dashboard for visualizing predictions
- Add backtesting functionality to evaluate prediction accuracy 