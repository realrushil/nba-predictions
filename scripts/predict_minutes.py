import os
import pandas as pd
import argparse
from datetime import datetime, timedelta
import sys
import json

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_collection import get_player_game_logs, get_player_info, get_team_info
from modules.minutes_processor import calculate_rolling_minutes
from modules.minutes_predictor import predict_team_minutes, predict_starting_lineup, predict_rotation_pattern

def load_injury_list(injury_file):
    """
    Load injury list from a JSON file.
    
    Args:
        injury_file (str): Path to injury list JSON file
        
    Returns:
        dict: Dictionary mapping team IDs to lists of injured player IDs
    """
    if not os.path.exists(injury_file):
        return {}
    
    with open(injury_file, 'r') as f:
        return json.load(f)

def main():
    """
    Predict player minutes and lineups for upcoming games.
    """
    parser = argparse.ArgumentParser(description='Predict NBA player minutes and lineups')
    parser.add_argument('--season', type=str, default='2023-24', help='NBA season in format YYYY-YY')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical game data to use')
    parser.add_argument('--team', type=str, required=True, help='Team abbreviation (e.g., LAL, GSW)')
    parser.add_argument('--opponent', type=str, help='Opponent team abbreviation')
    parser.add_argument('--date', type=str, help='Game date in format YYYY-MM-DD (default: tomorrow)')
    parser.add_argument('--depth-chart', type=str, default='data/processed/depth_charts.csv', 
                        help='Path to depth chart CSV file')
    parser.add_argument('--b2b', action='store_true', help='Flag for back-to-back game')
    parser.add_argument('--injuries', type=str, help='Path to injury list JSON file')
    parser.add_argument('--output-dir', type=str, default='data/processed/predictions', 
                        help='Output directory for predictions')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set game date (default: tomorrow)
    if args.date:
        game_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        game_date = datetime.now() + timedelta(days=1)
    
    game_date_str = game_date.strftime('%Y-%m-%d')
    
    print(f"Predicting minutes for {args.team} on {game_date_str}")
    
    # Load depth chart
    if not os.path.exists(args.depth_chart):
        print(f"Error: Depth chart file not found at {args.depth_chart}")
        print("Run generate_depth_charts.py first to create depth charts")
        return
    
    depth_chart_df = pd.read_csv(args.depth_chart)
    
    # Get team info
    team_info_df = pd.DataFrame(get_team_info())
    
    # Find team ID from abbreviation
    team_id = None
    for _, team in team_info_df.iterrows():
        if team['abbreviation'] == args.team:
            team_id = team['id']
            break
    
    if team_id is None:
        print(f"Error: Team '{args.team}' not found. Check the team abbreviation.")
        return
    
    # Find opponent ID if provided
    opponent_id = None
    if args.opponent:
        for _, team in team_info_df.iterrows():
            if team['abbreviation'] == args.opponent:
                opponent_id = team['id']
                break
        
        if opponent_id is None:
            print(f"Warning: Opponent '{args.opponent}' not found. Continuing without opponent info.")
    
    # Calculate date range for historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    date_from = start_date.strftime('%m/%d/%Y')
    date_to = end_date.strftime('%m/%d/%Y')
    
    # Fetch player game logs
    print("Fetching recent player game logs...")
    game_logs = get_player_game_logs(
        season=args.season,
        season_type="Regular Season",
        date_from=date_from,
        date_to=date_to
    )
    
    if game_logs.empty:
        print("Error: No game logs retrieved. Check your parameters and API access.")
        return
    
    # Calculate rolling minutes
    print("Calculating rolling minutes...")
    game_logs_with_rolling = calculate_rolling_minutes(
        game_logs, 
        window_sizes=[5, 10, 20]
    )
    
    # Load injury list if provided
    injury_list = []
    if args.injuries and os.path.exists(args.injuries):
        print(f"Loading injury list from {args.injuries}...")
        injury_data = load_injury_list(args.injuries)
        
        # Get injury list for the specified team
        if str(team_id) in injury_data:
            injury_list = injury_data[str(team_id)]
            print(f"Found {len(injury_list)} injured players for {args.team}")
    
    # Predict team minutes
    print(f"Predicting minutes for {args.team} (Team ID: {team_id})...")
    team_minutes = predict_team_minutes(
        game_logs_with_rolling,
        depth_chart_df,
        team_id,
        game_date,
        opponent_id,
        args.b2b,
        injury_list
    )
    
    if team_minutes.empty:
        print(f"Error: Could not predict minutes for team {args.team}. Check if depth chart exists.")
        return
    
    # Predict starting lineup
    print("Predicting starting lineup...")
    starting_lineup = predict_starting_lineup(team_minutes)
    
    # Predict rotation pattern
    print("Predicting rotation pattern...")
    rotation_pattern = predict_rotation_pattern(team_minutes)
    
    # Save predictions
    output_base = os.path.join(args.output_dir, f"{args.team}_{game_date_str}")
    
    team_minutes.to_csv(f"{output_base}_minutes.csv", index=False)
    starting_lineup.to_csv(f"{output_base}_lineup.csv", index=False)
    rotation_pattern.to_csv(f"{output_base}_rotation.csv", index=False)
    
    print(f"Predictions saved to {args.output_dir}")
    
    # Print summary
    print("\nMinutes Prediction Summary:")
    print(f"Team: {args.team}, Date: {game_date_str}")
    print("\nPredicted Starting Lineup:")
    for _, player in starting_lineup.iterrows():
        print(f"{player['POSITION']}: {player['PLAYER_NAME']} - {player['PREDICTED_MINUTES']} min")
    
    print("\nTop Minute Getters:")
    for _, player in team_minutes.sort_values('PREDICTED_MINUTES', ascending=False).head(8).iterrows():
        print(f"{player['PLAYER_NAME']} ({player['POSITION']}): {player['PREDICTED_MINUTES']} min")

if __name__ == "__main__":
    main() 