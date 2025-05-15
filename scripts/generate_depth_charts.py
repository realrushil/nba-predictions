import os
import pandas as pd
import argparse
from datetime import datetime, timedelta
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_collection import get_player_game_logs, get_player_info, get_team_info
from modules.minutes_processor import calculate_rolling_minutes, estimate_depth_chart, export_depth_chart_csv

def main():
    """
    Generate team depth charts based on recent player minutes.
    """
    parser = argparse.ArgumentParser(description='Generate NBA team depth charts')
    parser.add_argument('--season', type=str, default='2023-24', help='NBA season in format YYYY-YY')
    parser.add_argument('--days', type=int, default=30, help='Number of days of game data to fetch')
    parser.add_argument('--window', type=int, default=5, help='Window size for rolling minutes calculation')
    parser.add_argument('--output', type=str, default='data/processed/depth_charts.csv', 
                        help='Output file path for the depth charts')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Fetching data for season: {args.season}")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    date_from = start_date.strftime('%m/%d/%Y')
    date_to = end_date.strftime('%m/%d/%Y')
    
    print(f"Date range: {date_from} to {date_to}")
    
    # Fetch player game logs
    print("Fetching player game logs...")
    game_logs = get_player_game_logs(
        season=args.season,
        season_type="Regular Season",
        date_from=date_from,
        date_to=date_to
    )
    
    if game_logs.empty:
        print("Error: No game logs retrieved. Check your parameters and API access.")
        return
    
    print(f"Retrieved {len(game_logs)} game logs")
    
    # Get unique player IDs from game logs
    player_ids = game_logs['PLAYER_ID'].unique().tolist()
    
    # Fetch player information
    print("Fetching player information...")
    player_info = get_player_info(player_ids)
    
    if player_info.empty:
        print("Error: No player information retrieved.")
        return
    
    print(f"Retrieved information for {len(player_info)} players")
    
    # Calculate rolling minutes
    print("Calculating rolling minutes...")
    game_logs_with_rolling = calculate_rolling_minutes(
        game_logs, 
        window_sizes=[args.window, 10, 20]
    )
    
    # Generate depth charts
    print("Generating depth charts...")
    depth_charts = estimate_depth_chart(
        game_logs_with_rolling,
        player_info,
        window=args.window
    )
    
    # Export depth charts to CSV
    print(f"Exporting depth charts to {args.output}...")
    export_depth_chart_csv(depth_charts, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main() 