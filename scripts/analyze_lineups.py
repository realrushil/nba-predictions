import os
import pandas as pd
import argparse
from datetime import datetime
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_collection import get_lineup_data, get_team_info
from modules.minutes_processor import analyze_lineup_data

def main():
    """
    Collect and analyze NBA lineup data.
    """
    parser = argparse.ArgumentParser(description='Analyze NBA lineup data')
    parser.add_argument('--season', type=str, default='2023-24', help='NBA season in format YYYY-YY')
    parser.add_argument('--team', type=str, help='Team abbreviation to filter (e.g., LAL, GSW)')
    parser.add_argument('--min-minutes', type=int, default=50, 
                        help='Minimum minutes played to include in analysis')
    parser.add_argument('--output-dir', type=str, default='data/processed/lineups', 
                        help='Output directory for lineup data')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Fetching lineup data for season: {args.season}")
    
    # Fetch lineup data
    print("Fetching lineup data from NBA API...")
    lineups_df = get_lineup_data(
        season=args.season,
        season_type="Regular Season",
        measure_type="Base"
    )
    
    if lineups_df.empty:
        print("Error: No lineup data retrieved. Check your parameters and API access.")
        return
    
    print(f"Retrieved data for {len(lineups_df)} lineups")
    
    # Get team info if team filter is provided
    team_id = None
    if args.team:
        team_info_df = pd.DataFrame(get_team_info())
        
        # Find team ID from abbreviation
        for _, team in team_info_df.iterrows():
            if team['abbreviation'] == args.team:
                team_id = team['id']
                break
        
        if team_id is None:
            print(f"Warning: Team '{args.team}' not found. Analyzing all teams.")
    
    # Filter by team if specified
    if team_id is not None:
        # Extract team ID from GROUP_ID (format: 1610612XXX)
        lineups_df['TEAM_ID'] = lineups_df['GROUP_ID'].str.extract(r'(1610612\d+)').astype(int)
        
        # Filter lineups for the specified team
        lineups_df = lineups_df[lineups_df['TEAM_ID'] == team_id]
        
        if lineups_df.empty:
            print(f"No lineup data found for team {args.team}.")
            return
        
        print(f"Filtered to {len(lineups_df)} lineups for team {args.team}")
    
    # Analyze lineup data
    print("Analyzing lineup data...")
    analyzed_lineups = analyze_lineup_data(lineups_df, min_minutes_played=args.min_minutes)
    
    if analyzed_lineups.empty:
        print(f"No lineups meet the minimum minutes criteria of {args.min_minutes} minutes.")
        return
    
    print(f"Analysis complete. Found {len(analyzed_lineups)} lineups with at least {args.min_minutes} minutes played.")
    
    # Save the analysis results
    timestamp = datetime.now().strftime("%Y%m%d")
    
    if team_id is not None:
        output_file = os.path.join(args.output_dir, f"{args.team}_lineups_{timestamp}.csv")
    else:
        output_file = os.path.join(args.output_dir, f"all_teams_lineups_{timestamp}.csv")
    
    analyzed_lineups.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary of top lineups
    print("\nTop 5 Lineups by Minutes Played:")
    for i, (_, lineup) in enumerate(analyzed_lineups.head(5).iterrows()):
        print(f"{i+1}. {lineup['GROUP_NAME']} - {lineup['MIN']} minutes, {lineup['MIN_PER_GAME']:.1f} min/game")
        print(f"   Net Rating: {lineup['PLUS_MINUS']:.1f}")
    
    # Save the raw lineup data for reference
    raw_output_file = output_file.replace('.csv', '_raw.csv')
    lineups_df.to_csv(raw_output_file, index=False)
    print(f"Raw lineup data saved to {raw_output_file}")

if __name__ == "__main__":
    main() 