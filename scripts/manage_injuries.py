import os
import argparse
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.injury_tracker import (
    add_injured_player, 
    remove_injured_player, 
    get_team_injuries, 
    export_injury_report
)
from modules.data_collection import get_team_info

def main():
    """
    Command-line tool to manage NBA player injuries.
    """
    parser = argparse.ArgumentParser(description='Manage NBA player injuries')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Injury file path
    default_injury_file = 'data/processed/injuries.json'
    
    # Add player command
    add_parser = subparsers.add_parser('add', help='Add an injured player')
    add_parser.add_argument('--team', type=str, required=True, help='Team abbreviation (e.g., LAL, GSW)')
    add_parser.add_argument('--player-id', type=int, required=True, help='Player ID')
    add_parser.add_argument('--player-name', type=str, required=True, help='Player name')
    add_parser.add_argument('--injury', type=str, required=True, help='Type of injury')
    add_parser.add_argument('--return-date', type=str, help='Expected return date (YYYY-MM-DD)')
    add_parser.add_argument('--injury-file', type=str, default=default_injury_file, help='Injury file path')
    
    # Remove player command
    remove_parser = subparsers.add_parser('remove', help='Remove a player from injury list')
    remove_parser.add_argument('--team', type=str, required=True, help='Team abbreviation (e.g., LAL, GSW)')
    remove_parser.add_argument('--player-id', type=int, required=True, help='Player ID')
    remove_parser.add_argument('--injury-file', type=str, default=default_injury_file, help='Injury file path')
    
    # List injuries command
    list_parser = subparsers.add_parser('list', help='List injured players for a team')
    list_parser.add_argument('--team', type=str, required=True, help='Team abbreviation (e.g., LAL, GSW)')
    list_parser.add_argument('--injury-file', type=str, default=default_injury_file, help='Injury file path')
    
    # Export injury report command
    export_parser = subparsers.add_parser('export', help='Export injury report to CSV')
    export_parser.add_argument('--output', type=str, default='data/processed/injury_report.csv', help='Output CSV file path')
    export_parser.add_argument('--injury-file', type=str, default=default_injury_file, help='Injury file path')
    
    args = parser.parse_args()
    
    # Create injury file directory if it doesn't exist
    if args.command:
        injury_file = getattr(args, 'injury_file')
        os.makedirs(os.path.dirname(injury_file), exist_ok=True)
    
    # Process commands
    if args.command == 'add':
        # Get team ID from abbreviation
        team_id = get_team_id_from_abbr(args.team)
        if team_id is None:
            print(f"Error: Team '{args.team}' not found")
            return
        
        # Add injured player
        success = add_injured_player(
            args.injury_file,
            team_id,
            args.player_id,
            args.player_name,
            args.injury,
            args.return_date
        )
        
        if success:
            print(f"Successfully added {args.player_name} to injury list for {args.team}")
        else:
            print(f"Failed to add {args.player_name} to injury list")
    
    elif args.command == 'remove':
        # Get team ID from abbreviation
        team_id = get_team_id_from_abbr(args.team)
        if team_id is None:
            print(f"Error: Team '{args.team}' not found")
            return
        
        # Remove injured player
        success = remove_injured_player(
            args.injury_file,
            team_id,
            args.player_id
        )
        
        if success:
            print(f"Successfully removed player ID {args.player_id} from injury list for {args.team}")
        else:
            print(f"Failed to remove player from injury list")
    
    elif args.command == 'list':
        # Get team ID from abbreviation
        team_id = get_team_id_from_abbr(args.team)
        if team_id is None:
            print(f"Error: Team '{args.team}' not found")
            return
        
        # Get team injuries
        injuries = get_team_injuries(args.injury_file, team_id)
        
        if not injuries:
            print(f"No injuries reported for {args.team}")
        else:
            print(f"Injuries for {args.team}:")
            for player in injuries:
                return_date = player['expected_return_date'] or 'Unknown'
                print(f"- {player['player_name']}: {player['injury_type']} (Expected return: {return_date})")
    
    elif args.command == 'export':
        # Export injury report
        success = export_injury_report(args.injury_file, args.output)
        
        if success:
            print(f"Injury report exported to {args.output}")
        else:
            print("Failed to export injury report")
    
    else:
        parser.print_help()

def get_team_id_from_abbr(team_abbr):
    """
    Get team ID from team abbreviation.
    
    Args:
        team_abbr (str): Team abbreviation
        
    Returns:
        int or None: Team ID if found, None otherwise
    """
    team_info = get_team_info()
    
    for team in team_info:
        if team['abbreviation'] == team_abbr:
            return team['id']
    
    return None

if __name__ == "__main__":
    main() 