import pandas as pd
import json
import os
from datetime import datetime, timedelta

def add_injured_player(injury_file, team_id, player_id, player_name, injury_type, expected_return_date=None):
    """
    Add an injured player to the injury list.
    
    Args:
        injury_file (str): Path to injury list JSON file
        team_id (int): Team ID
        player_id (int): Player ID
        player_name (str): Player name
        injury_type (str): Type of injury
        expected_return_date (str, optional): Expected return date in format YYYY-MM-DD
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Load existing injury data
    injury_data = {}
    if os.path.exists(injury_file):
        with open(injury_file, 'r') as f:
            try:
                injury_data = json.load(f)
            except json.JSONDecodeError:
                injury_data = {}
    
    # Convert team_id to string (for JSON)
    team_id_str = str(team_id)
    
    # Initialize team entry if not exists
    if team_id_str not in injury_data:
        injury_data[team_id_str] = {
            'player_ids': [],
            'players': []
        }
    
    # Check if player is already in the injury list
    if player_id in injury_data[team_id_str]['player_ids']:
        # Update existing player
        for player in injury_data[team_id_str]['players']:
            if player['player_id'] == player_id:
                player['injury_type'] = injury_type
                player['expected_return_date'] = expected_return_date
                player['last_updated'] = datetime.now().strftime('%Y-%m-%d')
                break
    else:
        # Add new player
        injury_data[team_id_str]['player_ids'].append(player_id)
        injury_data[team_id_str]['players'].append({
            'player_id': player_id,
            'player_name': player_name,
            'injury_type': injury_type,
            'expected_return_date': expected_return_date,
            'date_added': datetime.now().strftime('%Y-%m-%d'),
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        })
    
    # Save updated injury data
    try:
        with open(injury_file, 'w') as f:
            json.dump(injury_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving injury data: {e}")
        return False

def remove_injured_player(injury_file, team_id, player_id):
    """
    Remove a player from the injury list.
    
    Args:
        injury_file (str): Path to injury list JSON file
        team_id (int): Team ID
        player_id (int): Player ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if injury file exists
    if not os.path.exists(injury_file):
        print(f"Error: Injury file {injury_file} not found")
        return False
    
    # Load existing injury data
    with open(injury_file, 'r') as f:
        try:
            injury_data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in injury file")
            return False
    
    # Convert team_id to string (for JSON)
    team_id_str = str(team_id)
    
    # Check if team exists in injury data
    if team_id_str not in injury_data:
        print(f"Error: Team ID {team_id} not found in injury data")
        return False
    
    # Check if player is in the injury list
    if player_id not in injury_data[team_id_str]['player_ids']:
        print(f"Error: Player ID {player_id} not found in team's injury list")
        return False
    
    # Remove player from list
    injury_data[team_id_str]['player_ids'].remove(player_id)
    injury_data[team_id_str]['players'] = [
        player for player in injury_data[team_id_str]['players'] 
        if player['player_id'] != player_id
    ]
    
    # Remove team entry if no more injured players
    if not injury_data[team_id_str]['player_ids']:
        del injury_data[team_id_str]
    
    # Save updated injury data
    try:
        with open(injury_file, 'w') as f:
            json.dump(injury_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving injury data: {e}")
        return False

def get_team_injuries(injury_file, team_id):
    """
    Get a list of injured players for a team.
    
    Args:
        injury_file (str): Path to injury list JSON file
        team_id (int): Team ID
        
    Returns:
        list: List of injured player dictionaries
    """
    # Check if injury file exists
    if not os.path.exists(injury_file):
        return []
    
    # Load injury data
    with open(injury_file, 'r') as f:
        try:
            injury_data = json.load(f)
        except json.JSONDecodeError:
            return []
    
    # Convert team_id to string (for JSON)
    team_id_str = str(team_id)
    
    # Return injured players for the team
    if team_id_str in injury_data:
        return injury_data[team_id_str]['players']
    else:
        return []

def get_all_injured_player_ids(injury_file):
    """
    Get a list of all injured player IDs.
    
    Args:
        injury_file (str): Path to injury list JSON file
        
    Returns:
        dict: Dictionary mapping team IDs to lists of injured player IDs
    """
    # Check if injury file exists
    if not os.path.exists(injury_file):
        return {}
    
    # Load injury data
    with open(injury_file, 'r') as f:
        try:
            injury_data = json.load(f)
        except json.JSONDecodeError:
            return {}
    
    # Create a dictionary of team_id -> player_ids
    injured_player_ids = {}
    for team_id, team_data in injury_data.items():
        injured_player_ids[team_id] = team_data['player_ids']
    
    return injured_player_ids

def export_injury_report(injury_file, output_file):
    """
    Export a formatted injury report to CSV.
    
    Args:
        injury_file (str): Path to injury list JSON file
        output_file (str): Output CSV file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if injury file exists
    if not os.path.exists(injury_file):
        print(f"Error: Injury file {injury_file} not found")
        return False
    
    # Load injury data
    with open(injury_file, 'r') as f:
        try:
            injury_data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in injury file")
            return False
    
    # Prepare data for report
    report_data = []
    
    for team_id, team_data in injury_data.items():
        for player in team_data['players']:
            report_data.append({
                'TEAM_ID': team_id,
                'PLAYER_ID': player['player_id'],
                'PLAYER_NAME': player['player_name'],
                'INJURY_TYPE': player['injury_type'],
                'EXPECTED_RETURN_DATE': player['expected_return_date'],
                'DATE_ADDED': player['date_added'],
                'LAST_UPDATED': player['last_updated']
            })
    
    # Create and save report
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_file, index=False)
        return True
    else:
        print("No injury data available for report")
        return False 