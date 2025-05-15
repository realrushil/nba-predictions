import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelogs, leaguedashlineups, commonplayerinfo
from nba_api.stats.static import teams, players

def get_player_game_logs(season, season_type="Regular Season", date_from=None, date_to=None):
    """
    Fetch player game logs for a specific season and date range.
    
    Args:
        season (str): Season in format 'YYYY-YY' (e.g., '2023-24')
        season_type (str): 'Regular Season', 'Playoffs', etc.
        date_from (str): Start date in format 'MM/DD/YYYY'
        date_to (str): End date in format 'MM/DD/YYYY'
        
    Returns:
        pandas.DataFrame: Player game logs
    """
    try:
        game_logs = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
            date_from_nullable=date_from,
            date_to_nullable=date_to
        )
        df = game_logs.get_data_frames()[0]
        
        # Add sleep to avoid API rate limiting
        time.sleep(1)
        
        return df
    except Exception as e:
        print(f"Error fetching player game logs: {e}")
        return pd.DataFrame()

def get_lineup_data(season, season_type="Regular Season", measure_type="Base"):
    """
    Fetch lineup data for a specific season.
    
    Args:
        season (str): Season in format 'YYYY-YY' (e.g., '2023-24')
        season_type (str): 'Regular Season', 'Playoffs', etc.
        measure_type (str): 'Base', 'Advanced', etc.
        
    Returns:
        pandas.DataFrame: Lineup data
    """
    try:
        lineups = leaguedashlineups.LeagueDashLineups(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense=measure_type,
            group_quantity=5
        )
        df = lineups.get_data_frames()[0]
        
        # Add sleep to avoid API rate limiting
        time.sleep(1)
        
        return df
    except Exception as e:
        print(f"Error fetching lineup data: {e}")
        return pd.DataFrame()

def get_player_info(player_ids=None):
    """
    Fetch player information including position data.
    
    Args:
        player_ids (list): List of player IDs. If None, all active players will be used.
        
    Returns:
        pandas.DataFrame: Player information
    """
    if player_ids is None:
        # Get all active players
        all_players = players.get_active_players()
        player_ids = [player['id'] for player in all_players]
    
    player_info_list = []
    
    for player_id in player_ids:
        try:
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            df = player_info.get_data_frames()[0]
            player_info_list.append(df)
            
            # Add sleep to avoid API rate limiting
            time.sleep(0.6)
        except Exception as e:
            print(f"Error fetching info for player ID {player_id}: {e}")
    
    if player_info_list:
        return pd.concat(player_info_list, ignore_index=True)
    else:
        return pd.DataFrame()

def get_team_info():
    """
    Get information about all NBA teams.
    
    Returns:
        pandas.DataFrame: Team information
    """
    team_info = teams.get_teams()
    return pd.DataFrame(team_info) 