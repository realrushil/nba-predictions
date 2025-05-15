import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_rolling_minutes(game_logs_df, window_sizes=[5, 10, 20]):
    """
    Calculate rolling average minutes for each player over different window sizes.
    
    Args:
        game_logs_df (pandas.DataFrame): Player game logs
        window_sizes (list): List of window sizes to calculate rolling averages
        
    Returns:
        pandas.DataFrame: Game logs with rolling minutes columns added
    """
    # Make a copy to avoid modifying the original
    df = game_logs_df.copy()
    
    # Convert GAME_DATE to datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Sort by player and date
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    
    # Calculate rolling minutes for each window size
    for window in window_sizes:
        col_name = f'MIN_ROLLING_{window}'
        df[col_name] = df.groupby('PLAYER_ID')['MIN'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    return df

def estimate_depth_chart(game_logs_df, player_info_df, window=5, date=None):
    """
    Estimate team depth charts based on recent minutes played and position data.
    
    Args:
        game_logs_df (pandas.DataFrame): Player game logs with rolling minutes
        player_info_df (pandas.DataFrame): Player information including positions
        window (int): Window size for rolling minutes to use
        date (str or datetime): Date to estimate depth chart for (default: most recent date in data)
        
    Returns:
        pandas.DataFrame: Depth chart by team and position
    """
    # Make a copy to avoid modifying the original
    df = game_logs_df.copy()
    
    # Convert GAME_DATE to datetime if it's not already
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # If no date is provided, use the most recent date in the data
    if date is None:
        date = df['GAME_DATE'].max()
    else:
        date = pd.to_datetime(date)
    
    # Filter to include only games before or on the specified date
    df = df[df['GAME_DATE'] <= date]
    
    # Get the most recent game for each player
    latest_games = df.sort_values('GAME_DATE').groupby('PLAYER_ID').last().reset_index()
    
    # Merge with player info to get position data
    merged_df = pd.merge(
        latest_games,
        player_info_df[['PERSON_ID', 'POSITION', 'TEAM_ID', 'TEAM_ABBREVIATION']],
        left_on='PLAYER_ID',
        right_on='PERSON_ID',
        how='left'
    )
    
    # Map NBA positions to standard positions
    position_mapping = {
        'Guard': 'G',
        'Guard-Forward': 'G-F',
        'Forward': 'F',
        'Forward-Guard': 'F-G',
        'Forward-Center': 'F-C',
        'Center': 'C',
        'Center-Forward': 'C-F'
    }
    
    # Apply position mapping
    merged_df['POSITION_SIMPLE'] = merged_df['POSITION'].map(position_mapping)
    
    # Define standard positions
    standard_positions = ['PG', 'SG', 'SF', 'PF', 'C']
    
    # Assign players to standard positions based on their position and minutes played
    depth_charts = {}
    
    for team_id in merged_df['TEAM_ID'].unique():
        team_df = merged_df[merged_df['TEAM_ID'] == team_id]
        
        # Initialize depth chart for this team
        team_chart = {pos: [] for pos in standard_positions}
        
        # Assign guards (PG, SG)
        guards = team_df[team_df['POSITION_SIMPLE'].isin(['G', 'G-F'])].sort_values(
            f'MIN_ROLLING_{window}', ascending=False
        )
        
        pg_assigned = 0
        sg_assigned = 0
        
        for _, player in guards.iterrows():
            if pg_assigned < 3:
                team_chart['PG'].append({
                    'PLAYER_ID': player['PLAYER_ID'],
                    'PLAYER_NAME': player['PLAYER_NAME'],
                    'MINUTES': player[f'MIN_ROLLING_{window}'],
                    'DEPTH': pg_assigned + 1
                })
                pg_assigned += 1
            elif sg_assigned < 3:
                team_chart['SG'].append({
                    'PLAYER_ID': player['PLAYER_ID'],
                    'PLAYER_NAME': player['PLAYER_NAME'],
                    'MINUTES': player[f'MIN_ROLLING_{window}'],
                    'DEPTH': sg_assigned + 1
                })
                sg_assigned += 1
        
        # Assign forwards (SF, PF)
        forwards = team_df[team_df['POSITION_SIMPLE'].isin(['F', 'F-G', 'F-C'])].sort_values(
            f'MIN_ROLLING_{window}', ascending=False
        )
        
        sf_assigned = 0
        pf_assigned = 0
        
        for _, player in forwards.iterrows():
            if sf_assigned < 3:
                team_chart['SF'].append({
                    'PLAYER_ID': player['PLAYER_ID'],
                    'PLAYER_NAME': player['PLAYER_NAME'],
                    'MINUTES': player[f'MIN_ROLLING_{window}'],
                    'DEPTH': sf_assigned + 1
                })
                sf_assigned += 1
            elif pf_assigned < 3:
                team_chart['PF'].append({
                    'PLAYER_ID': player['PLAYER_ID'],
                    'PLAYER_NAME': player['PLAYER_NAME'],
                    'MINUTES': player[f'MIN_ROLLING_{window}'],
                    'DEPTH': pf_assigned + 1
                })
                pf_assigned += 1
        
        # Assign centers (C)
        centers = team_df[team_df['POSITION_SIMPLE'].isin(['C', 'C-F', 'F-C'])].sort_values(
            f'MIN_ROLLING_{window}', ascending=False
        )
        
        c_assigned = 0
        
        for _, player in centers.iterrows():
            if c_assigned < 3:
                team_chart['C'].append({
                    'PLAYER_ID': player['PLAYER_ID'],
                    'PLAYER_NAME': player['PLAYER_NAME'],
                    'MINUTES': player[f'MIN_ROLLING_{window}'],
                    'DEPTH': c_assigned + 1
                })
                c_assigned += 1
        
        # Store team depth chart
        depth_charts[team_id] = team_chart
    
    # Convert depth charts to DataFrame format
    rows = []
    
    for team_id, positions in depth_charts.items():
        team_abbr = merged_df[merged_df['TEAM_ID'] == team_id]['TEAM_ABBREVIATION'].iloc[0]
        
        row = {
            'TEAM_ID': team_id,
            'TEAM_ABBREVIATION': team_abbr,
            'DATE': date.strftime('%Y-%m-%d')
        }
        
        for pos, players in positions.items():
            for i in range(min(3, len(players))):
                player = players[i]
                row[f'{pos}{i+1}_ID'] = player['PLAYER_ID']
                row[f'{pos}{i+1}_NAME'] = player['PLAYER_NAME']
                row[f'{pos}{i+1}_MIN'] = player['MINUTES']
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def export_depth_chart_csv(depth_chart_df, output_path):
    """
    Export depth chart DataFrame to CSV.
    
    Args:
        depth_chart_df (pandas.DataFrame): Depth chart DataFrame
        output_path (str): Path to save the CSV file
    """
    depth_chart_df.to_csv(output_path, index=False)
    print(f"Depth chart exported to {output_path}")

def analyze_lineup_data(lineup_df, min_minutes_played=50):
    """
    Analyze lineup data to identify frequently used lineups.
    
    Args:
        lineup_df (pandas.DataFrame): Lineup data from NBA API
        min_minutes_played (int): Minimum minutes played to include in analysis
        
    Returns:
        pandas.DataFrame: Analyzed lineup data
    """
    # Filter lineups with minimum minutes played
    filtered_lineups = lineup_df[lineup_df['MIN'] >= min_minutes_played].copy()
    
    # Calculate additional metrics
    filtered_lineups['MIN_PER_GAME'] = filtered_lineups['MIN'] / filtered_lineups['GP']
    
    # Sort by minutes played
    filtered_lineups = filtered_lineups.sort_values('MIN', ascending=False)
    
    return filtered_lineups 