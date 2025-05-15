import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def predict_player_minutes(game_logs_df, depth_chart_df, player_id, game_date, 
                           opponent_team_id=None, is_back_to_back=False, is_injured=False):
    """
    Predict minutes for a player in an upcoming game.
    
    Args:
        game_logs_df (pandas.DataFrame): Player game logs with rolling minutes
        depth_chart_df (pandas.DataFrame): Team depth chart
        player_id (int): Player ID
        game_date (str or datetime): Date of the game
        opponent_team_id (int, optional): Opponent team ID
        is_back_to_back (bool): Whether the game is part of a back-to-back
        is_injured (bool): Whether the player is injured
        
    Returns:
        float: Predicted minutes for the player
    """
    if is_injured:
        return 0.0
    
    # Convert game_date to datetime if it's not already
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date)
    
    # Filter game logs for the specified player
    player_logs = game_logs_df[game_logs_df['PLAYER_ID'] == player_id].copy()
    
    if player_logs.empty:
        return 0.0
    
    # Sort by date (most recent first)
    player_logs = player_logs.sort_values('GAME_DATE', ascending=False)
    
    # Get player's team ID
    team_id = player_logs.iloc[0]['TEAM_ID']
    
    # Get the most recent depth chart for the player's team
    team_depth_chart = depth_chart_df[depth_chart_df['TEAM_ID'] == team_id].sort_values('DATE', ascending=False).iloc[0]
    
    # Determine player's position and depth in the rotation
    position = None
    depth = None
    
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        for i in range(1, 4):  # Check depth 1-3
            player_id_col = f'{pos}{i}_ID'
            if player_id_col in team_depth_chart and team_depth_chart[player_id_col] == player_id:
                position = pos
                depth = i
                break
        if position is not None:
            break
    
    # Get recent minutes averages
    recent_5_avg = player_logs.iloc[0]['MIN_ROLLING_5'] if 'MIN_ROLLING_5' in player_logs.columns else player_logs.head(5)['MIN'].mean()
    recent_10_avg = player_logs.iloc[0]['MIN_ROLLING_10'] if 'MIN_ROLLING_10' in player_logs.columns else player_logs.head(10)['MIN'].mean()
    
    # Base prediction on recent average minutes
    predicted_minutes = recent_5_avg * 0.7 + recent_10_avg * 0.3
    
    # Adjust for depth in rotation
    if depth is not None:
        if depth == 1:  # Starter
            depth_factor = 1.0
        elif depth == 2:  # Key rotation player
            depth_factor = 0.8
        else:  # 3rd string
            depth_factor = 0.5
        
        predicted_minutes *= depth_factor
    
    # Adjust for back-to-back games
    if is_back_to_back:
        # Check if player is a starter or key player
        if depth == 1 or (depth == 2 and recent_5_avg >= 20):
            predicted_minutes *= 0.9  # Reduce minutes for starters/key players on back-to-backs
    
    # Ensure minutes are within reasonable bounds
    predicted_minutes = max(0, min(48, predicted_minutes))
    
    return round(predicted_minutes, 1)

def predict_team_minutes(game_logs_df, depth_chart_df, team_id, game_date, 
                         opponent_team_id=None, is_back_to_back=False, injury_list=None):
    """
    Predict minutes for all players on a team for an upcoming game.
    
    Args:
        game_logs_df (pandas.DataFrame): Player game logs with rolling minutes
        depth_chart_df (pandas.DataFrame): Team depth chart
        team_id (int): Team ID
        game_date (str or datetime): Date of the game
        opponent_team_id (int, optional): Opponent team ID
        is_back_to_back (bool): Whether the game is part of a back-to-back
        injury_list (list): List of player IDs who are injured
        
    Returns:
        pandas.DataFrame: DataFrame with predicted minutes for each player
    """
    if injury_list is None:
        injury_list = []
    
    # Get the most recent depth chart for the team
    team_depth_chart = depth_chart_df[depth_chart_df['TEAM_ID'] == team_id].sort_values('DATE', ascending=False).iloc[0]
    
    # Extract player IDs from the depth chart
    player_ids = []
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        for i in range(1, 4):  # Check depth 1-3
            player_id_col = f'{pos}{i}_ID'
            if player_id_col in team_depth_chart and not pd.isna(team_depth_chart[player_id_col]):
                player_ids.append(int(team_depth_chart[player_id_col]))
    
    # Predict minutes for each player
    player_minutes = []
    
    for player_id in player_ids:
        is_injured = player_id in injury_list
        
        predicted_min = predict_player_minutes(
            game_logs_df, 
            depth_chart_df, 
            player_id, 
            game_date, 
            opponent_team_id, 
            is_back_to_back, 
            is_injured
        )
        
        # Get player name
        player_name = None
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            for i in range(1, 4):
                id_col = f'{pos}{i}_ID'
                name_col = f'{pos}{i}_NAME'
                
                if id_col in team_depth_chart and team_depth_chart[id_col] == player_id:
                    player_name = team_depth_chart[name_col]
                    position = pos
                    depth = i
                    break
            
            if player_name is not None:
                break
        
        player_minutes.append({
            'TEAM_ID': team_id,
            'TEAM_ABBREVIATION': team_depth_chart['TEAM_ABBREVIATION'],
            'PLAYER_ID': player_id,
            'PLAYER_NAME': player_name,
            'POSITION': position,
            'DEPTH': depth,
            'PREDICTED_MINUTES': predicted_min,
            'GAME_DATE': game_date if isinstance(game_date, str) else game_date.strftime('%Y-%m-%d'),
            'IS_INJURED': is_injured,
            'IS_BACK_TO_BACK': is_back_to_back
        })
    
    # Convert to DataFrame
    player_minutes_df = pd.DataFrame(player_minutes)
    
    # Sort by predicted minutes (descending)
    player_minutes_df = player_minutes_df.sort_values('PREDICTED_MINUTES', ascending=False)
    
    return player_minutes_df

def predict_starting_lineup(team_minutes_df):
    """
    Predict the starting lineup based on predicted minutes.
    
    Args:
        team_minutes_df (pandas.DataFrame): Team minutes predictions
        
    Returns:
        pandas.DataFrame: Starting lineup (5 players)
    """
    # Get non-injured players
    available_players = team_minutes_df[~team_minutes_df['IS_INJURED']].copy()
    
    # Initialize lineup
    lineup = []
    
    # Ensure we have at least one player for each position
    for position in ['PG', 'SG', 'SF', 'PF', 'C']:
        position_players = available_players[available_players['POSITION'] == position]
        
        if not position_players.empty:
            # Get the player with most predicted minutes for this position
            top_player = position_players.sort_values('PREDICTED_MINUTES', ascending=False).iloc[0]
            lineup.append(top_player)
            
            # Remove this player from available players
            available_players = available_players[available_players['PLAYER_ID'] != top_player['PLAYER_ID']]
    
    # If we don't have 5 players yet, add the remaining players with the most minutes
    while len(lineup) < 5 and not available_players.empty:
        top_player = available_players.sort_values('PREDICTED_MINUTES', ascending=False).iloc[0]
        lineup.append(top_player)
        
        # Remove this player from available players
        available_players = available_players[available_players['PLAYER_ID'] != top_player['PLAYER_ID']]
    
    # Convert to DataFrame
    lineup_df = pd.DataFrame(lineup)
    
    # Sort by position
    position_order = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
    lineup_df['POS_ORDER'] = lineup_df['POSITION'].map(position_order)
    lineup_df = lineup_df.sort_values('POS_ORDER')
    lineup_df = lineup_df.drop('POS_ORDER', axis=1)
    
    return lineup_df

def predict_rotation_pattern(team_minutes_df, quarter_lengths=12, num_quarters=4):
    """
    Predict rotation pattern for players throughout the game.
    
    Args:
        team_minutes_df (pandas.DataFrame): Team minutes predictions
        quarter_lengths (int): Length of each quarter in minutes
        num_quarters (int): Number of quarters in the game
        
    Returns:
        pandas.DataFrame: DataFrame with predicted minutes by quarter for each player
    """
    # Calculate total game minutes
    total_game_minutes = quarter_lengths * num_quarters
    
    # Get non-injured players sorted by predicted minutes
    available_players = team_minutes_df[~team_minutes_df['IS_INJURED']].sort_values('PREDICTED_MINUTES', ascending=False).copy()
    
    # Initialize rotation pattern
    rotation = []
    
    for _, player in available_players.iterrows():
        # Skip players with 0 predicted minutes
        if player['PREDICTED_MINUTES'] <= 0:
            continue
        
        # Calculate what percentage of each quarter the player will play
        minutes_per_quarter = player['PREDICTED_MINUTES'] / num_quarters
        quarter_percentage = minutes_per_quarter / quarter_lengths
        
        # Initialize minutes distribution for this player
        player_rotation = {
            'PLAYER_ID': player['PLAYER_ID'],
            'PLAYER_NAME': player['PLAYER_NAME'],
            'POSITION': player['POSITION'],
            'TOTAL_MINUTES': player['PREDICTED_MINUTES']
        }
        
        # Distribute minutes differently based on role
        if player['DEPTH'] == 1:  # Starters
            # Starters play beginning and end of quarters
            for q in range(1, num_quarters + 1):
                # Starters play more in 1st and 3rd quarters, and end of game
                if q == 1 or q == 3:
                    q_mins = min(quarter_lengths, minutes_per_quarter * 1.2)
                elif q == num_quarters:
                    q_mins = min(quarter_lengths, minutes_per_quarter * 1.1)
                else:
                    q_mins = min(quarter_lengths, minutes_per_quarter * 0.7)
                
                player_rotation[f'Q{q}'] = round(q_mins, 1)
        else:  # Bench players
            # Bench players play middle of quarters
            for q in range(1, num_quarters + 1):
                # Bench players play more in 2nd and 4th quarters, except end of game
                if q == 2 or q == 4:
                    q_mins = min(quarter_lengths, minutes_per_quarter * 1.3)
                else:
                    q_mins = min(quarter_lengths, minutes_per_quarter * 0.7)
                
                if q == num_quarters:  # Less time in final quarter
                    q_mins *= 0.8
                
                player_rotation[f'Q{q}'] = round(q_mins, 1)
        
        rotation.append(player_rotation)
    
    # Convert to DataFrame
    rotation_df = pd.DataFrame(rotation)
    
    # Add total by quarter columns
    for q in range(1, num_quarters + 1):
        rotation_df[f'Q{q}_TOTAL'] = rotation_df[f'Q{q}'].sum()
    
    return rotation_df 