import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_collection import (
    get_player_game_logs,
    get_lineup_data,
    get_player_info,
    get_team_info
)

class TestDataCollection(unittest.TestCase):
    """Test cases for the data_collection module."""
    
    @patch('modules.data_collection.playergamelogs.PlayerGameLogs')
    @patch('modules.data_collection.time.sleep')
    def test_get_player_game_logs(self, mock_sleep, mock_player_game_logs):
        """Test the get_player_game_logs function."""
        # Mock the API response
        mock_df = pd.DataFrame({
            'PLAYER_ID': [1, 2, 3],
            'PLAYER_NAME': ['Player 1', 'Player 2', 'Player 3'],
            'GAME_DATE': ['2023-12-01', '2023-12-01', '2023-12-01'],
            'MIN': [30, 25, 20]
        })
        mock_player_game_logs.return_value.get_data_frames.return_value = [mock_df]
        
        # Call the function
        result = get_player_game_logs(
            season='2023-24',
            season_type='Regular Season',
            date_from='12/01/2023',
            date_to='12/10/2023'
        )
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result['PLAYER_NAME']), ['Player 1', 'Player 2', 'Player 3'])
        
        # Verify the API was called with correct parameters
        mock_player_game_logs.assert_called_once_with(
            season_nullable='2023-24',
            season_type_nullable='Regular Season',
            date_from_nullable='12/01/2023',
            date_to_nullable='12/10/2023'
        )
        
        # Verify sleep was called to avoid API rate limiting
        mock_sleep.assert_called_once_with(1)
    
    @patch('modules.data_collection.leaguedashlineups.LeagueDashLineups')
    @patch('modules.data_collection.time.sleep')
    def test_get_lineup_data(self, mock_sleep, mock_league_dash_lineups):
        """Test the get_lineup_data function."""
        # Mock the API response
        mock_df = pd.DataFrame({
            'GROUP_ID': ['1610612737-1', '1610612738-1'],
            'GROUP_NAME': ['Team 1 Lineup', 'Team 2 Lineup'],
            'MIN': [100, 95]
        })
        mock_league_dash_lineups.return_value.get_data_frames.return_value = [mock_df]
        
        # Call the function
        result = get_lineup_data(
            season='2023-24',
            season_type='Regular Season',
            measure_type='Base'
        )
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result['GROUP_NAME']), ['Team 1 Lineup', 'Team 2 Lineup'])
        
        # Verify the API was called with correct parameters
        mock_league_dash_lineups.assert_called_once_with(
            season='2023-24',
            season_type_all_star='Regular Season',
            measure_type_detailed_defense='Base',
            group_quantity=5
        )
        
        # Verify sleep was called to avoid API rate limiting
        mock_sleep.assert_called_once_with(1)
    
    @patch('modules.data_collection.commonplayerinfo.CommonPlayerInfo')
    @patch('modules.data_collection.players.get_active_players')
    @patch('modules.data_collection.time.sleep')
    def test_get_player_info(self, mock_sleep, mock_get_active_players, mock_common_player_info):
        """Test the get_player_info function."""
        # Mock the API response
        mock_df = pd.DataFrame({
            'PERSON_ID': [1, 2],
            'DISPLAY_FIRST_LAST': ['Player 1', 'Player 2'],
            'POSITION': ['Guard', 'Forward']
        })
        mock_common_player_info.return_value.get_data_frames.return_value = [mock_df]
        
        # Call the function with specific player IDs
        result = get_player_info(player_ids=[1, 2])
        
        # Verify that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Verify that the API was called for each player ID
        mock_common_player_info.assert_any_call(player_id=1)
        mock_common_player_info.assert_any_call(player_id=2)
        
        # Verify sleep was called to avoid API rate limiting
        self.assertEqual(mock_sleep.call_count, 2)
    
    @patch('modules.data_collection.teams.get_teams')
    def test_get_team_info(self, mock_get_teams):
        """Test the get_team_info function."""
        # Mock the API response
        mock_teams = [
            {'id': 1, 'full_name': 'Team 1', 'abbreviation': 'T1'},
            {'id': 2, 'full_name': 'Team 2', 'abbreviation': 'T2'}
        ]
        mock_get_teams.return_value = mock_teams
        
        # Call the function
        result = get_team_info()
        
        # Verify the result
        self.assertEqual(len(result), 2)
        
        # Check if result is a DataFrame or list
        if isinstance(result, pd.DataFrame):
            self.assertEqual(result.iloc[0]['full_name'], 'Team 1')
            self.assertEqual(result.iloc[1]['abbreviation'], 'T2')
        else:
            self.assertEqual(result[0]['full_name'], 'Team 1')
            self.assertEqual(result[1]['abbreviation'], 'T2')
        
        # Verify the API was called
        mock_get_teams.assert_called_once()

if __name__ == '__main__':
    unittest.main() 