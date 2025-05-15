import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.minutes_predictor import (
    predict_player_minutes,
    predict_team_minutes,
    predict_starting_lineup,
    predict_rotation_pattern
)

class TestMinutesPredictor(unittest.TestCase):
    """Test cases for the minutes_predictor module."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample game logs data with rolling minutes
        self.game_logs_df = pd.DataFrame({
            'PLAYER_ID': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            'PLAYER_NAME': ['Player 1', 'Player 1', 'Player 1', 'Player 2', 'Player 2', 'Player 2', 
                           'Player 3', 'Player 3', 'Player 3', 'Player 4', 'Player 4', 'Player 4',
                           'Player 5', 'Player 5', 'Player 5'],
            'TEAM_ID': [101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101],
            'TEAM_ABBREVIATION': ['TM1', 'TM1', 'TM1', 'TM1', 'TM1', 'TM1', 'TM1', 'TM1', 'TM1', 'TM1', 'TM1', 'TM1', 'TM1', 'TM1', 'TM1'],
            'GAME_DATE': ['2023-12-01', '2023-12-03', '2023-12-05', '2023-12-01', '2023-12-03', '2023-12-05',
                         '2023-12-01', '2023-12-03', '2023-12-05', '2023-12-01', '2023-12-03', '2023-12-05',
                         '2023-12-01', '2023-12-03', '2023-12-05'],
            'MIN': [30, 32, 28, 25, 24, 26, 20, 22, 18, 15, 14, 16, 10, 12, 8],
            'MIN_ROLLING_5': [30, 31, 30, 25, 24.5, 25, 20, 21, 20, 15, 14.5, 15, 10, 11, 10],
            'MIN_ROLLING_10': [30, 31, 30, 25, 24.5, 25, 20, 21, 20, 15, 14.5, 15, 10, 11, 10]
        })
        
        # Create sample depth chart data
        self.depth_chart_df = pd.DataFrame({
            'TEAM_ID': [101],
            'TEAM_ABBREVIATION': ['TM1'],
            'DATE': ['2023-12-05'],
            'PG1_ID': [1],
            'PG1_NAME': ['Player 1'],
            'PG1_MIN': [30],
            'SG1_ID': [2],
            'SG1_NAME': ['Player 2'],
            'SG1_MIN': [25],
            'SF1_ID': [3],
            'SF1_NAME': ['Player 3'],
            'SF1_MIN': [20],
            'PF1_ID': [4],
            'PF1_NAME': ['Player 4'],
            'PF1_MIN': [15],
            'C1_ID': [5],
            'C1_NAME': ['Player 5'],
            'C1_MIN': [10]
        })
    
    def test_predict_player_minutes(self):
        """Test the predict_player_minutes function."""
        # Call the function for a healthy player
        result = predict_player_minutes(
            self.game_logs_df,
            self.depth_chart_df,
            player_id=1,
            game_date='2023-12-10',
            is_back_to_back=False,
            is_injured=False
        )
        
        # Verify the result
        self.assertIsInstance(result, float)
        
        # The prediction should be based on rolling averages
        # For player 1: 0.7 * MIN_ROLLING_5 + 0.3 * MIN_ROLLING_10 = 0.7 * 30 + 0.3 * 30 = 30
        self.assertAlmostEqual(result, 30.0)
        
        # Test with an injured player
        result = predict_player_minutes(
            self.game_logs_df,
            self.depth_chart_df,
            player_id=1,
            game_date='2023-12-10',
            is_back_to_back=False,
            is_injured=True
        )
        
        # Injured players should get 0 minutes
        self.assertEqual(result, 0.0)
        
        # Test with a back-to-back game for a starter
        result = predict_player_minutes(
            self.game_logs_df,
            self.depth_chart_df,
            player_id=1,
            game_date='2023-12-10',
            is_back_to_back=True,
            is_injured=False
        )
        
        # Minutes should be reduced for starters in back-to-back games
        # 30 * 0.9 = 27
        self.assertAlmostEqual(result, 27.0)
    
    def test_predict_team_minutes(self):
        """Test the predict_team_minutes function."""
        # Call the function
        result = predict_team_minutes(
            self.game_logs_df,
            self.depth_chart_df,
            team_id=101,
            game_date='2023-12-10',
            is_back_to_back=False,
            injury_list=[]
        )
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)  # 5 players in our test data
        
        # Check that the columns are correct
        expected_columns = ['TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_ID', 'PLAYER_NAME', 
                           'POSITION', 'DEPTH', 'PREDICTED_MINUTES', 'GAME_DATE', 
                           'IS_INJURED', 'IS_BACK_TO_BACK']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check that players are sorted by predicted minutes
        self.assertEqual(result.iloc[0]['PLAYER_ID'], 1)  # Player 1 should have most minutes
        self.assertEqual(result.iloc[-1]['PLAYER_ID'], 5)  # Player 5 should have least minutes
        
        # Test with injuries
        result = predict_team_minutes(
            self.game_logs_df,
            self.depth_chart_df,
            team_id=101,
            game_date='2023-12-10',
            is_back_to_back=False,
            injury_list=[1]  # Player 1 is injured
        )
        
        # Player 1 should have 0 minutes
        player1_row = result[result['PLAYER_ID'] == 1].iloc[0]
        self.assertEqual(player1_row['PREDICTED_MINUTES'], 0.0)
        self.assertTrue(player1_row['IS_INJURED'])
    
    def test_predict_starting_lineup(self):
        """Test the predict_starting_lineup function."""
        # First predict team minutes
        team_minutes_df = predict_team_minutes(
            self.game_logs_df,
            self.depth_chart_df,
            team_id=101,
            game_date='2023-12-10',
            is_back_to_back=False,
            injury_list=[]
        )
        
        # Call the function
        result = predict_starting_lineup(team_minutes_df)
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)  # 5 players in starting lineup
        
        # Check that we have one player for each position
        positions = sorted(result['POSITION'].unique())
        self.assertEqual(positions, ['C', 'PF', 'PG', 'SF', 'SG'])
        
        # Test with an injured starter
        team_minutes_df = predict_team_minutes(
            self.game_logs_df,
            self.depth_chart_df,
            team_id=101,
            game_date='2023-12-10',
            is_back_to_back=False,
            injury_list=[1]  # Player 1 (PG) is injured
        )
        
        # In this case, we don't have a backup PG in our test data
        # So the function should return the best available players
        result = predict_starting_lineup(team_minutes_df)
        self.assertEqual(len(result), 4)  # Only 4 healthy players
    
    def test_predict_rotation_pattern(self):
        """Test the predict_rotation_pattern function."""
        # First predict team minutes
        team_minutes_df = predict_team_minutes(
            self.game_logs_df,
            self.depth_chart_df,
            team_id=101,
            game_date='2023-12-10',
            is_back_to_back=False,
            injury_list=[]
        )
        
        # Call the function
        result = predict_rotation_pattern(team_minutes_df)
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)  # 5 players in our test data
        
        # Check that the columns are correct
        expected_columns = ['PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'TOTAL_MINUTES', 
                           'Q1', 'Q2', 'Q3', 'Q4']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check that total minutes match predicted minutes
        for _, row in result.iterrows():
            player_id = row['PLAYER_ID']
            total_minutes = row['TOTAL_MINUTES']
            predicted_minutes = team_minutes_df[team_minutes_df['PLAYER_ID'] == player_id].iloc[0]['PREDICTED_MINUTES']
            self.assertEqual(total_minutes, predicted_minutes)
            
            # Check that quarter minutes sum up to total minutes (allow for small rounding differences)
            quarter_sum = row['Q1'] + row['Q2'] + row['Q3'] + row['Q4']
            # Use a larger tolerance to account for rounding in the implementation
            self.assertAlmostEqual(quarter_sum, total_minutes, delta=2.0)

if __name__ == '__main__':
    unittest.main() 