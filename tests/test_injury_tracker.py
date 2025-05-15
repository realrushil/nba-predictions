import unittest
import os
import json
import pandas as pd
import sys
import tempfile
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.injury_tracker import (
    add_injured_player,
    remove_injured_player,
    get_team_injuries,
    get_all_injured_player_ids,
    export_injury_report
)

class TestInjuryTracker(unittest.TestCase):
    """Test cases for the injury_tracker module."""
    
    def setUp(self):
        """Set up test data and temporary files."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.injury_file = os.path.join(self.temp_dir.name, 'test_injuries.json')
        
        # Sample injury data
        self.sample_injury_data = {
            '101': {
                'player_ids': [1, 2],
                'players': [
                    {
                        'player_id': 1,
                        'player_name': 'Player 1',
                        'injury_type': 'Ankle Sprain',
                        'expected_return_date': '2023-12-15',
                        'date_added': '2023-12-01',
                        'last_updated': '2023-12-01'
                    },
                    {
                        'player_id': 2,
                        'player_name': 'Player 2',
                        'injury_type': 'Knee Soreness',
                        'expected_return_date': '2023-12-10',
                        'date_added': '2023-12-01',
                        'last_updated': '2023-12-01'
                    }
                ]
            }
        }
        
        # Write sample data to the injury file
        with open(self.injury_file, 'w') as f:
            json.dump(self.sample_injury_data, f)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_add_injured_player(self):
        """Test the add_injured_player function."""
        # Add a new player to an existing team
        result = add_injured_player(
            self.injury_file,
            team_id=101,
            player_id=3,
            player_name='Player 3',
            injury_type='Concussion',
            expected_return_date='2023-12-20'
        )
        
        # Verify the result
        self.assertTrue(result)
        
        # Check that the player was added to the file
        with open(self.injury_file, 'r') as f:
            injury_data = json.load(f)
        
        self.assertIn('101', injury_data)
        self.assertIn(3, injury_data['101']['player_ids'])
        
        # Find the added player
        added_player = None
        for player in injury_data['101']['players']:
            if player['player_id'] == 3:
                added_player = player
                break
        
        self.assertIsNotNone(added_player)
        self.assertEqual(added_player['player_name'], 'Player 3')
        self.assertEqual(added_player['injury_type'], 'Concussion')
        self.assertEqual(added_player['expected_return_date'], '2023-12-20')
        
        # Add a player to a new team
        result = add_injured_player(
            self.injury_file,
            team_id=102,
            player_id=4,
            player_name='Player 4',
            injury_type='Hamstring',
            expected_return_date='2023-12-25'
        )
        
        # Verify the result
        self.assertTrue(result)
        
        # Check that the new team and player were added to the file
        with open(self.injury_file, 'r') as f:
            injury_data = json.load(f)
        
        self.assertIn('102', injury_data)
        self.assertIn(4, injury_data['102']['player_ids'])
    
    def test_remove_injured_player(self):
        """Test the remove_injured_player function."""
        # Remove an existing player
        result = remove_injured_player(
            self.injury_file,
            team_id=101,
            player_id=1
        )
        
        # Verify the result
        self.assertTrue(result)
        
        # Check that the player was removed from the file
        with open(self.injury_file, 'r') as f:
            injury_data = json.load(f)
        
        self.assertIn('101', injury_data)
        self.assertNotIn(1, injury_data['101']['player_ids'])
        
        # Check that the other player is still there
        self.assertIn(2, injury_data['101']['player_ids'])
        
        # Remove the last player from a team
        result = remove_injured_player(
            self.injury_file,
            team_id=101,
            player_id=2
        )
        
        # Verify the result
        self.assertTrue(result)
        
        # Check that the team was removed from the file
        with open(self.injury_file, 'r') as f:
            injury_data = json.load(f)
        
        self.assertNotIn('101', injury_data)
        
        # Try to remove a non-existent player
        result = remove_injured_player(
            self.injury_file,
            team_id=101,
            player_id=999
        )
        
        # Should return False for non-existent player
        self.assertFalse(result)
    
    def test_get_team_injuries(self):
        """Test the get_team_injuries function."""
        # Get injuries for an existing team
        injuries = get_team_injuries(self.injury_file, team_id=101)
        
        # Verify the result
        self.assertEqual(len(injuries), 2)
        self.assertEqual(injuries[0]['player_id'], 1)
        self.assertEqual(injuries[1]['player_id'], 2)
        
        # Get injuries for a non-existent team
        injuries = get_team_injuries(self.injury_file, team_id=999)
        
        # Should return empty list for non-existent team
        self.assertEqual(injuries, [])
    
    def test_get_all_injured_player_ids(self):
        """Test the get_all_injured_player_ids function."""
        # Get all injured player IDs
        injured_player_ids = get_all_injured_player_ids(self.injury_file)
        
        # Verify the result
        self.assertIn('101', injured_player_ids)
        self.assertEqual(injured_player_ids['101'], [1, 2])
        
        # Add a player to a new team
        add_injured_player(
            self.injury_file,
            team_id=102,
            player_id=3,
            player_name='Player 3',
            injury_type='Concussion'
        )
        
        # Get all injured player IDs again
        injured_player_ids = get_all_injured_player_ids(self.injury_file)
        
        # Verify the result
        self.assertIn('101', injured_player_ids)
        self.assertIn('102', injured_player_ids)
        self.assertEqual(injured_player_ids['102'], [3])
    
    def test_export_injury_report(self):
        """Test the export_injury_report function."""
        # Create a temporary file for the report
        report_file = os.path.join(self.temp_dir.name, 'test_injury_report.csv')
        
        # Export the injury report
        result = export_injury_report(self.injury_file, report_file)
        
        # Verify the result
        self.assertTrue(result)
        self.assertTrue(os.path.exists(report_file))
        
        # Read the report
        report_df = pd.read_csv(report_file)
        
        # Verify the report contents
        self.assertEqual(len(report_df), 2)
        self.assertEqual(list(report_df['PLAYER_ID']), [1, 2])
        
        # Convert team IDs to strings for comparison since they're stored as strings in JSON
        team_ids = [str(team_id) for team_id in report_df['TEAM_ID']]
        self.assertEqual(team_ids, ['101', '101'])
        
        self.assertEqual(list(report_df['INJURY_TYPE']), ['Ankle Sprain', 'Knee Soreness'])

if __name__ == '__main__':
    unittest.main() 