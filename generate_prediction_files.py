import pandas as pd
import json
import os

# Load the RAPTOR data
raptor_data = pd.read_csv('data/raptor/modern_RAPTOR_by_player.csv')

# Get unique seasons
seasons = sorted(raptor_data['season'].unique())

# Remove any existing prediction files for these years (except the ones we want to keep)
keep_files = ['predictions_2022.json', 'predictions_2025.json']
for file in os.listdir('models'):
    if file.startswith('predictions_') and file.endswith('.json') and file not in keep_files:
        os.remove(os.path.join('models', file))

# Generate prediction files for each season
for season in seasons:
    # Filter for the given season
    season_data = raptor_data[raptor_data['season'] == season]
    
    # Filter out players with less than 100 minutes total
    season_data = season_data[season_data['mp'] >= 100]
    
    # Select relevant columns and sort by total RAPTOR rating
    players = season_data[['player_name', 'raptor_offense', 'raptor_defense', 'raptor_total']]
    players = players.sort_values('raptor_total', ascending=False)
    
    # Rename columns to match the existing prediction files
    players = players.rename(columns={
        'raptor_offense': 'offensive_raptor',
        'raptor_defense': 'defensive_raptor',
        'raptor_total': 'total_raptor'
    })
    
    # Convert to list of dictionaries
    players_list = players.to_dict('records')
    
    # Save to JSON file
    output_file = f'models/predictions_{season}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(players_list, f, indent=2)
    
    print(f"Created {output_file} with {len(players_list)} players")

print("All prediction files generated successfully!") 