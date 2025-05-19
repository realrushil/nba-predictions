import pandas as pd
import json
import os
import glob
from fuzzywuzzy import process, fuzz

# Load RAPTOR ratings
print("Loading RAPTOR data...")
raptor_df = pd.read_csv('data/raptor/modern_RAPTOR_by_player.csv')
print(f"Sample RAPTOR data:")
print(raptor_df[['player_name', 'player_id', 'season']].head(10))

# Load a sample of NBA stats player data
print("\nLoading NBA stats data...")
json_sample_file = glob.glob("data/normalized/totals_*.json")[0]
with open(json_sample_file, 'r') as f:
    nba_data = json.load(f)

print(f"Sample NBA players:")
for player in nba_data[:10]:
    print(f"Name: {player.get('Name', 'N/A')}, ID: {player.get('RowId', 'N/A')}")

# Find matches based on names
print("\nTrying to match players by name...")
raptor_names = raptor_df['player_name'].tolist()
nba_names = [player.get('Name', '') for player in nba_data if player.get('Name')]

# Get a sample of 5 NBA player names
sample_nba_names = nba_names[:5]

for nba_name in sample_nba_names:
    # Find best match in RAPTOR names
    best_match, score = process.extractOne(nba_name, raptor_names, scorer=fuzz.token_sort_ratio)
    print(f"NBA name: {nba_name} -> Best RAPTOR match: {best_match} (Score: {score})")

# Create a mapping function
def find_raptor_match(nba_name, raptor_names, threshold=80):
    best_match, score = process.extractOne(nba_name, raptor_names, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return best_match
    return None

# Find a common season between datasets
nba_season = os.path.basename(json_sample_file).replace("totals_", "").replace(".json", "")
print(f"\nNBA stats season: {nba_season}")

if nba_season in raptor_df['season'].astype(str).values:
    print(f"Season {nba_season} found in RAPTOR data")
    
    # Get RAPTOR players for that season
    raptor_season_players = raptor_df[raptor_df['season'].astype(str) == nba_season]['player_name'].tolist()
    print(f"Number of RAPTOR players in season {nba_season}: {len(raptor_season_players)}")
    
    # Find potential matches for this season
    matches = 0
    high_confidence_matches = 0
    
    for nba_player in nba_data[:100]:  # Check first 100 players
        nba_name = nba_player.get('Name', '')
        if not nba_name:
            continue
            
        match = find_raptor_match(nba_name, raptor_season_players)
        if match:
            matches += 1
            if process.extractOne(nba_name, [match], scorer=fuzz.token_sort_ratio)[1] >= 90:
                high_confidence_matches += 1
    
    print(f"Found {matches} potential matches out of 100 players")
    print(f"High confidence matches (>=90% similarity): {high_confidence_matches}")
else:
    print(f"Season {nba_season} not found in RAPTOR data")
    
print("\nConclusions:")
print("1. Player IDs in the two datasets are completely different formats")
print("2. Need to create a name-based mapping to join the datasets")
print("3. Will need to use fuzzy matching for player names as they may not be identical") 