import pandas as pd
import json
import os
import glob

# Load RAPTOR ratings
print("Loading RAPTOR data...")
raptor_df = pd.read_csv('data/raptor/modern_RAPTOR_by_player.csv')
print(f"RAPTOR dataset has {len(raptor_df)} rows")
print(f"Number of unique players in RAPTOR: {len(raptor_df['player_id'].unique())}")
print(f"Number of unique seasons in RAPTOR: {len(raptor_df['season'].unique())}")
print(f"Sample player IDs from RAPTOR: {raptor_df['player_id'].head(5).tolist()}")
print(f"Sample seasons from RAPTOR: {raptor_df['season'].head(5).tolist()}")
print(f"Sample player names from RAPTOR: {raptor_df['player_name'].head(5).tolist()}")

# Load a sample of NBA stats player data
print("\nLoading NBA stats data...")
json_sample_file = glob.glob("data/normalized/totals_*.json")[0]
with open(json_sample_file, 'r') as f:
    nba_data = json.load(f)
    
print(f"Sample NBA stats file: {json_sample_file}")
print(f"NBA stats dataset has {len(nba_data)} players")
print(f"Sample player IDs from NBA: {[player.get('RowId', 'N/A') for player in nba_data[:5]]}")
print(f"Sample player names from NBA: {[player.get('Name', 'N/A') for player in nba_data[:5]]}")

# Check for overlapping player IDs
raptor_ids = set(raptor_df['player_id'])
nba_ids = set(player.get('RowId', '') for player in nba_data if player.get('RowId'))

print(f"\nNumber of unique player IDs in RAPTOR: {len(raptor_ids)}")
print(f"Number of unique player IDs in NBA stats: {len(nba_ids)}")
print(f"Number of overlapping player IDs: {len(raptor_ids.intersection(nba_ids))}")

# Print a sample of common players
common_ids = raptor_ids.intersection(nba_ids)
print(f"Sample of common player IDs: {list(common_ids)[:5] if common_ids else 'None'}")

# Analyze specific examples
if common_ids:
    common_id = list(common_ids)[0]
    raptor_player = raptor_df[raptor_df['player_id'] == common_id].iloc[0]
    nba_player = next((p for p in nba_data if p.get('RowId') == common_id), None)
    
    print(f"\nExample of a common player:")
    print(f"RAPTOR player: {raptor_player['player_name']} (ID: {raptor_player['player_id']})")
    print(f"NBA stats player: {nba_player.get('Name')} (ID: {nba_player.get('RowId')})")
else:
    print("\nAnalyzing format differences:")
    raptor_id_sample = list(raptor_ids)[:3]
    nba_id_sample = list(nba_ids)[:3]
    
    print(f"RAPTOR ID examples: {raptor_id_sample}")
    print(f"NBA stats ID examples: {nba_id_sample}")

# Get format of IDs
raptor_id_format = {
    "alphabetic": any(c.isalpha() for c in next(iter(raptor_ids), "")),
    "numeric": any(c.isdigit() for c in next(iter(raptor_ids), "")),
    "length": len(next(iter(raptor_ids), ""))
}

nba_id_format = {
    "alphabetic": any(c.isalpha() for c in next(iter(nba_ids), "")),
    "numeric": any(c.isdigit() for c in next(iter(nba_ids), "")),
    "length": len(next(iter(nba_ids), ""))
}

print(f"\nRAPTOR ID format: {raptor_id_format}")
print(f"NBA stats ID format: {nba_id_format}")

# Print sample modification to match
print("\nPossible solutions:")
if raptor_id_format['alphabetic'] and nba_id_format['numeric']:
    print("→ Need to map alphabetic RAPTOR IDs to numeric NBA IDs")
    print("→ Create a name-based mapping between datasets")
elif nba_id_format['alphabetic'] and raptor_id_format['numeric']:
    print("→ Need to map numeric RAPTOR IDs to alphabetic NBA IDs")
    print("→ Create a name-based mapping between datasets")
else:
    print("→ ID formats may be compatible but still not matching")
    print("→ Try mapping by player name instead of ID") 