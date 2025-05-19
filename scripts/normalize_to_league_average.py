import json
import os
import glob
from pathlib import Path
import numpy as np

# Create the output directory if it doesn't exist
output_dir = Path("data/normalized")
output_dir.mkdir(exist_ok=True)

# Define stats that should NOT be normalized relative to league average
non_normalized_stats = {
    # Identifiers and team info
    "EntityId", "TeamId", "Name", "ShortName", "RowId", "TeamAbbreviation",
    # Game-specific metrics
    "PlusMinus", "SecondsPlayed", "Minutes", "TotalPoss", "OffPoss", "DefPoss",
    "PenaltyOffPoss", "PenaltyDefPoss", "SecondChanceOffPoss",
    # Percentage values
    # Any field with "Pct" will be excluded in the loop
}

# Process each JSON file in the processed directory
for json_file in glob.glob("data/processed/totals_*.json"):
    filename = os.path.basename(json_file)
    output_file = os.path.join("data/normalized", filename)
    
    print(f"Processing {json_file}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Calculate league averages for all numeric fields
    league_avgs = {}
    player_count = len(data)
    
    # First pass: collect all fields and sum values
    for player in data:
        for key, value in player.items():
            if isinstance(value, (int, float)) and key not in non_normalized_stats and "Pct" not in key:
                if key not in league_avgs:
                    league_avgs[key] = 0.0
                league_avgs[key] += value
    
    # Calculate averages
    for key in league_avgs:
        league_avgs[key] /= player_count
    
    print(f"Calculated league averages for {len(league_avgs)} stats")
    
    # Second pass: normalize player stats relative to league average
    normalized_data = []
    
    for player in data:
        normalized_player = {}
        
        for key, value in player.items():
            # Keep non-normalized stats as is
            if key in non_normalized_stats or "Pct" in key:
                normalized_player[key] = value
            # Normalize numeric stats relative to league average
            elif isinstance(value, (int, float)) and key in league_avgs:
                # Calculate relative to league average
                normalized_player[key] = value - league_avgs[key]
            else:
                normalized_player[key] = value
                
        normalized_data.append(normalized_player)
    
    # Save the normalized data
    with open(output_file, 'w') as f:
        json.dump(normalized_data, f, indent=2)
    
    print(f"Saved normalized data to {output_file}")

print("All files normalized successfully!") 