import json
import os
import glob
from pathlib import Path

# Create the processed directory if it doesn't exist
processed_dir = Path("data/processed")
processed_dir.mkdir(exist_ok=True)

# Define stats that should NOT be scaled (percentages, identifiers, etc.)
non_scaled_stats = {
    "EntityId", "TeamId", "Name", "ShortName", "RowId", "TeamAbbreviation", 
    "PlusMinus", "SecondsPlayed", "Minutes", "TotalPoss", "OffPoss", "DefPoss",
    "OnOffRtg", "OnDefRtg",  # These are already per-100 possession metrics
    # Percentages are derived from other stats (FGM/FGA), so they don't need scaling
}

# Process each JSON file in the raw directory
for json_file in glob.glob("data/raw/totals_*.json"):
    filename = os.path.basename(json_file)
    output_file = os.path.join("data/processed", filename)
    
    print(f"Processing {json_file}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    processed_data = []
    
    for player in data:
        # Skip players with no possessions to avoid division by zero
        if player.get("TotalPoss", 0) == 0:
            processed_data.append(player)
            continue
            
        # Create a new player dict with scaled stats
        processed_player = {}
        
        # Calculate the scaling factor (per 100 possessions)
        scaling_factor = 100 / player["TotalPoss"]
        
        for key, value in player.items():
            # Don't scale certain stats
            if key in non_scaled_stats or "Pct" in key:
                processed_player[key] = value
            # Scale numeric stats
            elif isinstance(value, (int, float)):
                processed_player[key] = value * scaling_factor
            else:
                processed_player[key] = value
                
        processed_data.append(processed_player)
    
    # Save the processed data
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Saved processed data to {output_file}")

print("All files processed successfully!") 