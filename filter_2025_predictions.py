import json
import os

# Minutes threshold
MIN_MINUTES = 100

# Check if we have normalized data for 2024 (most recent available) to use as reference
normalized_file = 'data/normalized/totals_2024.json'
prediction_file = 'models/predictions_2025.json'

if not os.path.exists(normalized_file):
    print(f"Error: Missing required file {normalized_file}")
    exit(1)
    
if not os.path.exists(prediction_file):
    print(f"Error: Missing required file {prediction_file}")
    exit(1)

print(f"Processing 2025 predictions file with 2024 minutes data...")

# Load normalized data (contains minutes played)
with open(normalized_file, 'r', encoding='utf-8') as f:
    normalized_data = json.load(f)

# Create a dictionary of player minutes
player_minutes = {}
for player in normalized_data:
    name = player.get('Name')
    minutes = player.get('Minutes', 0)
    if name:
        player_minutes[name] = minutes

# Load prediction data
with open(prediction_file, 'r', encoding='utf-8') as f:
    predictions = json.load(f)

# Filter out players with less than MIN_MINUTES
filtered_predictions = []
excluded_count = 0

for player in predictions:
    player_name = player.get('player_name')
    minutes = player_minutes.get(player_name, 0)
    
    if minutes >= MIN_MINUTES:
        filtered_predictions.append(player)
    else:
        excluded_count += 1
        print(f"  Excluding {player_name} with only {minutes} minutes")

# Save the filtered predictions back to the file
with open(prediction_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_predictions, f, indent=2)

print(f"  Year 2025: Kept {len(filtered_predictions)} players, excluded {excluded_count} players")
print("Predictions updated!") 