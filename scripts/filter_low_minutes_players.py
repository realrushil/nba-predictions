import json
import os

# Minutes threshold
MIN_MINUTES = 100

# Process 2024 predictions
year = 2024
normalized_file = f'data/normalized/totals_{year}.json'
prediction_file = f'models/predictions_{year}.json'

print(f"Processing year {year}...")

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

print(f"  Year {year}: Kept {len(filtered_predictions)} players, excluded {excluded_count} players")
print("Predictions updated!") 