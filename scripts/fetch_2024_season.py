from fetch_player_totals import fetch_nba_player_totals
import os
from pathlib import Path

# Create raw directory if it doesn't exist
raw_dir = Path("data/raw")
raw_dir.mkdir(exist_ok=True, parents=True)

print("Fetching NBA player totals for the 2024-25 season...")
season = "2024-25"
output_file = "data/raw/totals_2024.json"
fetch_nba_player_totals(season, output_file=output_file)

print("Data fetching complete.") 