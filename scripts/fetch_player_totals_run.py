from fetch_player_totals import fetch_nba_player_totals
from tqdm import tqdm

# Iterate through seasons from 2013-14 to 2022-23
for year in tqdm(range(2013, 2023), desc="Fetching seasons", unit="season"):
    season = f"{year}-{str(year+1)[2:]}"  # Format like "2013-14"
    output_file = f"data/raw/totals_{year}.json"
    fetch_nba_player_totals(season, output_file=output_file)