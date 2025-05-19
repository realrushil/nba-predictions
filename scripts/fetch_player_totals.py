import requests
import time
import json
import random

# All 30 NBA team IDs
NBA_TEAM_IDS = [
    "1610612737", "1610612738", "1610612739", "1610612740", "1610612741",
    "1610612742", "1610612743", "1610612744", "1610612745", "1610612746",
    "1610612747", "1610612748", "1610612749", "1610612750", "1610612751",
    "1610612752", "1610612753", "1610612754", "1610612755", "1610612756",
    "1610612757", "1610612758", "1610612759", "1610612760", "1610612761",
    "1610612762", "1610612763", "1610612764", "1610612765", "1610612766"
]

def fetch_nba_player_totals(season: str, stat_type: str = None, output_file: str = None):
    """
    Fetches NBA player totals for all teams in a given season with retry on failure.
    
    Args:
        season (str): e.g. "2021-22"
        stat_type (str): Optional - "Traditional", "Advanced", "Scoring", etc.
        output_file (str): Optional - output file path (default: auto-generated)
        
    Returns:
        List of player stat dictionaries
    """
    all_player_data = []
    MAX_RETRIES = 5

    for team_id in NBA_TEAM_IDS:
        url = "https://api.pbpstats.com/get-totals/nba"
        params = {
            "Season": season,
            "SeasonType": "Regular Season",
            "Type": "Player",
            "Leverage": "Medium,High,VeryHigh",
            "StarterState": "All",
            "TeamId": team_id,
            "GroupBy": "Season",
            "StartType": "All"
        }
        if stat_type:
            params["StatType"] = stat_type

        headers = {
            "accept": "application/json"
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, params=params, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    players = data.get("multi_row_table_data", [])
                    all_player_data.extend(players)
                    print(f"✅ {team_id}: {len(players)} players fetched")
                    break  # Exit retry loop on success
                else:
                    print(f"⚠️  {team_id}: Error {response.status_code}, attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                print(f"❌ {team_id}: Network error: {e}, attempt {attempt + 1}")

            # Wait before retrying
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2 ** attempt + random.uniform(0, 1)
                time.sleep(sleep_time)
        else:
            print(f"❌ {team_id}: Failed after {MAX_RETRIES} attempts")

        time.sleep(0.5)  # To be nice to the server

    # Determine output file name if not provided
    if not output_file:
        suffix = stat_type.lower() if stat_type else "totals"
        output_file = f"nba_{season.replace('-', '_')}_{suffix}.json"

    with open(output_file, "w") as f:
        json.dump(all_player_data, f, indent=2)

    print(f"\n📁 Saved {len(all_player_data)} player records to '{output_file}'")
    return all_player_data
