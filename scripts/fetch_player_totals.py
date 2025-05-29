import requests
import time
import json
import random
import argparse

# All 30 NBA team IDs
NBA_TEAM_IDS = [
    "1610612737", "1610612738", "1610612739", "1610612740", "1610612741",
    "1610612742", "1610612743", "1610612744", "1610612745", "1610612746",
    "1610612747", "1610612748", "1610612749", "1610612750", "1610612751",
    "1610612752", "1610612753", "1610612754", "1610612755", "1610612756",
    "1610612757", "1610612758", "1610612759", "1610612760", "1610612761",
    "1610612762", "1610612763", "1610612764", "1610612765", "1610612766"
]

def fetch_nba_player_totals(season: str, stat_type: str = None, output_file: str = None, verbose: bool = False):
    """
    Fetches NBA player totals for all teams in a given season with retry on failure.
    
    Args:
        season (str): e.g. "2021-22"
        stat_type (str): Optional - "Traditional", "Advanced", "Scoring", etc.
        output_file (str): Optional - output file path (default: auto-generated)
        verbose (bool): Whether to print detailed debug information
        
    Returns:
        List of player stat dictionaries
    """
    all_player_data = []
    MAX_RETRIES = 5

    print(f"\n🏀 Fetching player data for {season} season...")

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

        if verbose:
            print(f"\n📡 Making request to {url}")
            print(f"Parameters: {json.dumps(params, indent=2)}")

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, params=params, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if verbose:
                        print(f"Response: {json.dumps(data, indent=2)[:500]}...")
                    players = data.get("multi_row_table_data", [])
                    all_player_data.extend(players)
                    print(f"✅ {team_id}: {len(players)} players fetched")
                    break  # Exit retry loop on success
                else:
                    print(f"⚠️  {team_id}: Error {response.status_code}, attempt {attempt + 1}")
                    if verbose:
                        print(f"Response: {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"❌ {team_id}: Network error: {e}, attempt {attempt + 1}")

            # Wait before retrying
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2 ** attempt + random.uniform(0, 1)
                print(f"Waiting {sleep_time:.1f} seconds before retry...")
                time.sleep(sleep_time)
        else:
            print(f"❌ {team_id}: Failed after {MAX_RETRIES} attempts")

        time.sleep(0.5)  # To be nice to the server

    print(f"\n📊 Total players fetched: {len(all_player_data)}")

    # Determine output file name if not provided
    if not output_file:
        suffix = stat_type.lower() if stat_type else "totals"
        output_file = f"nba_{season.replace('-', '_')}_{suffix}.json"

    if len(all_player_data) > 0:
        with open(output_file, "w") as f:
            json.dump(all_player_data, f, indent=2)
        print(f"📁 Saved {len(all_player_data)} player records to '{output_file}'")
    else:
        print("❌ No player data was fetched. Check the API response and parameters.")

    return all_player_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA player totals")
    parser.add_argument("--season", required=True, help="Season (e.g. '2021-22')")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--stat-type", help="Stat type (Traditional, Advanced, etc.)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed debug information")
    
    args = parser.parse_args()
    fetch_nba_player_totals(args.season, args.stat_type, args.output, args.verbose)
