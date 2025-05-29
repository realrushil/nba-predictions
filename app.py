import os
import json
import pandas as pd
import re
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

# NBA player ID mapping (for displaying player images)
# This is a small sample of known player IDs - in a real app, you'd want a complete database
PLAYER_IDS = {
    # Established Stars
    "LeBron James": "2544",
    "Kevin Durant": "201142",
    "Stephen Curry": "201939",
    "Giannis Antetokounmpo": "203507",
    "Nikola Jokić": "203999",
    "Luka Dončić": "1629029",
    "Joel Embiid": "203954",
    "Kawhi Leonard": "202695",
    "James Harden": "201935",
    "Anthony Davis": "203076",
    "Damian Lillard": "203081",
    "Jayson Tatum": "1628369",
    "Jimmy Butler": "202710",
    "Kyrie Irving": "202681",
    "Donovan Mitchell": "1628378",
    "Devin Booker": "1626164",
    "Karl-Anthony Towns": "1626157",
    "Trae Young": "1629027",
    "Ja Morant": "1629630",
    "Zion Williamson": "1629627",
    "Bam Adebayo": "1628389",
    "Jaylen Brown": "1627759",
    "Brandon Ingram": "1627742",
    "Shai Gilgeous-Alexander": "1628983",
    "Zach LaVine": "203897",
    "Bradley Beal": "203078",
    "Russell Westbrook": "201566",
    "Chris Paul": "101108",
    "Paul George": "202331",
    "Klay Thompson": "202691",
    "Draymond Green": "203110",
    
    # Young Stars & Prospects
    "Victor Wembanyama": "1641705",
    "Chet Holmgren": "1631096",
    "Paolo Banchero": "1631094",
    "LaMelo Ball": "1630163",
    "Anthony Edwards": "1630162",
    "Cade Cunningham": "1630595",
    "Jalen Green": "1630224",
    "Franz Wagner": "1630532",
    "Scottie Barnes": "1630567",
    "Evan Mobley": "1630596",
    "Josh Giddey": "1630581",
    "Jalen Suggs": "1630591",
    "Alperen Sengun": "1630578",
    "Jaden Ivey": "1631093",
    "Keegan Murray": "1631099",
    "Jabari Smith Jr.": "1631095",
    "Dyson Daniels": "1631109",
    "Bennedict Mathurin": "1631097",
    "Johnny Davis": "1631098",
    "Jalen Williams": "1631107",
    "AJ Griffin": "1631117",
    "Nikola Jovic": "1631107",
    "Walker Kessler": "1631124",
    
    # Rising Stars
    "Tyrese Maxey": "1630178",
    "Tyrese Haliburton": "1630169",
    "Jalen Brunson": "1628973",
    "Jordan Poole": "1629673",
    "Darius Garland": "1629636",
    "Tyler Herro": "1629639",
    "Desmond Bane": "1630217",
    "Anfernee Simons": "1629014",
    "Jarrett Allen": "1628386",
    "Mikal Bridges": "1628969",
    "Deandre Ayton": "1629028",
    "Collin Sexton": "1629012",
    "Michael Porter Jr.": "1629008",
    "Miles Bridges": "1628970",
    "De'Aaron Fox": "1628368",
    "Jaren Jackson Jr.": "1628991",
    
    # Established Veterans
    "Demar DeRozan": "201942",
    "Jrue Holiday": "201950",
    "Khris Middleton": "203114",
    "Fred VanVleet": "1627832",
    "Pascal Siakam": "1627783",
    "CJ McCollum": "203468",
    "Dejounte Murray": "1627749",
    "Aaron Gordon": "203932",
    "Julius Randle": "203944",
    "Tobias Harris": "202699",
    "John Collins": "1628381",
    "Myles Turner": "1626167",
    "Kristaps Porziņģis": "204001",
    "Derrick White": "1628401",
    "Marcus Smart": "203935",
    "Jerami Grant": "203924",
    "Robert Williams III": "1629057",
    "Wendell Carter Jr.": "1628976",
    "Jusuf Nurkić": "203994",
    "Malcolm Brogdon": "1627763",
    
    # Role Players
    "Alex Caruso": "1627936",
    "Matisse Thybulle": "1629680",
    "Caris LeVert": "1627747",
    "Kyle Kuzma": "1628398",
    "Jordan Clarkson": "203903",
    "Bojan Bogdanović": "202711",
    "Buddy Hield": "1627741",
    "Bogdan Bogdanović": "203992",
    "Gary Trent Jr.": "1629018",
    
    # Additional Players (Top Rated in the data)
    "Bobi Klintman": "1631132",
    "Yuki Kawamura": "1631610",
    "Jack McVeigh": "1630590",
    "MarJon Beauchamp": "1630547",
    "Ty Jerome": "1629660",
    "Alperen Sengun": "1630578",
}

def get_player_id(player_name):
    """
    Get NBA.com player ID for a given player name.
    First tries the lookup dictionary, then attempts to generate a fallback ID.
    """
    # First check if we have the player ID in our dictionary
    if player_name in PLAYER_IDS:
        return PLAYER_IDS[player_name]
    
    # For players not in our dictionary, try to normalize the name and lookup again
    # Some player names might have special characters or different formats
    normalized_name = player_name.lower().replace("'", "").replace(".", "")
    
    # Check if any key in the dictionary matches when normalized
    for name, player_id in PLAYER_IDS.items():
        if name.lower().replace("'", "").replace(".", "") == normalized_name:
            PLAYER_IDS[player_name] = player_id  # Add to our dictionary for future lookups
            return player_id
    
    # Return a default fallback image if we can't find the player
    return "fallback"

def load_player_data(year='2025'):
    """Load player data from prediction files"""
    file_path = os.path.join('models', f'predictions_{year}.json')
    
    if not os.path.exists(file_path):
        available_files = [f for f in os.listdir('models') if f.startswith('predictions_') and f.endswith('.json')]
        if available_files:
            file_path = os.path.join('models', available_files[0])
            year = available_files[0].replace('predictions_', '').replace('.json', '')
        else:
            return pd.DataFrame(), year
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    # Sort by total_raptor descending by default
    df = df.sort_values('total_raptor', ascending=False).reset_index(drop=True)
    
    # Add rank column
    df['rank'] = df.index + 1
    
    # Add player_id column for images using the get_player_id function
    df['player_id'] = df['player_name'].map(get_player_id)
    
    return df, year

def get_available_years():
    """Get list of available prediction years"""
    files = [f for f in os.listdir('models') if f.startswith('predictions_') and f.endswith('.json')]
    years = [f.replace('predictions_', '').replace('.json', '') for f in files]
    return sorted(years, reverse=True)  # Sort in descending order (newest first)

def find_similar_players(df, player_name, n=5):
    """Find players with similar RAPTOR ratings"""
    if player_name not in df['player_name'].values:
        return []
    
    player = df[df['player_name'] == player_name].iloc[0]
    
    # Calculate distance in RAPTOR space
    df['distance'] = ((df['offensive_raptor'] - player['offensive_raptor']) ** 2 + 
                       (df['defensive_raptor'] - player['defensive_raptor']) ** 2) ** 0.5
    
    # Get similar players (excluding the player itself)
    similar_players = df[df['player_name'] != player_name].sort_values('distance').head(n)
    
    return similar_players[['player_name', 'offensive_raptor', 'defensive_raptor', 'total_raptor', 'rank', 'player_id']].to_dict('records')

@app.route('/')
def index():
    years = get_available_years()
    selected_year = request.args.get('year', years[0] if years else '2025')
    
    df, year = load_player_data(selected_year)
    
    # Get top 100 players by default
    limit = int(request.args.get('limit', 100))
    metric = request.args.get('metric', 'overall')
    
    # Filter players based on the selected metric
    if metric == 'offense':
        df = df.sort_values('offensive_raptor', ascending=False)
    elif metric == 'defense':
        df = df.sort_values('defensive_raptor', ascending=False)
    else:
        df = df.sort_values('total_raptor', ascending=False)
    
    players = df.head(limit).to_dict('records')
    
    return render_template(
        'index.html',
        players=players,
        years=years,
        selected_year=year,
        limit=limit,
        metric=metric
    )

@app.route('/player/<player_name>')
def player_detail(player_name):
    year = request.args.get('year', '2025')
    
    df, year = load_player_data(year)
    
    # Find the player
    if player_name not in df['player_name'].values:
        return redirect(url_for('index'))
    
    player = df[df['player_name'] == player_name].iloc[0].to_dict()
    rank = df[df['player_name'] == player_name].iloc[0]['rank']
    
    # Find similar players
    similar_players = find_similar_players(df, player_name)
    
    return render_template(
        'player.html',
        player=player,
        rank=rank,
        similar_players=similar_players,
        year=year
    )

@app.route('/api/players')
def api_players():
    year = request.args.get('year', '2025')
    limit = int(request.args.get('limit', 100))
    
    df, year = load_player_data(year)
    
    players = df.head(limit).to_dict('records')
    
    return jsonify({
        'players': players,
        'year': year,
        'limit': limit
    })

@app.route('/api/player/<player_name>')
def api_player_detail(player_name):
    year = request.args.get('year', '2025')
    
    df, year = load_player_data(year)
    
    if player_name not in df['player_name'].values:
        return jsonify({'error': 'Player not found'}), 404
    
    player = df[df['player_name'] == player_name].iloc[0].to_dict()
    rank = df[df['player_name'] == player_name].iloc[0]['rank']
    similar_players = find_similar_players(df, player_name)
    
    return jsonify({
        'player': player,
        'rank': rank,
        'similar_players': similar_players,
        'year': year
    })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True) 