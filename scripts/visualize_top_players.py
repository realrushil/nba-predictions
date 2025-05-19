import json
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from pathlib import Path

def load_predictions(file_path):
    """Load prediction data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_player_stats(season="2021"):
    """Load player stats to get minutes played data"""
    file_path = f"data/normalized/totals_{season}.json"
    with open(file_path, 'r') as f:
        return json.load(f)

def filter_by_minutes(predictions, player_stats, min_minutes=100):
    """Filter predictions to only include players with minimum minutes played"""
    # Create a dictionary mapping player names to their stats
    player_dict = {}
    for player in player_stats:
        name = player.get('Name', '')
        if name:
            # Use both name and ID for more accurate matching
            player_id = player.get('RowId', '')
            key = f"{name}_{player_id}"
            player_dict[name] = player  # Keep a name-only entry for fallback
            player_dict[key] = player   # More specific entry with ID
    
    # Filter predictions based on minutes threshold
    filtered_preds = []
    excluded_count = 0
    skipped_players = []
    
    for pred in predictions:
        player_name = pred['player_name']
        player_data = player_dict.get(player_name, None)
        
        if player_data and 'Minutes' in player_data:
            minutes = player_data.get('Minutes', 0)
            
            if minutes >= min_minutes:
                # Add minutes to prediction data
                pred_with_minutes = pred.copy()
                pred_with_minutes['minutes'] = minutes
                filtered_preds.append(pred_with_minutes)
            else:
                excluded_count += 1
                skipped_players.append((player_name, minutes))
        else:
            # Player not found or missing minutes
            excluded_count += 1
            skipped_players.append((player_name, "not found"))
    
    print(f"Filtered out {excluded_count} players with less than {min_minutes} minutes played")
    print(f"Remaining players: {len(filtered_preds)}")
    
    # Print some examples of excluded players for debugging
    print("Examples of excluded players:")
    for player, mins in sorted(skipped_players[:10]):
        print(f"  {player}: {mins}")
    
    return filtered_preds

def create_top_players_chart(predictions, category, n=15, title=None, output_path=None):
    """
    Create a horizontal bar chart of the top n players in a specific category
    
    Parameters:
    - predictions: list of player prediction dictionaries
    - category: 'offensive_raptor', 'defensive_raptor', or 'total_raptor'
    - n: number of top players to display
    - title: chart title
    - output_path: path to save the visualization
    """
    # Sort predictions by the category in descending order
    sorted_players = sorted(predictions, key=lambda x: x[category], reverse=True)
    top_players = sorted_players[:n]
    
    # Create a DataFrame for easy plotting
    df = pd.DataFrame(top_players)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(top_players)), df[category], color='#1f77b4')
    
    # Add player names as y-tick labels
    plt.yticks(range(len(top_players)), df['player_name'])
    
    # Add values at the end of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{df[category].iloc[i]:.2f}', va='center')
    
    # Add title and labels
    category_name = category.replace('_', ' ').title()
    plt.title(title or f'Top {n} Players by {category_name} (2021-2022 Season)')
    plt.xlabel(category_name)
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Chart saved to {output_path}")
        
    return plt.gcf()  # Return the figure for display

def create_top_bottom_chart(predictions, category, n=10, title=None, output_path=None):
    """
    Create a chart showing both top and bottom players in a category
    
    Parameters:
    - predictions: list of player prediction dictionaries
    - category: 'offensive_raptor', 'defensive_raptor', or 'total_raptor'
    - n: number of top/bottom players to display
    - title: chart title
    - output_path: path to save the visualization
    """
    # Sort predictions by the category
    sorted_players = sorted(predictions, key=lambda x: x[category], reverse=True)
    top_players = sorted_players[:n]
    bottom_players = sorted_players[-n:][::-1]  # Reverse to show worst at bottom
    
    # Create DataFrames
    top_df = pd.DataFrame(top_players)
    bottom_df = pd.DataFrame(bottom_players)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot top players
    bars1 = ax1.barh(range(len(top_players)), top_df[category], color='#2ca02c')
    ax1.set_yticks(range(len(top_players)))
    ax1.set_yticklabels(top_df['player_name'])
    
    # Add values for top players
    for i, bar in enumerate(bars1):
        ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{top_df[category].iloc[i]:.2f}', va='center')
    
    # Plot bottom players
    bars2 = ax2.barh(range(len(bottom_players)), bottom_df[category], color='#d62728')
    ax2.set_yticks(range(len(bottom_players)))
    ax2.set_yticklabels(bottom_df['player_name'])
    
    # Add values for bottom players
    for i, bar in enumerate(bars2):
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{bottom_df[category].iloc[i]:.2f}', va='center')
    
    # Add titles and labels
    category_name = category.replace('_', ' ').title()
    fig.suptitle(title or f'Top and Bottom {n} Players by {category_name} (2021-2022 Season)')
    ax1.set_title(f'Top {n} Players')
    ax2.set_title(f'Bottom {n} Players')
    ax1.set_xlabel(category_name)
    ax2.set_xlabel(category_name)
    
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Chart saved to {output_path}")
        
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize top RAPTOR players from predictions')
    parser.add_argument('--input', type=str, default='models/predictions_2022.json',
                        help='Path to predictions JSON file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualization images')
    parser.add_argument('--top_n', type=int, default=15, 
                        help='Number of top players to display')
    parser.add_argument('--min_minutes', type=int, default=100,
                        help='Minimum minutes played filter (default: 100)')
    parser.add_argument('--season', type=str, default='2021',
                        help='Season to use for player stats (default: 2021 for 2021-2022 season)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load predictions
    predictions = load_predictions(args.input)
    print(f"Loaded predictions for {len(predictions)} players")
    
    # Load player stats to get minutes data
    player_stats = load_player_stats(args.season)
    print(f"Loaded stats for {len(player_stats)} players from season {args.season}")
    
    # Filter predictions by minutes played
    if args.min_minutes > 0:
        filtered_predictions = filter_by_minutes(predictions, player_stats, args.min_minutes)
        print(f"Applying {args.min_minutes} minutes filter: {len(filtered_predictions)} players remain")
    else:
        filtered_predictions = predictions
    
    # Create visualizations with filtered data
    print("Creating visualizations...")
    
    # Top offensive players
    create_top_players_chart(
        filtered_predictions, 
        'offensive_raptor', 
        n=args.top_n,
        title=f'Top {args.top_n} Players by Offensive RAPTOR (Min. {args.min_minutes} Minutes)',
        output_path=output_dir / f'top_offensive_raptor_{args.min_minutes}min.png'
    )
    
    # Top defensive players
    create_top_players_chart(
        filtered_predictions, 
        'defensive_raptor', 
        n=args.top_n,
        title=f'Top {args.top_n} Players by Defensive RAPTOR (Min. {args.min_minutes} Minutes)',
        output_path=output_dir / f'top_defensive_raptor_{args.min_minutes}min.png'
    )
    
    # Top overall players
    create_top_players_chart(
        filtered_predictions, 
        'total_raptor', 
        n=args.top_n,
        title=f'Top {args.top_n} Players by Total RAPTOR (Min. {args.min_minutes} Minutes)',
        output_path=output_dir / f'top_total_raptor_{args.min_minutes}min.png'
    )
    
    # Top-bottom offensive comparison
    create_top_bottom_chart(
        filtered_predictions,
        'offensive_raptor',
        n=10,
        title=f'Top and Bottom Players by Offensive RAPTOR (Min. {args.min_minutes} Minutes)',
        output_path=output_dir / f'top_bottom_offensive_raptor_{args.min_minutes}min.png'
    )
    
    # Top-bottom defensive comparison
    create_top_bottom_chart(
        filtered_predictions,
        'defensive_raptor',
        n=10,
        title=f'Top and Bottom Players by Defensive RAPTOR (Min. {args.min_minutes} Minutes)',
        output_path=output_dir / f'top_bottom_defensive_raptor_{args.min_minutes}min.png'
    )
    
    print(f"Visualizations saved to the {args.output_dir} directory")

if __name__ == "__main__":
    main() 