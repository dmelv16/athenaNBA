"""
SMART NBA Player Filter - Get only players who played 2014-2025
Uses league-wide endpoints instead of checking each player individually
"""

from nba_api.stats.endpoints import commonallplayers
from nba_api.stats.static import players as static_players
import pandas as pd
import time


def get_players_by_season_fast():
    """
    Use the commonallplayers endpoint to get all players who played in each season
    This is MUCH faster than checking each player individually
    """
    print("=" * 70)
    print("SMART NBA PLAYER FILTER")
    print("Using league-wide endpoints (fast method)")
    print("=" * 70)
    
    seasons = [
        '2014-15', '2015-16', '2016-17', '2017-18', '2018-19',
        '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25'
    ]
    
    all_modern_players = set()
    player_details = {}
    
    print(f"\nFetching players for {len(seasons)} seasons...")
    print("This will take about 1-2 minutes...\n")
    
    for idx, season in enumerate(seasons, 1):
        try:
            print(f"  [{idx}/{len(seasons)}] Fetching {season}...", end='')
            
            # Get all players who played this season
            all_players = commonallplayers.CommonAllPlayers(
                is_only_current_season=0,
                league_id='00',
                season=season
            )
            
            df = all_players.get_data_frames()[0]
            
            # Extract player IDs and details
            for _, row in df.iterrows():
                player_id = row['PERSON_ID']
                all_modern_players.add(player_id)
                
                # Store player details
                if player_id not in player_details:
                    player_details[player_id] = {
                        'id': player_id,
                        'full_name': row.get('DISPLAY_FIRST_LAST', 'Unknown'),
                        'is_active': row.get('ROSTERSTATUS', 0) == 1,
                        'first_season': season,
                        'team': row.get('TEAM_ABBREVIATION', '')
                    }
                
                # Update last season and active status
                player_details[player_id]['last_season'] = season
                if row.get('ROSTERSTATUS', 0) == 1:
                    player_details[player_id]['is_active'] = True
            
            print(f" {len(df)} players")
            time.sleep(0.6)  # Rate limiting
            
        except Exception as e:
            print(f" Error: {e}")
            continue
    
    print(f"\n✓ Complete!")
    print(f"\nResults:")
    print(f"  Total unique players (2014-2025): {len(all_modern_players)}")
    
    # Convert to list of dicts
    players_list = list(player_details.values())
    
    # Count active vs inactive
    active_count = sum(1 for p in players_list if p.get('is_active', False))
    print(f"  Active players: {active_count}")
    print(f"  Retired/inactive: {len(players_list) - active_count}")
    
    return players_list


def save_results(players_list):
    """Save filtered players to files"""
    
    # Sort by last season (most recent first)
    players_list.sort(key=lambda x: x.get('last_season', ''), reverse=True)
    
    # Save player IDs for pipeline
    ids_str = ','.join(str(p['id']) for p in players_list)
    with open('modern_players_ids.txt', 'w') as f:
        f.write(ids_str)
    
    # Save active players only
    active_players = [p for p in players_list if p.get('is_active', False)]
    active_ids = ','.join(str(p['id']) for p in active_players)
    with open('active_players_ids.txt', 'w') as f:
        f.write(active_ids)
    
    # Save as CSV for viewing
    with open('modern_players_list.csv', 'w', encoding='utf-8') as f:
        f.write('player_id,full_name,is_active,first_season,last_season,team\n')
        for p in players_list:
            f.write(f"{p['id']},{p['full_name']},{p.get('is_active', False)},"
                   f"{p.get('first_season', '')},"
                   f"{p.get('last_season', '')},"
                   f"{p.get('team', '')}\n")
    
    # Save as CSV for active only
    with open('active_players_list.csv', 'w', encoding='utf-8') as f:
        f.write('player_id,full_name,team\n')
        for p in active_players:
            f.write(f"{p['id']},{p['full_name']},{p.get('team', '')}\n")
    
    print(f"\n✓ Files saved:")
    print(f"  modern_players_ids.txt - All players 2014-2025 ({len(players_list)} players)")
    print(f"  active_players_ids.txt - Active players only ({len(active_players)} players)")
    print(f"  modern_players_list.csv - Full list with details")
    print(f"  active_players_list.csv - Active players list")
    
    print(f"\n⏱️  Time estimates at 25 players/hour:")
    print(f"  All modern players: {len(players_list) / 25:.1f} hours ({len(players_list) / 25 / 24:.1f} days)")
    print(f"  Active players only: {len(active_players) / 25:.1f} hours")
    
    print(f"\n🚀 To run pipeline:")
    print(f"  # All modern players:")
    print(f"  $ids = Get-Content modern_players_ids.txt")
    print(f"  python main.py --mode full --players $ids --seasons 2024-25")
    print(f"")
    print(f"  # Active players only (recommended):")
    print(f"  $ids = Get-Content active_players_ids.txt")
    print(f"  python main.py --mode full --players $ids --seasons 2024-25")


def get_active_players_instant():
    """
    Ultra-fast: Get only currently active players (no API calls needed)
    Uses the static players list
    """
    print("=" * 70)
    print("INSTANT FILTER - Active Players Only")
    print("=" * 70)
    
    all_players = static_players.get_players()
    active = [p for p in all_players if p.get('is_active', False)]
    
    print(f"\nActive players: {len(active)}")
    
    # Save IDs
    ids_str = ','.join(str(p['id']) for p in active)
    with open('active_instant_ids.txt', 'w') as f:
        f.write(ids_str)
    
    # Save CSV
    with open('active_instant_list.csv', 'w', encoding='utf-8') as f:
        f.write('player_id,full_name\n')
        for p in active:
            f.write(f"{p['id']},{p['full_name']}\n")
    
    print(f"\n✓ Files saved:")
    print(f"  active_instant_ids.txt")
    print(f"  active_instant_list.csv")
    
    print(f"\n⏱️  Time estimate: {len(active) / 25:.1f} hours")
    
    print(f"\n🚀 To run pipeline:")
    print(f"  $ids = Get-Content active_instant_ids.txt")
    print(f"  python main.py --mode full --players $ids --seasons 2024-25")
    
    return active


def main():
    import sys
    
    print("\n🏀 NBA Smart Player Filter\n")
    print("Choose filtering method:\n")
    print("  1. INSTANT - Active players only (0 seconds, ~450 players)")
    print("     └─ Best for: Current season props betting")
    print("")
    print("  2. SMART - All players 2014-2025 (1-2 minutes, ~800-1000 players)")
    print("     └─ Best for: Historical analysis and ML models")
    print("")
    print("  3. BOTH - Get both lists")
    print("")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\n✓ Getting active players (instant)...\n")
        get_active_players_instant()
        print("\n✅ Done!")
        
    elif choice == '2':
        print("\n✓ Getting all modern players (2014-2025)...\n")
        modern_players = get_players_by_season_fast()
        save_results(modern_players)
        print("\n✅ Done!")
        
    elif choice == '3':
        print("\n✓ Getting both lists...\n")
        print("\n1/2: Active players...")
        get_active_players_instant()
        print("\n2/2: Modern players (2014-2025)...")
        modern_players = get_players_by_season_fast()
        save_results(modern_players)
        print("\n✅ Done!")
        
    else:
        print("Invalid choice.")
        return
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Review the CSV files to see the players")
    print("  2. Choose which list to use (active or modern)")
    print("  3. Run the ETL pipeline with the filtered players")
    print("=" * 70)


if __name__ == "__main__":
    main()