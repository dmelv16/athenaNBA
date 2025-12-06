"""
NBA Odds Fetcher - Sports Game Odds API
Fetches and stores NBA odds from api.sportsgameodds.com/v2

Usage:
    # Fetch today's odds
    python -m models.api.fetch_odds
    
    # Fetch for specific date
    python -m models.api.fetch_odds --date 2025-12-01
    
    # Continuous fetching (for pregame updates)
    python -m models.api.fetch_odds --continuous --interval 30
"""

import argparse
import os
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.api.odds_client import SportsGameOddsClient, PlayerPropLine, GameLine
from models.api.odds_uploader import OddsUploader
from models.api.player_matcher import PlayerTeamMatcher


class NBAOddsFetcher:
    """Fetches NBA odds from Sports Game Odds API and stores in database"""
    
    def __init__(self, api_key: str, bookmaker: str = 'fanduel'):
        self.client = SportsGameOddsClient(api_key, preferred_bookmaker=bookmaker)
        self.uploader = OddsUploader()
        self.matcher = PlayerTeamMatcher()
        self.bookmaker = bookmaker
        
        # Try to load cached ID mappings
        cache_path = Path(__file__).parent / 'sgo_match_cache.json'
        self.matcher.load_match_cache(str(cache_path))
        
        # Initialize reference data
        self._init_reference_data()
    
    def _init_reference_data(self):
        """Load reference data - skip API calls since we have DB data"""
        print("\n📚 Loading reference data...")
        
        # We already have players/teams in our database via PlayerTeamMatcher
        # Skip loading from API to save rate-limited requests
        print(f"  Using {len(self.matcher._db_players)} players from database")
        print(f"  Using {len(self.matcher._db_teams)} teams from database")
        print(f"  ℹ️ Skipping API player/team fetch to conserve rate limits")
    
    def fetch_daily_odds(self, target_date: date = None) -> Dict:
        """
        Fetch all odds for a date
        
        Returns:
            Dict with counts of fetched data
        """
        target_date = target_date or date.today()
        snapshot_time = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"FETCHING NBA ODDS - {target_date}")
        print(f"{'='*60}")
        print(f"Bookmaker: {self.bookmaker}")
        print(f"Snapshot: {snapshot_time}")
        print(f"API: Sports Game Odds (sportsgameodds.com)")
        
        results = {
            'date': target_date,
            'fixtures_found': 0,
            'game_odds': 0,
            'player_props': 0,
            'errors': []
        }
        
        # Fetch NBA events for the date
        print(f"\n📅 Fetching NBA events...")
        events = self.client.get_nba_events(
            from_date=target_date,
            to_date=target_date + timedelta(days=1),
            include_odds=True
        )
        
        results['fixtures_found'] = len(events)
        
        if not events:
            print("  ⚠️ No NBA events found")
            self.uploader.log_snapshot(
                target_date, 0, 0, 0, self.bookmaker,
                'partial', 'No events found'
            )
            return results
        
        print(f"  Found {len(events)} NBA games")
        
        all_game_lines = []
        all_player_props = []
        
        for event in events:
            event_id = event.get('eventID')
            home_team = event.get('homeTeam', {})
            away_team = event.get('awayTeam', {})
            
            home_name = home_team.get('name', 'Home')
            away_name = away_team.get('name', 'Away')
            
            print(f"\n  🏀 {away_name} @ {home_name}")
            print(f"     Event ID: {event_id}")
            print(f"     Start: {event.get('startTime', 'TBD')}")
            
            # Fetch game odds
            try:
                game_line = self.client.get_game_lines(event_id)
                if game_line:
                    # Map team IDs to our database
                    home_match = self.matcher.match_team(
                        home_team.get('teamID', ''), 
                        home_name
                    )
                    away_match = self.matcher.match_team(
                        away_team.get('teamID', ''),
                        away_name
                    )
                    
                    # Update with matched IDs
                    if home_match.db_team_id:
                        game_line.home_team_id = home_match.db_team_id
                    if away_match.db_team_id:
                        game_line.away_team_id = away_match.db_team_id
                    
                    # Set abbreviations
                    game_line.home_team = home_match.db_team_abbrev or self.client.get_team_abbrev(home_name)
                    game_line.away_team = away_match.db_team_abbrev or self.client.get_team_abbrev(away_name)
                    
                    # Override date from event
                    start = event.get('startTime')
                    if start:
                        try:
                            game_line.game_date = datetime.fromisoformat(
                                start.replace('Z', '+00:00')
                            ).date()
                        except:
                            game_line.game_date = target_date
                    
                    all_game_lines.append(game_line)
                    
                    # Print summary
                    spread_str = f"Spread: {game_line.spread_line}" if game_line.spread_line else "No spread"
                    total_str = f"Total: {game_line.total_line}" if game_line.total_line else "No total"
                    ml_str = f"ML: {game_line.home_ml_american}/{game_line.away_ml_american}" if game_line.home_ml_american else "No ML"
                    print(f"     {spread_str} | {total_str} | {ml_str}")
                else:
                    print(f"     ⚠️ No game lines found")
                    
            except Exception as e:
                results['errors'].append(f"Game odds {event_id}: {e}")
                print(f"     ⚠️ Error fetching game odds: {e}")
            
            # Fetch player props
            try:
                props = self.client.get_player_props(event_id)
                
                if props:
                    # Match player IDs to our database
                    matched_props = []
                    for prop in props:
                        player_match = self.matcher.match_player(
                            prop.player_id or '',
                            prop.player_name
                        )
                        
                        if player_match.db_player_id:
                            # Update with matched ID
                            prop.player_id = player_match.db_player_id
                            prop.player_name = player_match.db_player_name or prop.player_name
                        
                        matched_props.append(prop)
                    
                    all_player_props.extend(matched_props)
                    
                    # Count by type
                    prop_counts = {}
                    for p in matched_props:
                        prop_counts[p.prop_type] = prop_counts.get(p.prop_type, 0) + 1
                    
                    print(f"     Player props: {len(matched_props)}")
                    for pt, cnt in sorted(prop_counts.items()):
                        print(f"       - {pt}: {cnt}")
                else:
                    print(f"     ⚠️ No player props found")
                    
            except Exception as e:
                results['errors'].append(f"Player props {event_id}: {e}")
                print(f"     ⚠️ Error fetching player props: {e}")
            
            # Rate limiting - SGO Amateur plan: 10 req/min = 6 sec between
            # We make 2 requests per event (game lines + props)
            time.sleep(3)
        
        # Upload to database
        print(f"\n💾 Uploading to database...")
        
        if all_game_lines:
            game_count = self.uploader.upload_game_odds(all_game_lines, snapshot_time)
            results['game_odds'] = game_count
            print(f"  ✓ Game odds: {game_count}")
        
        if all_player_props:
            prop_count = self.uploader.upload_player_props(
                all_player_props, target_date, snapshot_time
            )
            results['player_props'] = prop_count
            print(f"  ✓ Player props: {prop_count}")
        
        # Log snapshot
        status = 'success' if not results['errors'] else 'partial'
        error_msg = '; '.join(results['errors'][:5]) if results['errors'] else None
        
        self.uploader.log_snapshot(
            target_date,
            results['fixtures_found'],
            results['player_props'],
            results['game_odds'],
            self.bookmaker,
            status,
            error_msg
        )
        
        # Save match cache for future runs
        cache_path = Path(__file__).parent / 'sgo_match_cache.json'
        self.matcher.save_match_cache(str(cache_path))
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Events: {results['fixtures_found']}")
        print(f"Game odds uploaded: {results['game_odds']}")
        print(f"Player props uploaded: {results['player_props']}")
        if results['errors']:
            print(f"Errors: {len(results['errors'])}")
        
        return results
    
    def run_continuous(self, interval_minutes: int = 30):
        """
        Run continuous fetching for pregame odds updates
        
        Note: With Amateur plan (10 req/min, 1000 obj/month),
        be careful with continuous fetching!
        """
        print(f"\n🔄 Starting continuous odds fetching")
        print(f"   Interval: {interval_minutes} minutes")
        print(f"   ⚠️ Amateur plan: 1000 objects/month limit!")
        print(f"   Press Ctrl+C to stop\n")
        
        while True:
            try:
                today = date.today()
                self.client.request_count = 0  # Reset for tracking
                
                self.fetch_daily_odds(today)
                
                print(f"\n   API requests this cycle: {self.client.request_count}")
                
                # Also fetch tomorrow if afternoon (but careful with limits!)
                if datetime.now().hour >= 18:  # After 6 PM
                    tomorrow = today + timedelta(days=1)
                    self.fetch_daily_odds(tomorrow)
                
                print(f"\n⏰ Next fetch in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Stopping continuous fetch...")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
                print(f"Retrying in 5 minutes...")
                time.sleep(300)
    
    def close(self):
        self.client.close()
        self.uploader.close()
        self.matcher.close()


def main():
    parser = argparse.ArgumentParser(description='NBA Odds Fetcher (Sports Game Odds API)')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--bookmaker', type=str, default='fanduel',
                       help='Bookmaker ID (fanduel, draftkings, etc.)')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous fetching')
    parser.add_argument('--interval', type=int, default=60,
                       help='Interval between fetches (minutes) - default 60 for rate limits')
    parser.add_argument('--init-schema', action='store_true',
                       help='Initialize database schema')
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.environ.get('SGO_API_KEY') or os.environ.get('ODDSPAPI_KEY')
    if not api_key:
        print("Error: SGO_API_KEY or ODDSPAPI_KEY environment variable not set")
        print("Set it with: export SGO_API_KEY='your-api-key'")
        sys.exit(1)
    
    fetcher = NBAOddsFetcher(api_key, args.bookmaker)
    
    if args.init_schema:
        fetcher.uploader.init_schema()
    
    try:
        if args.continuous:
            fetcher.run_continuous(args.interval)
        else:
            target_date = None
            if args.date:
                target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
            
            fetcher.fetch_daily_odds(target_date)
    finally:
        fetcher.close()


if __name__ == '__main__':
    main()