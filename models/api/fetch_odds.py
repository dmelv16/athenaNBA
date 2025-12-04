"""
NBA Odds Fetcher
Fetches and stores NBA odds from OddsPAPI v4

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

from models.api.odds_client import OddsPAPIClient, PlayerPropLine, GameLine
from models.api.odds_uploader import OddsUploader


class NBAOddsFetcher:
    """Fetches NBA odds and stores in database"""
    
    def __init__(self, api_key: str, bookmaker: str = 'pinnacle'):
        self.client = OddsPAPIClient(api_key, preferred_bookmaker=bookmaker)
        self.uploader = OddsUploader()
        self.bookmaker = bookmaker
        
        # Initialize reference data
        self._init_reference_data()
    
    def _init_reference_data(self):
        """Load reference data from API"""
        print("\n📚 Loading reference data...")
        
        # Load sports
        sports = self.client.get_sports()
        print(f"  Sports: {len(sports)}")
        
        # Load bookmakers
        bookmakers = self.client.get_bookmakers()
        if self.bookmaker not in bookmakers:
            print(f"  ⚠️ Preferred bookmaker '{self.bookmaker}' not found")
            print(f"  Available: {bookmakers[:10]}...")
        
        # Load markets
        markets = self.client.get_markets()
        print(f"  Markets: {len(markets)}")
        print(f"  Player prop markets: {len(self.client.PLAYER_PROP_MARKETS)}")
        
        # Load NBA participants
        participants = self.client.get_participants(sport_id=11)
        print(f"  Basketball participants: {len(participants)}")
    
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
        
        results = {
            'date': target_date,
            'fixtures_found': 0,
            'game_odds': 0,
            'player_props': 0,
            'errors': []
        }
        
        # Fetch NBA fixtures
        print(f"\n📅 Fetching NBA fixtures...")
        fixtures = self.client.get_nba_fixtures(
            from_date=target_date,
            to_date=target_date + timedelta(days=1),
            has_odds=True
        )
        
        results['fixtures_found'] = len(fixtures)
        
        if not fixtures:
            print("  ⚠️ No NBA fixtures found")
            self.uploader.log_snapshot(
                target_date, 0, 0, 0, self.bookmaker,
                'partial', 'No fixtures found'
            )
            return results
        
        print(f"  Found {len(fixtures)} NBA games")
        
        all_game_lines = []
        all_player_props = []
        
        for fixture in fixtures:
            fixture_id = fixture.get('fixtureId')
            p1_name = fixture.get('participant1Name', 'Team 1')
            p2_name = fixture.get('participant2Name', 'Team 2')
            
            print(f"\n  🏀 {p2_name} @ {p1_name}")
            print(f"     Fixture ID: {fixture_id}")
            print(f"     Start: {fixture.get('startTime', 'TBD')}")
            
            # Fetch game odds
            try:
                game_line = self.client.get_game_lines(fixture_id)
                if game_line:
                    # Override date from fixture
                    start = fixture.get('startTime')
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
                results['errors'].append(f"Game odds {fixture_id}: {e}")
                print(f"     ⚠️ Error fetching game odds: {e}")
            
            # Fetch player props
            try:
                props = self.client.get_player_props(fixture_id)
                
                if props:
                    all_player_props.extend(props)
                    
                    # Count by type
                    prop_counts = {}
                    for p in props:
                        prop_counts[p.prop_type] = prop_counts.get(p.prop_type, 0) + 1
                    
                    print(f"     Player props: {len(props)}")
                    for pt, cnt in sorted(prop_counts.items()):
                        print(f"       - {pt}: {cnt}")
                else:
                    print(f"     ⚠️ No player props found")
                    
            except Exception as e:
                results['errors'].append(f"Player props {fixture_id}: {e}")
                print(f"     ⚠️ Error fetching player props: {e}")
            
            # Respect rate limits
            time.sleep(0.5)
        
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
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Fixtures: {results['fixtures_found']}")
        print(f"Game odds uploaded: {results['game_odds']}")
        print(f"Player props uploaded: {results['player_props']}")
        if results['errors']:
            print(f"Errors: {len(results['errors'])}")
        
        return results
    
    def run_continuous(self, interval_minutes: int = 30):
        """
        Run continuous fetching for pregame odds updates
        """
        print(f"\n🔄 Starting continuous odds fetching")
        print(f"   Interval: {interval_minutes} minutes")
        print(f"   Press Ctrl+C to stop\n")
        
        while True:
            try:
                today = date.today()
                self.fetch_daily_odds(today)
                
                # Also fetch tomorrow if it's afternoon
                if datetime.now().hour >= 12:
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


def main():
    parser = argparse.ArgumentParser(description='NBA Odds Fetcher')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--bookmaker', type=str, default='pinnacle',
                       help='Bookmaker slug (pinnacle, bet365, etc.)')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous fetching')
    parser.add_argument('--interval', type=int, default=30,
                       help='Interval between fetches (minutes)')
    parser.add_argument('--init-schema', action='store_true',
                       help='Initialize database schema')
    parser.add_argument('--list-bookmakers', action='store_true',
                       help='List available bookmakers')
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.environ.get('ODDSPAPI_KEY')
    if not api_key:
        print("Error: ODDSPAPI_KEY environment variable not set")
        print("Set it with: export ODDSPAPI_KEY='your-api-key'")
        sys.exit(1)
    
    fetcher = NBAOddsFetcher(api_key, args.bookmaker)
    
    if args.init_schema:
        fetcher.uploader.init_schema()
    
    if args.list_bookmakers:
        print("\nAvailable bookmakers:")
        for bm in fetcher.client.get_bookmakers():
            print(f"  - {bm}")
        fetcher.close()
        return
    
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