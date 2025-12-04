"""
Backfill Historical NBA Odds
Backfills odds data using OddsPAPI v4 historical-odds endpoint

Usage:
    python -m models.api.backfill_odds --start 2024-10-21 --end 2024-12-02
    python -m models.api.backfill_odds --days 45
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


class OddsBackfiller:
    """Backfill historical odds data using OddsPAPI"""
    
    def __init__(self, api_key: str, bookmaker: str = 'pinnacle'):
        self.client = OddsPAPIClient(api_key, preferred_bookmaker=bookmaker)
        self.uploader = OddsUploader()
        self.bookmaker = bookmaker
        
        # Initialize reference data
        print("\n📚 Loading reference data...")
        self.client.get_sports()
        self.client.get_markets()
        self.client.get_participants(sport_id=11)
    
    def backfill_fixture(self, fixture: Dict, snapshot_time: datetime) -> Dict:
        """
        Backfill odds for a single fixture using historical endpoint
        """
        fixture_id = fixture.get('fixtureId')
        result = {
            'fixture_id': fixture_id,
            'game_lines': 0,
            'player_props': 0,
            'status': 'success'
        }
        
        try:
            # Get historical odds (5 second cooldown!)
            historical = self.client.get_historical_odds(
                fixture_id, 
                [self.bookmaker]
            )
            
            if not historical:
                result['status'] = 'no_data'
                return result
            
            # Parse game lines from historical data
            game_line = self._parse_historical_game_lines(
                historical, fixture, self.bookmaker
            )
            
            if game_line:
                self.uploader.upload_game_odds([game_line], snapshot_time)
                result['game_lines'] = 1
            
            # Parse player props from historical data
            props = self._parse_historical_player_props(
                historical, fixture_id, self.bookmaker
            )
            
            if props:
                game_date = fixture.get('startTime', '')[:10]
                if game_date:
                    self.uploader.upload_player_props(
                        props, 
                        datetime.strptime(game_date, '%Y-%m-%d').date(),
                        snapshot_time
                    )
                result['player_props'] = len(props)
            
        except Exception as e:
            result['status'] = f'error: {e}'
        
        return result
    
    def _parse_historical_game_lines(
        self, 
        historical: Dict, 
        fixture: Dict, 
        bookmaker: str
    ) -> Optional[GameLine]:
        """Parse game lines from historical odds response"""
        try:
            game_line = GameLine(
                game_id=historical.get('fixtureId', ''),
                game_date=date.today(),
                home_team=fixture.get('participant1Name', ''),
                away_team=fixture.get('participant2Name', ''),
                home_team_id=fixture.get('participant1Id'),
                away_team_id=fixture.get('participant2Id'),
                bookmaker=bookmaker
            )
            
            # Parse start time
            start = fixture.get('startTime', '')
            if start:
                try:
                    game_line.game_date = datetime.fromisoformat(
                        start.replace('Z', '+00:00')
                    ).date()
                except:
                    pass
            
            # Historical data structure: bookmakers -> markets -> outcomes -> players -> [history]
            bm_data = historical.get('bookmakers', {}).get(bookmaker, {})
            markets = bm_data.get('markets', {})
            
            for market_id_str, market_data in markets.items():
                market_id = int(market_id_str)
                market_info = self.client._markets_cache.get(market_id, {})
                market_type = market_info.get('marketType', '')
                
                outcomes = market_data.get('outcomes', {})
                
                for oid_str, outcome_data in outcomes.items():
                    players = outcome_data.get('players', {})
                    
                    for pid_str, history_list in players.items():
                        if not history_list:
                            continue
                        
                        # Get most recent active entry
                        latest = None
                        for entry in history_list:
                            if entry.get('active', False):
                                latest = entry
                                break
                        
                        if not latest:
                            latest = history_list[0]
                        
                        price = latest.get('price')
                        if not price:
                            continue
                        
                        # Determine line type from market
                        if market_type == '1x2' or 'moneyline' in str(market_id).lower():
                            # This is simplified - would need outcome mapping
                            pass
                        elif 'spread' in market_type:
                            handicap = market_info.get('handicap', 0)
                            game_line.spread_line = handicap
                            game_line.spread_home_odds = price
                            game_line.spread_home_american = self.client.decimal_to_american(price)
                        elif market_type == 'totals':
                            handicap = market_info.get('handicap', 0)
                            game_line.total_line = handicap
                            game_line.over_odds = price
                            game_line.over_american = self.client.decimal_to_american(price)
            
            game_line.last_update = datetime.now()
            return game_line
            
        except Exception as e:
            print(f"Error parsing historical game lines: {e}")
            return None
    
    def _parse_historical_player_props(
        self,
        historical: Dict,
        fixture_id: str,
        bookmaker: str
    ) -> List[PlayerPropLine]:
        """Parse player props from historical odds response"""
        props = []
        
        try:
            bm_data = historical.get('bookmakers', {}).get(bookmaker, {})
            markets = bm_data.get('markets', {})
            
            for market_id_str, market_data in markets.items():
                market_id = int(market_id_str)
                market_info = self.client._markets_cache.get(market_id, {})
                
                if not market_info.get('playerProp', False):
                    continue
                
                prop_type = self.client.PLAYER_PROP_MARKETS.get(market_id)
                if not prop_type:
                    continue
                
                outcomes = market_data.get('outcomes', {})
                
                for oid_str, outcome_data in outcomes.items():
                    players = outcome_data.get('players', {})
                    
                    for player_id_str, history_list in players.items():
                        player_id = int(player_id_str)
                        if player_id == 0 or not history_list:
                            continue
                        
                        # Get most recent entry
                        latest = history_list[0]
                        for entry in history_list:
                            if entry.get('active', False):
                                latest = entry
                                break
                        
                        price = latest.get('price')
                        if not price:
                            continue
                        
                        player_name = self.client._participants_cache.get(
                            player_id, f"Player {player_id}"
                        )
                        
                        # Get line from market handicap
                        line = market_info.get('handicap', 0)
                        
                        props.append(PlayerPropLine(
                            player_name=player_name,
                            player_id=player_id,
                            prop_type=prop_type,
                            line=line,
                            over_odds=price,
                            under_odds=price,  # May need to find paired under
                            over_odds_american=self.client.decimal_to_american(price),
                            under_odds_american=self.client.decimal_to_american(price),
                            bookmaker=bookmaker,
                            last_update=datetime.now(),
                            game_id=fixture_id,
                            market_id=market_id
                        ))
        
        except Exception as e:
            print(f"Error parsing historical player props: {e}")
        
        return props
    
    def backfill_date(self, target_date: date) -> Dict:
        """Backfill odds for a single date"""
        results = {
            'date': target_date,
            'fixtures': 0,
            'game_odds': 0,
            'player_props': 0,
            'status': 'success'
        }
        
        # Get fixtures for this date
        fixtures = self.client.get_nba_fixtures(
            from_date=target_date,
            to_date=target_date + timedelta(days=1),
            has_odds=True
        )
        
        results['fixtures'] = len(fixtures)
        
        if not fixtures:
            results['status'] = 'no_fixtures'
            return results
        
        # Snapshot time - use game day evening
        snapshot_time = datetime.combine(target_date, datetime.min.time())
        snapshot_time = snapshot_time.replace(hour=18)
        
        for fixture in fixtures:
            p1 = fixture.get('participant1Name', '?')
            p2 = fixture.get('participant2Name', '?')
            
            fix_result = self.backfill_fixture(fixture, snapshot_time)
            
            results['game_odds'] += fix_result['game_lines']
            results['player_props'] += fix_result['player_props']
            
            status = "✓" if fix_result['status'] == 'success' else "⚠️"
            print(f"    {status} {p2} @ {p1}: {fix_result['game_lines']} game, {fix_result['player_props']} props")
            
            # 5 second cooldown for historical endpoint!
            time.sleep(5.5)
        
        return results
    
    def backfill_range(
        self,
        start_date: date,
        end_date: date,
        skip_existing: bool = True
    ) -> Dict:
        """Backfill odds for a date range"""
        print(f"\n{'='*60}")
        print("NBA ODDS BACKFILL")
        print(f"{'='*60}")
        print(f"Range: {start_date} to {end_date}")
        print(f"Bookmaker: {self.bookmaker}")
        print(f"Skip existing: {skip_existing}")
        print(f"\n⚠️  Note: Historical endpoint has 5 second cooldown per request")
        
        total_days = (end_date - start_date).days
        print(f"Total days: {total_days}")
        
        summary = {
            'dates_processed': 0,
            'dates_skipped': 0,
            'total_fixtures': 0,
            'total_game_odds': 0,
            'total_player_props': 0
        }
        
        current = start_date
        while current < end_date:
            day_num = summary['dates_processed'] + summary['dates_skipped'] + 1
            print(f"\n[{day_num}/{total_days}] {current}")
            
            # Check existing
            if skip_existing:
                existing = self.uploader.get_game_odds(current, self.bookmaker)
                if existing:
                    print(f"  ⏭️ Already has {len(existing)} games, skipping")
                    summary['dates_skipped'] += 1
                    current += timedelta(days=1)
                    continue
            
            # Backfill
            results = self.backfill_date(current)
            
            if results['fixtures'] > 0:
                summary['dates_processed'] += 1
                summary['total_fixtures'] += results['fixtures']
                summary['total_game_odds'] += results['game_odds']
                summary['total_player_props'] += results['player_props']
                print(f"  ✓ {results['fixtures']} games, {results['game_odds']} lines, {results['player_props']} props")
            else:
                summary['dates_skipped'] += 1
                print(f"  ⚠️ No fixtures found")
            
            current += timedelta(days=1)
        
        print(f"\n{'='*60}")
        print("BACKFILL COMPLETE")
        print(f"{'='*60}")
        print(f"Processed: {summary['dates_processed']}")
        print(f"Skipped: {summary['dates_skipped']}")
        print(f"Total fixtures: {summary['total_fixtures']}")
        print(f"Total game odds: {summary['total_game_odds']}")
        print(f"Total player props: {summary['total_player_props']}")
        
        return summary
    
    def close(self):
        self.client.close()
        self.uploader.close()


def main():
    parser = argparse.ArgumentParser(description='Backfill NBA Odds')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, help='Number of days to backfill from today')
    parser.add_argument('--bookmaker', type=str, default='pinnacle')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing')
    parser.add_argument('--init-schema', action='store_true')
    
    args = parser.parse_args()
    
    api_key = os.getenv('ODDSPAPI_KEY')
    if not api_key:
        print("Error: ODDSPAPI_KEY not found")
        print("Add ODDSPAPI_KEY=your-key to your .env file")
        sys.exit(1)
    
    backfiller = OddsBackfiller(api_key, args.bookmaker)
    
    if args.init_schema:
        backfiller.uploader.init_schema()
    
    try:
        if args.days:
            end_date = date.today()
            start_date = end_date - timedelta(days=args.days)
        elif args.start:
            start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
            end_date = datetime.strptime(args.end, '%Y-%m-%d').date() if args.end else date.today()
        else:
            # Default: Oct 21 2024 to Dec 2 2024
            start_date = date(2024, 10, 21)
            end_date = date(2024, 12, 3)
        
        backfiller.backfill_range(start_date, end_date, not args.overwrite)
    finally:
        backfiller.close()


if __name__ == '__main__':
    main()