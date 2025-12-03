"""
Incremental ETL Update - Fetch only missing games

This script:
1. Checks the latest game date in your database
2. Fetches all games from that date to yesterday
3. Only inserts games that don't already exist

Usage:
    python etl/incremental_update.py                    # Update current season
    python etl/incremental_update.py --season 2024-25  # Specific season
    python etl/incremental_update.py --days 7          # Last 7 days only
    python etl/incremental_update.py --full-season     # Full current season
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set, Optional, Tuple, Dict

import pandas as pd
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv2,
    playergamelog,
    teamgamelog,
    commonallplayers
)
from nba_api.stats.static import teams, players as static_players

etl_dir = Path(__file__).parent
sys.path.insert(0, str(etl_dir))
sys.path.insert(0, str(etl_dir.parent))

from etl.database.connection import get_db_connection
from etl.database.operations import DataLoader
from etl.config.settings import ETLConfig
from etl.utils.logger import get_logger

logger = get_logger(__name__)


class IncrementalUpdater:
    """Handles incremental updates to game logs"""
    
    def __init__(self):
        self.db = get_db_connection()
        self.loader = DataLoader()
        self.request_count = 0
    
    def get_existing_player_ids(self) -> Set[int]:
        """Get set of player IDs already in database"""
        query = "SELECT player_id FROM players"
        with self.db.get_cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
        return {r[0] for r in results}
    
    def fetch_current_season_players(self, season: str) -> List[Dict]:
        """Fetch all players who have played in the current season"""
        logger.info(f"Fetching players for season {season}...")
        try:
            self.rate_limit()
            all_players = commonallplayers.CommonAllPlayers(
                is_only_current_season=1,
                league_id='00',
                season=season
            )
            df = all_players.get_data_frames()[0]
            players_list = []
            for _, row in df.iterrows():
                players_list.append({
                    'id': row['PERSON_ID'],
                    'full_name': row.get('DISPLAY_FIRST_LAST', 'Unknown'),
                    'first_name': row.get('DISPLAY_FIRST_LAST', 'Unknown').split()[0] if ' ' in str(row.get('DISPLAY_FIRST_LAST', '')) else '',
                    'last_name': ' '.join(row.get('DISPLAY_FIRST_LAST', 'Unknown').split()[1:]) if ' ' in str(row.get('DISPLAY_FIRST_LAST', '')) else row.get('DISPLAY_FIRST_LAST', 'Unknown'),
                    'is_active': True
                })
            logger.info(f"Found {len(players_list)} players in {season}")
            return players_list
        except Exception as e:
            logger.error(f"Error fetching season players: {e}")
            return []
    
    def update_players_table(self, season: str) -> int:
        """Check for and add any new players"""
        logger.info("=" * 60)
        logger.info("CHECKING FOR NEW PLAYERS")
        logger.info("=" * 60)
        
        existing_ids = self.get_existing_player_ids()
        logger.info(f"Existing players in DB: {len(existing_ids)}")
        
        current_players = self.fetch_current_season_players(season)
        if not current_players:
            logger.info("Could not fetch current players, skipping...")
            return 0
        
        new_players = [p for p in current_players if p['id'] not in existing_ids]
        if not new_players:
            logger.info("No new players found")
            return 0
        
        logger.info(f"Found {len(new_players)} new players:")
        for p in new_players:
            logger.info(f"  + {p['full_name']} (ID: {p['id']})")
        
        try:
            self.loader.insert_players(new_players)
            logger.info(f"✓ Inserted {len(new_players)} new players")
            return len(new_players)
        except Exception as e:
            logger.error(f"✗ Error inserting new players: {e}")
            return 0
        
    def get_latest_game_date(self, table: str) -> Optional[datetime]:
        """Get the most recent game date in database"""
        query = f"SELECT MAX(game_date) FROM {table}"
        with self.db.get_cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()[0]
        if result:
            logger.info(f"Latest game in {table}: {result}")
            if hasattr(result, 'hour'):
                return result
            else:
                return datetime.combine(result, datetime.min.time())
        return None
    
    def get_existing_game_ids(self, table: str, season: str = None) -> Set[Tuple[str, int]]:
        """Get set of (game_id, team_id) pairs already in database for a season"""
        query = f"SELECT game_id, team_id FROM {table}"
        if season:
            query += f" WHERE season = '{season}'"
        
        with self.db.get_cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
        
        return {(str(r[0]), int(r[1])) for r in results}
    
    def get_existing_player_game_keys(self, season: str = None) -> Set[Tuple[str, int]]:
        """Get set of (game_id, player_id) already in database"""
        query = "SELECT game_id, player_id FROM player_game_logs"
        if season:
            query += f" WHERE season = '{season}'"
        
        with self.db.get_cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
        
        return {(str(r[0]), int(r[1])) for r in results}
    
    def rate_limit(self):
        """Apply rate limiting"""
        self.request_count += 1
        time.sleep(ETLConfig.REQUEST_DELAY)
        if self.request_count % ETLConfig.BATCH_SIZE == 0:
            logger.info(f"Processed {self.request_count} requests, pausing {ETLConfig.BATCH_DELAY}s...")
            time.sleep(ETLConfig.BATCH_DELAY)
    
    def get_season_string(self, date: datetime = None) -> str:
        """Get NBA season string for a date"""
        if date is None:
            date = datetime.now()
        year = date.year
        month = date.month
        if month >= 10:
            return f"{year}-{str(year + 1)[-2:]}"
        else:
            return f"{year - 1}-{str(year)[-2:]}"
    
    def fetch_all_season_games(self, season: str) -> pd.DataFrame:
        """Fetch ALL games for a season using LeagueGameFinder - returns both teams' perspectives"""
        logger.info(f"Fetching all games for season {season}...")
        
        try:
            self.rate_limit()
            game_finder = leaguegamefinder.LeagueGameFinder(
                league_id_nullable='00',
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            games_df = game_finder.get_data_frames()[0]
            
            if not games_df.empty:
                games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
                unique_games = games_df['GAME_ID'].nunique()
                logger.info(f"Found {len(games_df)} team-game records ({unique_games} unique games)")
            
            return games_df
        except Exception as e:
            logger.error(f"Error fetching games: {e}")
            return pd.DataFrame()
    
    def fetch_player_game_log(self, player_id: int, season: str, player_name: str = "Unknown") -> Optional[pd.DataFrame]:
        """Fetch player game log for a season"""
        try:
            self.rate_limit()
            game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            df = game_log.get_data_frames()[0]
            if not df.empty:
                df['season'] = season
                df.columns = df.columns.str.lower()
                logger.debug(f"  {player_name}: {len(df)} games")
            return df
        except Exception as e:
            logger.warning(f"  Error fetching {player_name}: {e}")
            return None
    
    def fetch_team_game_log(self, team_id: int, season: str, team_name: str = "Unknown") -> Optional[pd.DataFrame]:
        """Fetch team game log for a season"""
        try:
            self.rate_limit()
            game_log = teamgamelog.TeamGameLog(team_id=team_id, season=season)
            df = game_log.get_data_frames()[0]
            if not df.empty:
                df['season'] = season
                df.columns = df.columns.str.lower()
                logger.debug(f"  {team_name}: {len(df)} games")
            return df
        except Exception as e:
            logger.warning(f"  Error fetching {team_name}: {e}")
            return None
    
    def update_team_game_logs(self, season: str, since_date: datetime = None) -> int:
        """Update team game logs - ensures both teams' perspectives are captured"""
        logger.info("=" * 60)
        logger.info("UPDATING TEAM GAME LOGS")
        logger.info("=" * 60)
        
        # Get existing (game_id, team_id) pairs for this SEASON (not filtered by date)
        existing_games = self.get_existing_game_ids('team_game_logs', season=season)
        logger.info(f"Existing team-game records in DB for {season}: {len(existing_games)}")
        
        # Debug: show sample of existing keys
        if existing_games:
            sample = list(existing_games)[:3]
            logger.info(f"Sample existing keys: {sample}")
        
        today = pd.Timestamp(datetime.now().date())
        logger.info(f"Excluding games from {today.date()} onwards")
        
        all_teams = teams.get_teams()
        total_inserted = 0
        skipped_teams = 0
        
        for idx, team in enumerate(all_teams, 1):
            team_id = team['id']
            team_name = team['full_name']
            
            logger.info(f"[{idx}/{len(all_teams)}] {team_name}")
            
            df = self.fetch_team_game_log(team_id, season, team_name)
            
            if df is None or df.empty:
                logger.info(f"  No games found from API")
                continue
            
            # Convert and prepare data
            df['game_date'] = pd.to_datetime(df['game_date'], format='mixed', errors='coerce')
            df['game_id'] = df['game_id'].astype(str)
            df['team_id'] = team_id  # Ensure team_id is set
            
            # Create composite key for comparison
            df['_key'] = list(zip(df['game_id'], df['team_id']))
            
            # Debug: show sample API keys
            if len(df) > 0:
                logger.debug(f"  Sample API keys: {df['_key'].tolist()[:3]}")
            
            # Filter to only games NOT in database
            new_games = df[~df['_key'].isin(existing_games)].copy()
            new_games = new_games.drop(columns=['_key'])
            
            # Skip games not yet played (wl is null means game hasn't happened)
            new_games = new_games[new_games['wl'].notna()]
            
            # Exclude today's games and future games
            new_games = new_games[new_games['game_date'] < today]
            
            # Optionally filter by since_date
            if since_date:
                since_ts = pd.Timestamp(since_date)
                new_games = new_games[new_games['game_date'] >= since_ts]
            
            if new_games.empty:
                logger.info(f"  No new games to insert")
                skipped_teams += 1
                continue
            
            logger.info(f"  Found {len(new_games)} new games to insert")
            
            # Transform and load
            new_games = self._transform_team_game_log(new_games)
            
            try:
                self.loader.insert_dataframe(new_games, 'team_game_logs', ['game_id', 'team_id'])
                total_inserted += len(new_games)
                logger.info(f"  ✓ Inserted {len(new_games)} new games")
                
                # Add to existing set to prevent duplicates in same run
                for _, row in new_games.iterrows():
                    existing_games.add((str(row['game_id']), int(row['team_id'])))
                
            except Exception as e:
                logger.error(f"  ✗ Error inserting: {e}")
        
        logger.info(f"\nTotal team game logs inserted: {total_inserted}")
        logger.info(f"Teams with no new games: {skipped_teams}")
        return total_inserted
    
    def update_player_game_logs(self, season: str, since_date: datetime = None, player_ids: List[int] = None) -> int:
        """Update player game logs"""
        logger.info("=" * 60)
        logger.info("UPDATING PLAYER GAME LOGS")
        logger.info("=" * 60)
        
        # Get existing keys for this SEASON
        existing_keys = self.get_existing_player_game_keys(season=season)
        logger.info(f"Existing player-game records for {season}: {len(existing_keys)}")
        
        today = pd.Timestamp(datetime.now().date())
        logger.info(f"Excluding games from {today.date()} onwards")
        
        if player_ids:
            query = f"SELECT player_id, full_name FROM players WHERE player_id IN ({','.join(map(str, player_ids))})"
        else:
            query = "SELECT player_id, full_name FROM players WHERE is_active = true"
        
        with self.db.get_cursor() as cur:
            cur.execute(query)
            players = cur.fetchall()
        
        logger.info(f"Players to check: {len(players)}")
        
        total_inserted = 0
        
        for idx, (player_id, player_name) in enumerate(players, 1):
            logger.info(f"[{idx}/{len(players)}] {player_name}")
            
            df = self.fetch_player_game_log(player_id, season, player_name)
            
            if df is None or df.empty:
                continue
            
            df['game_date'] = pd.to_datetime(df['game_date'], format='mixed', errors='coerce')
            df['game_id'] = df['game_id'].astype(str)
            df['player_id'] = player_id
            
            df['_key'] = list(zip(df['game_id'], df['player_id']))
            new_games = df[~df['_key'].isin(existing_keys)].copy()
            new_games = new_games.drop(columns=['_key'])
            
            new_games = new_games[new_games['wl'].notna()]
            new_games = new_games[new_games['game_date'] < today]
            
            if since_date:
                since_ts = pd.Timestamp(since_date)
                new_games = new_games[new_games['game_date'] >= since_ts]
            
            if new_games.empty:
                logger.info(f"  No new games")
                continue
            
            new_games = self._transform_player_game_log(new_games)
            
            try:
                self.loader.insert_dataframe(new_games, 'player_game_logs', ['game_id', 'player_id'])
                total_inserted += len(new_games)
                logger.info(f"  ✓ Inserted {len(new_games)} new games")
                
                for _, row in new_games.iterrows():
                    existing_keys.add((str(row['game_id']), int(row['player_id'])))
                
            except Exception as e:
                logger.error(f"  ✗ Error inserting: {e}")
        
        logger.info(f"\nTotal player game logs inserted: {total_inserted}")
        return total_inserted
    
    def _transform_player_game_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform player game log for database"""
        column_map = {
            'game_id': 'game_id', 'player_id': 'player_id', 'season': 'season',
            'game_date': 'game_date', 'matchup': 'matchup', 'wl': 'wl', 'min': 'min',
            'fgm': 'fgm', 'fga': 'fga', 'fg_pct': 'fg_pct', 'fg3m': 'fg3m',
            'fg3a': 'fg3a', 'fg3_pct': 'fg3_pct', 'ftm': 'ftm', 'fta': 'fta',
            'ft_pct': 'ft_pct', 'oreb': 'oreb', 'dreb': 'dreb', 'reb': 'reb',
            'ast': 'ast', 'stl': 'stl', 'blk': 'blk', 'tov': 'tov', 'pf': 'pf',
            'pts': 'pts', 'plus_minus': 'plus_minus'
        }
        
        available = {k: v for k, v in column_map.items() if k in df.columns}
        result = df[list(available.keys())].copy()
        result.columns = list(available.values())
        result['game_date'] = pd.to_datetime(result['game_date'], format='mixed', errors='coerce')
        
        if 'min' in result.columns:
            result['min'] = result['min'].apply(self._convert_minutes)
        
        return result
    
    def _transform_team_game_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform team game log for database"""
        column_map = {
            'game_id': 'game_id', 'team_id': 'team_id', 'season': 'season',
            'game_date': 'game_date', 'matchup': 'matchup', 'wl': 'wl', 'min': 'min',
            'fgm': 'fgm', 'fga': 'fga', 'fg_pct': 'fg_pct', 'fg3m': 'fg3m',
            'fg3a': 'fg3a', 'fg3_pct': 'fg3_pct', 'ftm': 'ftm', 'fta': 'fta',
            'ft_pct': 'ft_pct', 'oreb': 'oreb', 'dreb': 'dreb', 'reb': 'reb',
            'ast': 'ast', 'stl': 'stl', 'blk': 'blk', 'tov': 'tov', 'pf': 'pf',
            'pts': 'pts', 'plus_minus': 'plus_minus'
        }
        
        available = {k: v for k, v in column_map.items() if k in df.columns}
        result = df[list(available.keys())].copy()
        result.columns = list(available.values())
        result['game_date'] = pd.to_datetime(result['game_date'], format='mixed', errors='coerce')
        
        return result
    
    def _convert_minutes(self, val):
        """Convert minutes from MM:SS to decimal"""
        if pd.isna(val):
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str) and ':' in val:
            try:
                parts = val.split(':')
                return float(parts[0]) + float(parts[1]) / 60
            except:
                return None
        try:
            return float(val)
        except:
            return None
    
    def run_incremental_update(self, season: str = None, days_back: int = None, full_season: bool = False):
        """Run incremental update"""
        logger.info("\n" + "=" * 60)
        logger.info("NBA INCREMENTAL UPDATE")
        logger.info("=" * 60)
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if season is None:
            season = self.get_season_string()
        logger.info(f"Season: {season}")
        
        yesterday = datetime.now().date() - timedelta(days=1)
        
        if days_back:
            since_date = datetime.now() - timedelta(days=days_back)
            logger.info(f"Looking back {days_back} days (since {since_date.date()})")
        elif full_season:
            season_start_year = int(season.split('-')[0])
            since_date = datetime(season_start_year, 10, 1)
            logger.info(f"Checking full season since {since_date.date()}")
        else:
            # For incremental: don't filter by date, filter by what's missing
            since_date = None
            logger.info(f"Checking for any missing games in {season}")
        
        self.db.connect()
        
        try:
            new_players_count = self.update_players_table(season)
            team_count = self.update_team_game_logs(season, since_date)
            player_count = self.update_player_game_logs(season, since_date)
            
            logger.info("\n" + "=" * 60)
            logger.info("UPDATE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"New players added: {new_players_count}")
            logger.info(f"Team game logs added: {team_count}")
            logger.info(f"Player game logs added: {player_count}")
            logger.info(f"Total API requests: {self.request_count}")
            logger.info("=" * 60)
            
        finally:
            self.db.close()


def main():
    parser = argparse.ArgumentParser(
        description='Incremental NBA Data Update',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python incremental_update.py                     # Smart update - finds missing games
  python incremental_update.py --season 2024-25   # Specific season
  python incremental_update.py --days 7           # Last 7 days only
  python incremental_update.py --full-season      # Check entire season for gaps
        """
    )
    
    parser.add_argument('--season', type=str, help='NBA season (e.g., 2024-25). Defaults to current season.')
    parser.add_argument('--days', type=int, help='Only look back this many days')
    parser.add_argument('--full-season', action='store_true', help='Check entire season for missing games')
    
    args = parser.parse_args()
    
    updater = IncrementalUpdater()
    updater.run_incremental_update(season=args.season, days_back=args.days, full_season=args.full_season)


if __name__ == "__main__":
    main()