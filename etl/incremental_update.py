"""
Incremental ETL Update - Fetch only missing games

This script:
1. Fetches current rosters for all teams
2. Checks which games are missing for each team
3. Only fetches player data for roster players missing those games

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
    playergamelog,
    teamgamelog,
    commonallplayers,
    commonteamroster
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
        
        # Cache for rosters and missing games
        self.team_rosters: Dict[int, List[Dict]] = {}
        self.team_missing_games: Dict[int, Set[str]] = {}
    
    # =========================================================================
    # ROSTER FETCHING
    # =========================================================================
    
    def fetch_team_roster(self, team_id: int, season: str) -> List[Dict]:
        """Fetch current roster for a team"""
        if team_id in self.team_rosters:
            return self.team_rosters[team_id]
        
        try:
            self.rate_limit()
            roster = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season
            )
            roster_df = roster.get_data_frames()[0]
            
            players = []
            for _, row in roster_df.iterrows():
                players.append({
                    'player_id': row['PLAYER_ID'],
                    'player_name': row['PLAYER'],
                    'position': row.get('POSITION', ''),
                    'number': row.get('NUM', '')
                })
            
            self.team_rosters[team_id] = players
            return players
            
        except Exception as e:
            logger.warning(f"Error fetching roster for team {team_id}: {e}")
            return []
    
    def fetch_all_rosters(self, season: str) -> Dict[int, List[Dict]]:
        """Fetch rosters for all 30 teams"""
        logger.info("=" * 60)
        logger.info("FETCHING CURRENT ROSTERS")
        logger.info("=" * 60)
        
        all_teams = teams.get_teams()
        total_players = 0
        
        for idx, team in enumerate(all_teams, 1):
            team_id = team['id']
            team_name = team['full_name']
            
            roster = self.fetch_team_roster(team_id, season)
            total_players += len(roster)
            
            logger.info(f"[{idx}/{len(all_teams)}] {team_name}: {len(roster)} players")
        
        logger.info(f"\nTotal roster players: {total_players}")
        return self.team_rosters
    
    # =========================================================================
    # PLAYER MANAGEMENT
    # =========================================================================
    
    def get_existing_player_ids(self) -> Set[int]:
        """Get set of player IDs already in database"""
        query = "SELECT player_id FROM players"
        with self.db.get_cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
        return {r[0] for r in results}
    
    def update_players_from_rosters(self) -> int:
        """Add any roster players not in our database"""
        logger.info("=" * 60)
        logger.info("CHECKING FOR NEW PLAYERS FROM ROSTERS")
        logger.info("=" * 60)
        
        existing_ids = self.get_existing_player_ids()
        logger.info(f"Existing players in DB: {len(existing_ids)}")
        
        # Collect all roster players
        new_players = []
        for team_id, roster in self.team_rosters.items():
            for player in roster:
                if player['player_id'] not in existing_ids:
                    # Check if we already added this player
                    if not any(p['id'] == player['player_id'] for p in new_players):
                        name_parts = player['player_name'].split(' ', 1)
                        new_players.append({
                            'id': player['player_id'],
                            'full_name': player['player_name'],
                            'first_name': name_parts[0] if len(name_parts) > 0 else '',
                            'last_name': name_parts[1] if len(name_parts) > 1 else '',
                            'is_active': True
                        })
        
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
    
    # =========================================================================
    # GAME LOG CHECKING
    # =========================================================================
    
    def get_existing_game_ids(self, table: str, season: str = None) -> Set[Tuple[str, int]]:
        """Get set of (game_id, team_id) pairs already in database for a season"""
        query = f"SELECT game_id, team_id FROM {table}"
        if season:
            query += f" WHERE season = '{season}'"
        
        with self.db.get_cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
        
        return {(str(r[0]), int(r[1])) for r in results}
    
    def get_existing_player_games(self, season: str, player_ids: List[int] = None) -> Dict[int, Set[str]]:
        """
        Get existing game_ids for players
        Returns: {player_id: {game_id1, game_id2, ...}}
        """
        query = "SELECT player_id, game_id FROM player_game_logs WHERE season = %s"
        
        if player_ids:
            placeholders = ','.join(['%s'] * len(player_ids))
            query += f" AND player_id IN ({placeholders})"
            params = [season] + player_ids
        else:
            params = [season]
        
        with self.db.get_cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        
        player_games: Dict[int, Set[str]] = {}
        for player_id, game_id in results:
            if player_id not in player_games:
                player_games[player_id] = set()
            player_games[player_id].add(str(game_id))
        
        return player_games
    
    def get_team_games_from_db(self, team_id: int, season: str, since_date: datetime = None) -> Set[str]:
        """Get all game_ids for a team from team_game_logs, optionally filtered by date"""
        query = """
            SELECT game_id FROM team_game_logs 
            WHERE team_id = %s AND season = %s
        """
        params = [team_id, season]
        
        if since_date:
            query += " AND game_date >= %s"
            params.append(since_date)
        
        with self.db.get_cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        return {str(r[0]) for r in results}
    
    # =========================================================================
    # TEAM GAME LOGS
    # =========================================================================
    
    def fetch_team_game_log(self, team_id: int, season: str, team_name: str = "Unknown") -> Optional[pd.DataFrame]:
        """Fetch team game log for a season"""
        try:
            self.rate_limit()
            game_log = teamgamelog.TeamGameLog(team_id=team_id, season=season)
            df = game_log.get_data_frames()[0]
            if not df.empty:
                df['season'] = season
                df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            logger.warning(f"  Error fetching {team_name}: {e}")
            return None
    
    def update_team_game_logs(self, season: str, since_date: datetime = None) -> int:
        """Update team game logs - ensures both teams' perspectives are captured"""
        logger.info("=" * 60)
        logger.info("UPDATING TEAM GAME LOGS")
        logger.info("=" * 60)
        
        existing_games = self.get_existing_game_ids('team_game_logs', season=season)
        logger.info(f"Existing team-game records in DB for {season}: {len(existing_games)}")
        
        today = pd.Timestamp(datetime.now().date())
        logger.info(f"Excluding games from {today.date()} onwards")
        
        all_teams = teams.get_teams()
        total_inserted = 0
        
        for idx, team in enumerate(all_teams, 1):
            team_id = team['id']
            team_name = team['full_name']
            
            logger.info(f"[{idx}/{len(all_teams)}] {team_name}")
            
            df = self.fetch_team_game_log(team_id, season, team_name)
            
            if df is None or df.empty:
                logger.info(f"  No games found from API")
                continue
            
            # Prepare data
            df['game_date'] = pd.to_datetime(df['game_date'], format='mixed', errors='coerce')
            df['game_id'] = df['game_id'].astype(str)
            df['team_id'] = team_id
            
            # Filter to new games only
            df['_key'] = list(zip(df['game_id'], df['team_id']))
            new_games = df[~df['_key'].isin(existing_games)].copy()
            new_games = new_games.drop(columns=['_key'])
            
            # Filter: played games only, before today
            new_games = new_games[new_games['wl'].notna()]
            new_games = new_games[new_games['game_date'] < today]
            
            if since_date:
                new_games = new_games[new_games['game_date'] >= pd.Timestamp(since_date)]
            
            if new_games.empty:
                logger.info(f"  No new games")
                # Store the games this team has played (for player lookup)
                self.team_missing_games[team_id] = set()
                continue
            
            # Track which games we're adding for this team
            new_game_ids = set(new_games['game_id'].tolist())
            self.team_missing_games[team_id] = new_game_ids
            
            logger.info(f"  Found {len(new_games)} new games to insert")
            
            # Transform and load
            new_games = self._transform_team_game_log(new_games)
            
            try:
                self.loader.insert_dataframe(new_games, 'team_game_logs', ['game_id', 'team_id'])
                total_inserted += len(new_games)
                logger.info(f"  ✓ Inserted {len(new_games)} new games")
                
                for _, row in new_games.iterrows():
                    existing_games.add((str(row['game_id']), int(row['team_id'])))
                
            except Exception as e:
                logger.error(f"  ✗ Error inserting: {e}")
        
        logger.info(f"\nTotal team game logs inserted: {total_inserted}")
        return total_inserted
    
    # =========================================================================
    # PLAYER GAME LOGS (OPTIMIZED)
    # =========================================================================
    
    def fetch_player_game_log(self, player_id: int, season: str, player_name: str = "Unknown") -> Optional[pd.DataFrame]:
        """Fetch player game log for a season"""
        try:
            self.rate_limit()
            game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            df = game_log.get_data_frames()[0]
            if not df.empty:
                df['season'] = season
                df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            logger.warning(f"  Error fetching {player_name}: {e}")
            return None
    
    def update_player_game_logs(self, season: str, since_date: datetime = None) -> int:
        """
        Update player game logs - OPTIMIZED VERSION
        Only fetches data for roster players who are missing games
        """
        logger.info("=" * 60)
        logger.info("UPDATING PLAYER GAME LOGS (ROSTER-BASED)")
        logger.info("=" * 60)
        
        today = pd.Timestamp(datetime.now().date())
        since_ts = pd.Timestamp(since_date) if since_date else None
        
        # Collect all roster player IDs
        all_roster_players = {}  # player_id -> {team_id, player_name}
        for team_id, roster in self.team_rosters.items():
            for player in roster:
                all_roster_players[player['player_id']] = {
                    'team_id': team_id,
                    'player_name': player['player_name']
                }
        
        logger.info(f"Total roster players to check: {len(all_roster_players)}")
        
        # Get existing player-game records
        roster_player_ids = list(all_roster_players.keys())
        existing_player_games = self.get_existing_player_games(season, roster_player_ids)
        
        # For each team, get the games they've played (from team_game_logs)
        # NOW WITH DATE FILTERING
        team_games: Dict[int, Set[str]] = {}
        for team_id in self.team_rosters.keys():
            team_games[team_id] = self.get_team_games_from_db(team_id, season, since_date)
        
        # Determine which players need updates
        players_to_update = []
        for player_id, info in all_roster_players.items():
            team_id = info['team_id']
            player_name = info['player_name']
            
            # Games this team has played (already filtered by date)
            team_game_ids = team_games.get(team_id, set())
            
            # Games this player already has logs for
            player_existing_games = existing_player_games.get(player_id, set())
            
            # Missing games = team games - player games
            missing_games = team_game_ids - player_existing_games
            
            if missing_games:
                players_to_update.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'team_id': team_id,
                    'missing_games': missing_games
                })
        
        logger.info(f"Players with missing game logs: {len(players_to_update)}")
        
        if not players_to_update:
            logger.info("All roster players are up to date!")
            return 0
        
        # Fetch and insert missing player game logs
        total_inserted = 0
        
        for idx, player_info in enumerate(players_to_update, 1):
            player_id = player_info['player_id']
            player_name = player_info['player_name']
            missing_games = player_info['missing_games']
            
            logger.info(f"[{idx}/{len(players_to_update)}] {player_name} - {len(missing_games)} missing games")
            
            df = self.fetch_player_game_log(player_id, season, player_name)
            
            if df is None or df.empty:
                logger.info(f"  No data from API")
                continue
            
            # Prepare data
            df['game_date'] = pd.to_datetime(df['game_date'], format='mixed', errors='coerce')
            df['game_id'] = df['game_id'].astype(str)
            df['player_id'] = player_id
            
            # Filter to only the missing games
            new_games = df[df['game_id'].isin(missing_games)].copy()
            
            # Additional filters
            new_games = new_games[new_games['wl'].notna()]
            new_games = new_games[new_games['game_date'] < today]
            
            # Date filter is already applied via missing_games, but keep for safety
            if since_ts:
                new_games = new_games[new_games['game_date'] >= since_ts]
            
            if new_games.empty:
                logger.info(f"  No new games after filtering")
                continue
            
            # Transform and load
            new_games = self._transform_player_game_log(new_games)
            
            try:
                self.loader.insert_dataframe(new_games, 'player_game_logs', ['game_id', 'player_id'])
                total_inserted += len(new_games)
                logger.info(f"  ✓ Inserted {len(new_games)} games")
                
            except Exception as e:
                logger.error(f"  ✗ Error inserting: {e}")
        
        logger.info(f"\nTotal player game logs inserted: {total_inserted}")
        return total_inserted
    
    # =========================================================================
    # TRANSFORMERS
    # =========================================================================
    
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
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
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
    
    # =========================================================================
    # MAIN RUNNER
    # =========================================================================
    
    def run_incremental_update(self, season: str = None, days_back: int = None, full_season: bool = False):
        """Run incremental update"""
        logger.info("\n" + "=" * 60)
        logger.info("NBA INCREMENTAL UPDATE (ROSTER-OPTIMIZED)")
        logger.info("=" * 60)
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if season is None:
            season = self.get_season_string()
        logger.info(f"Season: {season}")
        
        if days_back:
            since_date = datetime.now() - timedelta(days=days_back)
            logger.info(f"Looking back {days_back} days (since {since_date.date()})")
        elif full_season:
            season_start_year = int(season.split('-')[0])
            since_date = datetime(season_start_year, 10, 1)
            logger.info(f"Checking full season since {since_date.date()}")
        else:
            since_date = None
            logger.info(f"Checking for any missing games in {season}")
        
        self.db.connect()
        
        try:
            # Step 1: Fetch all current rosters
            self.fetch_all_rosters(season)
            
            # Step 2: Add any new players from rosters
            new_players_count = self.update_players_from_rosters()
            
            # Step 3: Update team game logs (this also identifies missing games per team)
            team_count = self.update_team_game_logs(season, since_date)
            
            # Step 4: Update player game logs (only for roster players with missing games)
            player_count = self.update_player_game_logs(season, since_date)
            
            # Summary
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
        description='Incremental NBA Data Update (Roster-Optimized)',
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