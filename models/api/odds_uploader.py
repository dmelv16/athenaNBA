"""
Odds Database Uploader
Handles storing and retrieving odds data from PostgreSQL
"""

import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Union
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from etl.config.settings import DatabaseConfig


class OddsUploader:
    """Upload and retrieve odds data from database"""
    
    TEAM_NAME_MAP = {
        'Los Angeles Lakers': 'LAL', 'Boston Celtics': 'BOS',
        'Golden State Warriors': 'GSW', 'Phoenix Suns': 'PHX',
        'Milwaukee Bucks': 'MIL', 'Miami Heat': 'MIA',
        'Philadelphia 76ers': 'PHI', 'Denver Nuggets': 'DEN',
        'Memphis Grizzlies': 'MEM', 'Cleveland Cavaliers': 'CLE',
        'Sacramento Kings': 'SAC', 'Brooklyn Nets': 'BKN',
        'New York Knicks': 'NYK', 'Dallas Mavericks': 'DAL',
        'Atlanta Hawks': 'ATL', 'Toronto Raptors': 'TOR',
        'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
        'LA Clippers': 'LAC', 'Chicago Bulls': 'CHI',
        'Utah Jazz': 'UTA', 'Oklahoma City Thunder': 'OKC',
        'Indiana Pacers': 'IND', 'Portland Trail Blazers': 'POR',
        'Orlando Magic': 'ORL', 'Charlotte Hornets': 'CHA',
        'Houston Rockets': 'HOU', 'San Antonio Spurs': 'SAS',
        'Detroit Pistons': 'DET', 'Washington Wizards': 'WAS',
    }
    
    def __init__(self):
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=DatabaseConfig.HOST,
                port=DatabaseConfig.PORT,
                dbname=DatabaseConfig.NAME,
                user=DatabaseConfig.USER,
                password=DatabaseConfig.PASSWORD,
                sslmode=DatabaseConfig.SSLMODE
            )
            print(f"✓ Connected to database")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            raise
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    def init_schema(self):
        """Initialize odds tables"""
        schema_path = Path(__file__).parent / 'odds_schema.sql'
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            try:
                with self.conn.cursor() as cur:
                    cur.execute(schema_sql)
                    self.conn.commit()
                print("✓ Odds schema initialized")
            except Exception as e:
                print(f"Schema initialization error: {e}")
                self.conn.rollback()
        else:
            print(f"Schema file not found: {schema_path}")
    
    def _get_team_abbrev(self, team_name: str) -> str:
        """Get team abbreviation from full name"""
        return self.TEAM_NAME_MAP.get(team_name, team_name[:3].upper() if team_name else 'UNK')
    
    # ==========================================
    # Upload Methods
    # ==========================================
    
    def upload_game_odds(
        self,
        game_lines: List,  # List of GameLine objects
        snapshot_time: datetime = None
    ) -> int:
        """Upload game odds (spread, total, moneyline)"""
        if not game_lines:
            return 0
        
        snapshot_time = snapshot_time or datetime.now()
        
        query = """
            INSERT INTO nba_game_odds (
                game_id, game_date, snapshot_time,
                home_team, away_team, home_team_abbrev, away_team_abbrev,
                home_team_id, away_team_id,
                spread_line, spread_home_odds, spread_away_odds,
                total_line, over_odds, under_odds,
                home_ml, away_ml,
                bookmaker, is_pregame
            ) VALUES %s
            ON CONFLICT (game_id, bookmaker, snapshot_time)
            DO UPDATE SET
                spread_line = EXCLUDED.spread_line,
                spread_home_odds = EXCLUDED.spread_home_odds,
                spread_away_odds = EXCLUDED.spread_away_odds,
                total_line = EXCLUDED.total_line,
                over_odds = EXCLUDED.over_odds,
                under_odds = EXCLUDED.under_odds,
                home_ml = EXCLUDED.home_ml,
                away_ml = EXCLUDED.away_ml
        """
        
        values = []
        for gl in game_lines:
            home_abbrev = self._get_team_abbrev(gl.home_team)
            away_abbrev = self._get_team_abbrev(gl.away_team)
            
            # Use American odds for storage (more common)
            spread_home = gl.spread_home_american if hasattr(gl, 'spread_home_american') else None
            spread_away = gl.spread_away_american if hasattr(gl, 'spread_away_american') else None
            over = gl.over_american if hasattr(gl, 'over_american') else None
            under = gl.under_american if hasattr(gl, 'under_american') else None
            home_ml = gl.home_ml_american if hasattr(gl, 'home_ml_american') else None
            away_ml = gl.away_ml_american if hasattr(gl, 'away_ml_american') else None
            
            values.append((
                gl.game_id,
                gl.game_date,
                snapshot_time,
                gl.home_team,
                gl.away_team,
                home_abbrev,
                away_abbrev,
                gl.home_team_id,
                gl.away_team_id,
                gl.spread_line,
                spread_home,
                spread_away,
                gl.total_line,
                over,
                under,
                home_ml,
                away_ml,
                gl.bookmaker,
                True
            ))
        
        try:
            with self.conn.cursor() as cur:
                execute_values(cur, query, values)
                self.conn.commit()
            return len(values)
        except Exception as e:
            print(f"Error uploading game odds: {e}")
            self.conn.rollback()
            return 0
    
    def upload_player_props(
        self,
        props: List,  # List of PlayerPropLine objects
        game_date: date,
        snapshot_time: datetime = None
    ) -> int:
        """Upload player prop odds"""
        if not props:
            return 0
        
        snapshot_time = snapshot_time or datetime.now()
        
        query = """
            INSERT INTO nba_player_prop_odds (
                game_id, game_date, snapshot_time,
                player_name, player_id, team_abbrev, opponent_abbrev,
                prop_type, line, over_odds, under_odds,
                bookmaker, is_pregame
            ) VALUES %s
            ON CONFLICT (game_id, player_name, prop_type, bookmaker, snapshot_time)
            DO UPDATE SET
                line = EXCLUDED.line,
                over_odds = EXCLUDED.over_odds,
                under_odds = EXCLUDED.under_odds
        """
        
        values = []
        for p in props:
            # Use American odds
            over = p.over_odds_american if hasattr(p, 'over_odds_american') else -110
            under = p.under_odds_american if hasattr(p, 'under_odds_american') else -110
            
            values.append((
                p.game_id,
                game_date,
                snapshot_time,
                p.player_name,
                p.player_id,
                p.team_abbrev,
                None,  # opponent_abbrev
                p.prop_type,
                p.line,
                over,
                under,
                p.bookmaker,
                True
            ))
        
        try:
            with self.conn.cursor() as cur:
                execute_values(cur, query, values)
                self.conn.commit()
            return len(values)
        except Exception as e:
            print(f"Error uploading player props: {e}")
            self.conn.rollback()
            return 0
    
    def log_snapshot(
        self,
        game_date: date,
        games_captured: int,
        player_props_captured: int,
        game_odds_captured: int,
        bookmaker: str,
        status: str = 'success',
        error_message: str = None
    ):
        """Log an odds snapshot"""
        query = """
            INSERT INTO nba_odds_snapshots (
                snapshot_time, game_date,
                games_captured, player_props_captured, game_odds_captured,
                bookmaker, status, error_message
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (
                    datetime.now(), game_date,
                    games_captured, player_props_captured, game_odds_captured,
                    bookmaker, status, error_message
                ))
                self.conn.commit()
        except Exception as e:
            print(f"Error logging snapshot: {e}")
            self.conn.rollback()
    
    # ==========================================
    # Retrieval Methods
    # ==========================================
    
    def get_game_odds(self, game_date: date, bookmaker: str = None) -> List[Dict]:
        """Get game odds for a date"""
        query = """
            SELECT DISTINCT ON (game_id, bookmaker)
                * FROM nba_game_odds
            WHERE game_date = %s AND is_pregame = TRUE
        """
        params = [game_date]
        
        if bookmaker:
            query += " AND bookmaker = %s"
            params.append(bookmaker)
        
        query += " ORDER BY game_id, bookmaker, snapshot_time DESC"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting game odds: {e}")
            return []
    
    def get_player_props(
        self,
        game_date: date,
        player_name: str = None,
        prop_type: str = None,
        bookmaker: str = None
    ) -> List[Dict]:
        """Get player props for a date"""
        query = """
            SELECT DISTINCT ON (game_id, player_name, prop_type, bookmaker)
                * FROM nba_player_prop_odds
            WHERE game_date = %s AND is_pregame = TRUE
        """
        params = [game_date]
        
        if player_name:
            query += " AND player_name ILIKE %s"
            params.append(f'%{player_name}%')
        if prop_type:
            query += " AND prop_type = %s"
            params.append(prop_type)
        if bookmaker:
            query += " AND bookmaker = %s"
            params.append(bookmaker)
        
        query += " ORDER BY game_id, player_name, prop_type, bookmaker, snapshot_time DESC"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting player props: {e}")
            return []
    
    def get_player_prop_line(
        self,
        player_name: str,
        prop_type: str,
        game_date: date,
        bookmaker: str = None
    ) -> Optional[float]:
        """Get a specific player's prop line"""
        query = """
            SELECT line FROM nba_player_prop_odds
            WHERE game_date = %s
              AND player_name ILIKE %s
              AND prop_type = %s
              AND is_pregame = TRUE
        """
        params = [game_date, f'%{player_name}%', prop_type]
        
        if bookmaker:
            query += " AND bookmaker = %s"
            params.append(bookmaker)
        
        query += " ORDER BY snapshot_time DESC LIMIT 1"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                result = cur.fetchone()
                return float(result[0]) if result else None
        except:
            return None
    
    def get_game_spread(
        self,
        home_team_abbrev: str,
        game_date: date,
        bookmaker: str = None
    ) -> Optional[float]:
        """Get spread line for a game"""
        query = """
            SELECT spread_line FROM nba_game_odds
            WHERE game_date = %s
              AND home_team_abbrev = %s
              AND is_pregame = TRUE
        """
        params = [game_date, home_team_abbrev]
        
        if bookmaker:
            query += " AND bookmaker = %s"
            params.append(bookmaker)
        
        query += " ORDER BY snapshot_time DESC LIMIT 1"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                result = cur.fetchone()
                return float(result[0]) if result and result[0] else None
        except:
            return None
    
    def get_game_total(
        self,
        home_team_abbrev: str,
        game_date: date,
        bookmaker: str = None
    ) -> Optional[float]:
        """Get total line for a game"""
        query = """
            SELECT total_line FROM nba_game_odds
            WHERE game_date = %s
              AND home_team_abbrev = %s
              AND is_pregame = TRUE
        """
        params = [game_date, home_team_abbrev]
        
        if bookmaker:
            query += " AND bookmaker = %s"
            params.append(bookmaker)
        
        query += " ORDER BY snapshot_time DESC LIMIT 1"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                result = cur.fetchone()
                return float(result[0]) if result and result[0] else None
        except:
            return None