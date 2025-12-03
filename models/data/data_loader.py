"""
Data loading utilities for the prediction system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from etl.database.connection import get_db_connection


class NBADataLoader:
    """Load and prepare NBA data for predictions"""
    
    def __init__(self, min_date: str = '2020-01-01'):
        """
        Initialize data loader
        
        Args:
            min_date: Minimum date to load data from
        """
        self.min_date = min_date
        self.db = get_db_connection()
        
        # Cached data
        self._player_logs = None
        self._team_logs = None
        self._players = None
        self._teams = None
    
    def load_player_game_logs(self, force_reload: bool = False) -> pd.DataFrame:
        """Load player game logs from database"""
        if self._player_logs is not None and not force_reload:
            return self._player_logs
        
        query = f"""
            SELECT 
                pgl.*,
                p.full_name as player_name,
                p.is_active
            FROM player_game_logs pgl
            LEFT JOIN players p ON pgl.player_id = p.player_id
            WHERE pgl.game_date >= '{self.min_date}'
            ORDER BY pgl.player_id, pgl.game_date
        """
        
        with self.db.get_cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            self._player_logs = pd.DataFrame(cur.fetchall(), columns=columns)
        
        # Convert types
        self._player_logs['game_date'] = pd.to_datetime(self._player_logs['game_date'])
        
        return self._player_logs
    
    def load_team_game_logs(self, force_reload: bool = False) -> pd.DataFrame:
        """Load team game logs from database"""
        if self._team_logs is not None and not force_reload:
            return self._team_logs
        
        query = f"""
            SELECT 
                tgl.*,
                t.team_name,
                t.abbreviation
            FROM team_game_logs tgl
            LEFT JOIN teams t ON tgl.team_id = t.team_id
            WHERE tgl.game_date >= '{self.min_date}'
            ORDER BY tgl.team_id, tgl.game_date
        """
        
        with self.db.get_cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            self._team_logs = pd.DataFrame(cur.fetchall(), columns=columns)
        
        self._team_logs['game_date'] = pd.to_datetime(self._team_logs['game_date'])
        
        return self._team_logs
    
    def load_players(self, active_only: bool = True) -> pd.DataFrame:
        """Load players reference table"""
        if self._players is not None:
            df = self._players
            if active_only:
                df = df[df['is_active'] == True]
            return df
        
        query = "SELECT * FROM players"
        
        with self.db.get_cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            self._players = pd.DataFrame(cur.fetchall(), columns=columns)
        
        if active_only:
            return self._players[self._players['is_active'] == True]
        return self._players
    
    def load_teams(self) -> pd.DataFrame:
        """Load teams reference table"""
        if self._teams is not None:
            return self._teams
        
        query = "SELECT * FROM teams"
        
        with self.db.get_cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            self._teams = pd.DataFrame(cur.fetchall(), columns=columns)
        
        return self._teams
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all required data"""
        player_logs = self.load_player_game_logs()
        team_logs = self.load_team_game_logs()
        players = self.load_players()
        teams = self.load_teams()
        
        return player_logs, team_logs, players, teams
    
    def get_player_recent_games(
        self,
        player_id: int,
        num_games: int = 20
    ) -> pd.DataFrame:
        """Get recent games for a specific player"""
        df = self.load_player_game_logs()
        player_df = df[df['player_id'] == player_id].tail(num_games)
        return player_df
    
    def get_team_recent_games(
        self,
        team_id: int,
        num_games: int = 20
    ) -> pd.DataFrame:
        """Get recent games for a specific team"""
        df = self.load_team_game_logs()
        team_df = df[df['team_id'] == team_id].tail(num_games)
        return team_df
    
    def get_head_to_head(
        self,
        team1_id: int,
        team2_id: int,
        num_games: int = 10
    ) -> pd.DataFrame:
        """Get head-to-head matchup history"""
        df = self.load_team_game_logs()
        
        # Get games involving team1
        team1_games = df[df['team_id'] == team1_id].copy()
        
        # Filter to games against team2 (check matchup string)
        team2_info = self.load_teams()
        team2_abbrev = team2_info[team2_info['team_id'] == team2_id]['abbreviation'].iloc[0]
        
        h2h = team1_games[team1_games['matchup'].str.contains(team2_abbrev, na=False)]
        
        return h2h.tail(num_games)
    
    def get_player_vs_opponent(
        self,
        player_id: int,
        opponent_abbrev: str,
        num_games: int = 10
    ) -> pd.DataFrame:
        """Get player's history against a specific opponent"""
        df = self.load_player_game_logs()
        player_df = df[df['player_id'] == player_id].copy()
        
        # Filter to games against opponent
        vs_opp = player_df[player_df['matchup'].str.contains(opponent_abbrev, na=False)]
        
        return vs_opp.tail(num_games)
    
    def get_active_player_ids(self) -> List[int]:
        """Get list of active player IDs"""
        players = self.load_players(active_only=True)
        return players['player_id'].tolist()
    
    def get_team_id_mapping(self) -> Dict[str, int]:
        """Get mapping of team abbreviation to team ID"""
        teams = self.load_teams()
        return dict(zip(teams['abbreviation'], teams['team_id']))
    
    def get_player_info(self, player_id: int) -> Optional[Dict]:
        """Get player information"""
        players = self.load_players(active_only=False)
        player = players[players['player_id'] == player_id]
        
        if player.empty:
            return None
        
        return player.iloc[0].to_dict()
    
    def get_team_info(self, team_id: int) -> Optional[Dict]:
        """Get team information"""
        teams = self.load_teams()
        team = teams[teams['team_id'] == team_id]
        
        if team.empty:
            return None
        
        return team.iloc[0].to_dict()
    
    def close(self):
        """Close database connection"""
        self.db.close()


class LiveDataIntegration:
    """Integration point for live data feeds (lines, injuries, etc.)"""
    
    def __init__(self):
        # Placeholder for API connections
        self.lines_api = None
        self.injury_api = None
    
    def get_current_lines(self, game_id: str) -> Dict:
        """
        Get current betting lines for a game
        
        In production, this would connect to an odds API
        """
        # Placeholder - return sample lines
        return {
            'spread': -3.5,
            'total': 224.5,
            'moneyline_home': -150,
            'moneyline_away': +130
        }
    
    def get_player_prop_lines(self, player_id: int, game_id: str) -> Dict:
        """
        Get current player prop lines
        
        In production, this would connect to a props API
        """
        # Placeholder - return sample lines
        return {
            'pts': 24.5,
            'reb': 7.5,
            'ast': 6.5,
            'pra': 38.5,
            'stl': 1.5,
            'blk': 0.5
        }
    
    def get_injury_report(self) -> List[Dict]:
        """
        Get current injury report
        
        In production, this would connect to an injury API
        """
        # Placeholder
        return []
    
    def get_todays_games(self) -> List[Dict]:
        """
        Get today's game schedule
        
        In production, this would connect to an NBA schedule API
        """
        # Placeholder
        return []