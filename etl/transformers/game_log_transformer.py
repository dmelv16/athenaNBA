"""
Transformers for game log data
"""

import pandas as pd
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class GameLogTransformer:
    """Transform game log data for database insertion"""
    
    # Column mapping from NBA API to database schema
    PLAYER_GAME_LOG_COLUMNS = {
        'game_id': 'game_id',
        'player_id': 'player_id',
        'season': 'season',
        'game_date': 'game_date',
        'matchup': 'matchup',
        'wl': 'wl',
        'min': 'min',
        'fgm': 'fgm',
        'fga': 'fga',
        'fg_pct': 'fg_pct',
        'fg3m': 'fg3m',
        'fg3a': 'fg3a',
        'fg3_pct': 'fg3_pct',
        'ftm': 'ftm',
        'fta': 'fta',
        'ft_pct': 'ft_pct',
        'oreb': 'oreb',
        'dreb': 'dreb',
        'reb': 'reb',
        'ast': 'ast',
        'stl': 'stl',
        'blk': 'blk',
        'tov': 'tov',
        'pf': 'pf',
        'pts': 'pts',
        'plus_minus': 'plus_minus'
    }
    
    TEAM_GAME_LOG_COLUMNS = {
        'game_id': 'game_id',
        'team_id': 'team_id',
        'season': 'season',
        'game_date': 'game_date',
        'matchup': 'matchup',
        'wl': 'wl',
        'min': 'min',
        'fgm': 'fgm',
        'fga': 'fga',
        'fg_pct': 'fg_pct',
        'fg3m': 'fg3m',
        'fg3a': 'fg3a',
        'fg3_pct': 'fg3_pct',
        'ftm': 'ftm',
        'fta': 'fta',
        'ft_pct': 'ft_pct',
        'oreb': 'oreb',
        'dreb': 'dreb',
        'reb': 'reb',
        'ast': 'ast',
        'stl': 'stl',
        'blk': 'blk',
        'tov': 'tov',
        'pf': 'pf',
        'pts': 'pts',
        'plus_minus': 'plus_minus'
    }
    
    @staticmethod
    def transform_player_game_log(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Transform player game log DataFrame
        
        Args:
            df: Raw DataFrame from NBA API
            
        Returns:
            Transformed DataFrame ready for database insertion
        """
        if df is None or df.empty:
            return None
        
        # Select and rename columns
        available_cols = {
            k: v for k, v in GameLogTransformer.PLAYER_GAME_LOG_COLUMNS.items()
            if k in df.columns
        }
        
        df_clean = df[list(available_cols.keys())].copy()
        df_clean.columns = list(available_cols.values())
        
        # Data type conversions and cleaning
        if 'game_date' in df_clean.columns:
            # Use 'mixed' format to handle different date formats from NBA API
            # The API sometimes returns "Apr 21, 2021" and sometimes "April 21, 2021"
            df_clean['game_date'] = pd.to_datetime(df_clean['game_date'], format='mixed', errors='coerce')
        
        # Convert minutes from string format (e.g., "32:15") to decimal
        if 'min' in df_clean.columns:
            df_clean['min'] = GameLogTransformer._convert_minutes(df_clean['min'])
        
        return df_clean
    
    @staticmethod
    def transform_team_game_log(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Transform team game log DataFrame
        
        Args:
            df: Raw DataFrame from NBA API
            
        Returns:
            Transformed DataFrame ready for database insertion
        """
        if df is None or df.empty:
            return None
        
        # Select and rename columns
        available_cols = {
            k: v for k, v in GameLogTransformer.TEAM_GAME_LOG_COLUMNS.items()
            if k in df.columns
        }
        
        df_clean = df[list(available_cols.keys())].copy()
        df_clean.columns = list(available_cols.values())
        
        # Data type conversions
        if 'game_date' in df_clean.columns:
            # Use 'mixed' format to handle different date formats
            df_clean['game_date'] = pd.to_datetime(df_clean['game_date'], format='mixed', errors='coerce')
        
        return df_clean
    
    @staticmethod
    def _convert_minutes(minutes_series: pd.Series) -> pd.Series:
        """
        Convert minutes from MM:SS format to decimal
        
        Args:
            minutes_series: Series with minutes in MM:SS or decimal format
            
        Returns:
            Series with minutes as decimal
        """
        def convert_single(val):
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
            return float(val)
        
        return minutes_series.apply(convert_single)