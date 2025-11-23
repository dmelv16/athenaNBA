"""
Transformer for opponent stats data
"""

import pandas as pd
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class OpponentStatsTransformer:
    """Transform opponent stats data for database insertion"""
    
    OPPONENT_STATS_COLUMNS = {
        'player_id': 'player_id',
        'season': 'season',
        'vs_team_id': 'opponent_team_id',
        'gp': 'gp',
        'w': 'w',
        'l': 'l',
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
        'tov': 'tov',
        'stl': 'stl',
        'blk': 'blk',
        'blka': 'blka',
        'pf': 'pf',
        'pfd': 'pfd',
        'pts': 'pts',
        'plus_minus': 'plus_minus'
    }
    
    @staticmethod
    def transform_opponent_stats(
        df: pd.DataFrame,
        player_id: int
    ) -> Optional[pd.DataFrame]:
        """
        Transform opponent stats DataFrame
        
        Args:
            df: Raw DataFrame from NBA API
            player_id: Player ID to add to DataFrame
            
        Returns:
            Transformed DataFrame ready for database insertion
        """
        if df is None or df.empty:
            return None
        
        df = df.copy()
        df['player_id'] = player_id
        
        # Select and rename columns
        available_cols = {
            k: v for k, v in OpponentStatsTransformer.OPPONENT_STATS_COLUMNS.items()
            if k in df.columns
        }
        
        df_clean = df[list(available_cols.keys())].copy()
        df_clean.columns = list(available_cols.values())
        
        return df_clean