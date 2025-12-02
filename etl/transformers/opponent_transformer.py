"""
Transformer for team performance stats data (alternative to opponent stats)
"""

import pandas as pd
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class OpponentStatsTransformer:
    """Transform team performance stats data for database insertion"""
    
    # Updated column mapping for team performance endpoint
    OPPONENT_STATS_COLUMNS = {
        'player_id': 'player_id',
        'season': 'season',
        'team_id': 'opponent_team_id',  # May vary based on endpoint response
        'vs_team_id': 'opponent_team_id',
        'opp_team_id': 'opponent_team_id',
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
        Transform team performance stats DataFrame
        
        Args:
            df: Raw DataFrame from NBA API
            player_id: Player ID to add to DataFrame
            
        Returns:
            Transformed DataFrame ready for database insertion
        """
        if df is None or df.empty:
            return None
        
        try:
            df = df.copy()
            df['player_id'] = player_id
            
            # Try to find the opponent_team_id column (it might have different names)
            opponent_col = None
            for col in ['team_id', 'vs_team_id', 'opp_team_id']:
                if col in df.columns:
                    opponent_col = col
                    break
            
            if opponent_col and opponent_col != 'opponent_team_id':
                df['opponent_team_id'] = df[opponent_col]
            
            # Handle any date columns with mixed format
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            for date_col in date_columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not parse date column {date_col}: {e}")
            
            # Select and rename columns
            available_cols = {}
            for k, v in OpponentStatsTransformer.OPPONENT_STATS_COLUMNS.items():
                if k in df.columns:
                    # Only add if we haven't already added this target column
                    if v not in available_cols.values():
                        available_cols[k] = v
            
            if not available_cols:
                logger.warning("No matching columns found in opponent stats")
                return None
            
            df_clean = df[list(available_cols.keys())].copy()
            df_clean.columns = list(available_cols.values())
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error transforming opponent stats for player {player_id}: {e}")
            return None