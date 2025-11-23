"""
Extractor for player-specific NBA data
"""

from typing import Optional
import pandas as pd
from nba_api.stats.endpoints import (
    playergamelog,
    playerdashboardbyopponent,
    playerdashboardbygeneralsplits
)

from extractors.base_extractor import BaseExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


class PlayerDataExtractor(BaseExtractor):
    """Extract player-specific data from NBA API"""
    
    def extract(self):
        """Not used - use specific methods instead"""
        pass
    
    def get_player_game_logs(
        self,
        player_id: int,
        season: str,
        player_name: str = "Unknown"
    ) -> Optional[pd.DataFrame]:
        """
        Get game-by-game logs for a player in a season
        
        Args:
            player_id: NBA player ID
            season: Season string (e.g., '2023-24')
            player_name: Player name for logging (optional)
            
        Returns:
            DataFrame with game logs or None if failed
        """
        def _extract():
            game_log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season
            )
            df = game_log.get_data_frames()[0]
            
            if not df.empty:
                df['season'] = season
                df.columns = df.columns.str.lower()
                logger.debug(f"  ✓ {player_name} ({season}): {len(df)} games")
            
            return df
        
        result = self.extract_with_retry(_extract)
        
        if result is None or result.empty:
            logger.debug(f"  ✗ {player_name} ({season}): No game logs")
        
        return result
    
    def get_player_opponent_stats(
        self,
        player_id: int,
        season: str,
        player_name: str = "Unknown"
    ) -> Optional[pd.DataFrame]:
        """
        Get player statistics vs each opponent
        
        Args:
            player_id: NBA player ID
            season: Season string
            player_name: Player name for logging
            
        Returns:
            DataFrame with opponent stats or None if failed
        """
        def _extract():
            opp_stats = playerdashboardbyopponent.PlayerDashboardByOpponent(
                player_id=player_id,
                season=season
            )
            df = opp_stats.get_data_frames()[0]
            
            if not df.empty:
                df['season'] = season
                df.columns = df.columns.str.lower()
                logger.debug(f"  ✓ {player_name} ({season}): opponent stats")
            
            return df
        
        result = self.extract_with_retry(_extract)
        
        if result is None or result.empty:
            logger.debug(f"  ✗ {player_name} ({season}): No opponent stats")
        
        return result
    
    def get_player_general_splits(
        self,
        player_id: int,
        season: str,
        player_name: str = "Unknown"
    ) -> Optional[pd.DataFrame]:
        """
        Get player splits (home/away, wins/losses, etc.)
        
        Args:
            player_id: NBA player ID
            season: Season string
            player_name: Player name for logging
            
        Returns:
            DataFrame with splits or None if failed
        """
        def _extract():
            splits = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
                player_id=player_id,
                season=season
            )
            df = splits.get_data_frames()[0]
            
            if not df.empty:
                df['season'] = season
                df.columns = df.columns.str.lower()
                df['split_type'] = 'general'
                logger.debug(f"  ✓ {player_name} ({season}): splits")
            
            return df
        
        result = self.extract_with_retry(_extract)
        
        if result is None or result.empty:
            logger.debug(f"  ✗ {player_name} ({season}): No splits")
        
        return result
    
    def get_all_player_data(
        self,
        player_id: int,
        season: str,
        player_name: str = "Unknown"
    ) -> dict:
        """
        Get all available data for a player in a season
        
        Args:
            player_id: NBA player ID
            season: Season string
            player_name: Player name for logging
            
        Returns:
            Dictionary with 'game_logs', 'opponent_stats', 'splits' keys
        """
        logger.info(f"Extracting all data for {player_name} ({season})")
        
        return {
            'game_logs': self.get_player_game_logs(player_id, season, player_name),
            'opponent_stats': self.get_player_opponent_stats(player_id, season, player_name),
            'splits': self.get_player_general_splits(player_id, season, player_name)
        }