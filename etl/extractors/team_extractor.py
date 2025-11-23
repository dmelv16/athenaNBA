"""
Extractor for team-specific NBA data
"""

from typing import Optional
import pandas as pd
from nba_api.stats.endpoints import teamgamelog

from extractors.base_extractor import BaseExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


class TeamDataExtractor(BaseExtractor):
    """Extract team-specific data from NBA API"""
    
    def extract(self):
        """Not used - use specific methods instead"""
        pass
    
    def get_team_game_logs(
        self,
        team_id: int,
        season: str,
        team_name: str = "Unknown"
    ) -> Optional[pd.DataFrame]:
        """
        Get game-by-game logs for a team in a season
        
        Args:
            team_id: NBA team ID
            season: Season string (e.g., '2023-24')
            team_name: Team name for logging (optional)
            
        Returns:
            DataFrame with game logs or None if failed
        """
        def _extract():
            game_log = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season
            )
            df = game_log.get_data_frames()[0]
            
            if not df.empty:
                df['season'] = season
                df.columns = df.columns.str.lower()
                logger.debug(f"  {team_name} ({season}): {len(df)} games")
            
            return df
        
        result = self.extract_with_retry(_extract)
        
        if result is None or result.empty:
            logger.debug(f"  ✗ {team_name} ({season}): No game logs")
        
        return result