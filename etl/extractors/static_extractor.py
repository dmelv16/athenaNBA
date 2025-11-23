"""
Extractor for static NBA data (players and teams lists)
"""

from typing import List, Dict, Any
from nba_api.stats.static import players, teams

from extractors.base_extractor import BaseExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


class StaticDataExtractor(BaseExtractor):
    """Extract static NBA data that doesn't change frequently"""
    
    def extract(self):
        """Not used for static extractor"""
        pass
    
    def get_all_players(self) -> List[Dict[str, Any]]:
        """
        Get all NBA players (current and historical)
        
        Returns:
            List of player dictionaries
        """
        logger.info("Fetching all NBA players...")
        try:
            players_data = players.get_players()
            logger.info(f"✓ Retrieved {len(players_data)} players")
            return players_data
        except Exception as e:
            logger.error(f"Failed to fetch players: {e}")
            return []
    
    def get_all_teams(self) -> List[Dict[str, Any]]:
        """
        Get all NBA teams
        
        Returns:
            List of team dictionaries
        """
        logger.info("Fetching all NBA teams...")
        try:
            teams_data = teams.get_teams()
            logger.info(f"✓ Retrieved {len(teams_data)} teams")
            return teams_data
        except Exception as e:
            logger.error(f"Failed to fetch teams: {e}")
            return []
    
    def get_active_players(self) -> List[Dict[str, Any]]:
        """
        Get only active NBA players
        
        Returns:
            List of active player dictionaries
        """
        all_players = self.get_all_players()
        active = [p for p in all_players if p.get('is_active', False)]
        logger.info(f"✓ Filtered to {len(active)} active players")
        return active
    
    def get_player_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Search for players by name
        
        Args:
            name: Player name to search for
            
        Returns:
            List of matching player dictionaries
        """
        all_players = self.get_all_players()
        matches = [
            p for p in all_players 
            if name.lower() in p['full_name'].lower()
        ]
        logger.info(f"Found {len(matches)} players matching '{name}'")
        return matches
    
    def get_seasons_list(self, start_season: str, end_season: str) -> List[str]:
        """
        Generate list of seasons between start and end
        
        Args:
            start_season: Starting season (e.g., '2014-15')
            end_season: Ending season (e.g., '2024-25')
            
        Returns:
            List of season strings
        """
        start_year = int(start_season.split('-')[0])
        end_year = int(end_season.split('-')[0])
        
        seasons = []
        for year in range(start_year, end_year + 1):
            next_year = str(year + 1)[-2:]
            seasons.append(f"{year}-{next_year}")
        
        logger.info(f"Generated {len(seasons)} seasons: {start_season} to {end_season}")
        return seasons