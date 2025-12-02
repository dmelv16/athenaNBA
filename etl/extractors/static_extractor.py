"""
Extractor for static NBA data (players and teams lists)
"""

from typing import List, Dict, Any
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonallplayers
import pandas as pd
import time

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
    
    def get_players_for_seasons(self, seasons: List[str]) -> List[Dict[str, Any]]:
        """
        Get all players who played in the specified seasons
        Uses league-wide endpoints for efficiency
        
        Args:
            seasons: List of season strings (e.g., ['2023-24', '2024-25'])
            
        Returns:
            List of player dictionaries who played in any of those seasons
        """
        logger.info(f"Fetching players who played in seasons: {', '.join(seasons)}")
        
        all_player_ids = set()
        player_details = {}
        
        for season in seasons:
            try:
                logger.info(f"  Fetching players for {season}...")
                
                # Get ALL players (not just currently active)
                # is_only_current_season=0 returns all players in history
                all_players_endpoint = commonallplayers.CommonAllPlayers(
                    is_only_current_season=0,  # 0 = all players, we'll filter ourselves
                    league_id='00',
                    season=season
                )
                
                df = all_players_endpoint.get_data_frames()[0]
                
                logger.info(f"    API returned {len(df)} total players")
                
                # CRITICAL: Filter to only players who played in this specific season
                # A player played in season YYYY-YY if:
                # - Their FROM_YEAR is <= YYYY (they started before or during)
                # - Their TO_YEAR is >= YYYY (they were still playing during or after)
                season_year = int(season.split('-')[0])  # e.g., 2023 from "2023-24"
                
                # Convert year columns to integers, handling any format issues
                df['from_year_int'] = pd.to_numeric(df['FROM_YEAR'], errors='coerce').fillna(0).astype(int)
                df['to_year_int'] = pd.to_numeric(df['TO_YEAR'], errors='coerce').fillna(9999).astype(int)
                
                # Filter: player's career must overlap with this season
                df_filtered = df[
                    (df['from_year_int'] <= season_year) &
                    (df['to_year_int'] >= season_year)
                ]
                
                logger.info(f"    Filtered: {len(df)} total -> {len(df_filtered)} who played in {season}")
                
                # Extract player info from filtered data
                for _, row in df_filtered.iterrows():
                    player_id = row['PERSON_ID']
                    all_player_ids.add(player_id)
                    
                    if player_id not in player_details:
                        to_year = row.get('TO_YEAR', '')
                        # Player is active if their TO_YEAR matches current season year
                        is_active = str(to_year) == str(season_year)
                        
                        player_details[player_id] = {
                            'id': player_id,
                            'full_name': row.get('DISPLAY_FIRST_LAST', 'Unknown'),
                            'first_name': row.get('DISPLAY_FIRST_LAST', 'Unknown').split()[0] if ' ' in row.get('DISPLAY_FIRST_LAST', '') else '',
                            'last_name': ' '.join(row.get('DISPLAY_FIRST_LAST', 'Unknown').split()[1:]) if ' ' in row.get('DISPLAY_FIRST_LAST', '') else row.get('DISPLAY_FIRST_LAST', 'Unknown'),
                            'is_active': is_active,
                            'from_year': row.get('FROM_YEAR', ''),
                            'to_year': row.get('TO_YEAR', '')
                        }
                
                logger.info(f"    ✓ Found {len(df_filtered)} players who played in {season}")
                time.sleep(0.6)  # Rate limiting
                
            except Exception as e:
                logger.error(f"    ✗ Failed to fetch players for {season}: {e}")
                logger.exception(e)  # Show full error for debugging
                continue
        
        players_list = list(player_details.values())
        logger.info(f"✓ Total unique players across specified seasons: {len(players_list)}")
        
        return players_list
    
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