"""
NBA Roster Fetcher
Fetches current team rosters using the CommonTeamRoster endpoint

Usage:
    from models.data.roster_fetcher import RosterFetcher
    
    fetcher = RosterFetcher()
    roster = fetcher.get_team_roster(1610612747)  # Lakers
    
    # Or get rosters for a game
    rosters = fetcher.get_game_rosters(home_team_id, away_team_id)
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams as nba_teams


class RosterFetcher:
    """Fetches current team rosters from NBA API"""
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize roster fetcher
        
        Args:
            cache_enabled: Whether to cache roster data (default True)
        """
        self.cache_enabled = cache_enabled
        self.roster_cache: Dict[str, List[Dict]] = {}  # key: "team_id_season"
        self.request_count = 0
        
        # Build team mappings
        self.team_id_to_abbrev = {}
        self.abbrev_to_team_id = {}
        self._build_team_mappings()
    
    def _build_team_mappings(self):
        """Build team ID <-> abbreviation mappings"""
        all_teams = nba_teams.get_teams()
        for team in all_teams:
            self.team_id_to_abbrev[team['id']] = team['abbreviation']
            self.abbrev_to_team_id[team['abbreviation']] = team['id']
    
    def _rate_limit(self):
        """Apply rate limiting to avoid API throttling"""
        self.request_count += 1
        time.sleep(0.6)
        
        # Extra pause every 10 requests
        if self.request_count % 10 == 0:
            time.sleep(2)
    
    def _get_current_season(self) -> str:
        """Get current NBA season string"""
        now = datetime.now()
        year = now.year if now.month >= 10 else now.year - 1
        return f"{year}-{str(year + 1)[-2:]}"
    
    def _get_cache_key(self, team_id: int, season: str) -> str:
        """Generate cache key"""
        return f"{team_id}_{season}"
    
    def get_team_roster(
        self,
        team_id: int,
        season: str = None,
        include_coaches: bool = False
    ) -> List[Dict]:
        """
        Get current roster for a team
        
        Args:
            team_id: NBA team ID
            season: Season string (e.g., '2024-25'). Defaults to current.
            include_coaches: Whether to include coaching staff
            
        Returns:
            List of player dictionaries
        """
        season = season or self._get_current_season()
        cache_key = self._get_cache_key(team_id, season)
        
        # Check cache
        if self.cache_enabled and cache_key in self.roster_cache:
            return self.roster_cache[cache_key]
        
        try:
            self._rate_limit()
            
            roster_endpoint = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season
            )
            
            dfs = roster_endpoint.get_data_frames()
            roster_df = dfs[0]  # CommonTeamRoster dataset
            
            players = []
            for _, row in roster_df.iterrows():
                player_dict = {
                    'player_id': row['PLAYER_ID'],
                    'player_name': row['PLAYER'],
                    'player_slug': row.get('PLAYER_SLUG', ''),
                    'number': row.get('NUM', ''),
                    'position': row.get('POSITION', ''),
                    'height': row.get('HEIGHT', ''),
                    'weight': row.get('WEIGHT', ''),
                    'birth_date': row.get('BIRTH_DATE', ''),
                    'age': row.get('AGE', ''),
                    'experience': row.get('EXP', ''),
                    'school': row.get('SCHOOL', ''),
                    'team_id': team_id,
                    'team_abbrev': self.team_id_to_abbrev.get(team_id, ''),
                    'season': season
                }
                players.append(player_dict)
            
            # Cache result
            if self.cache_enabled:
                self.roster_cache[cache_key] = players
            
            return players
            
        except Exception as e:
            print(f"Error fetching roster for team {team_id}: {e}")
            return []
    
    def get_team_roster_by_abbrev(
        self,
        team_abbrev: str,
        season: str = None
    ) -> List[Dict]:
        """
        Get roster using team abbreviation
        
        Args:
            team_abbrev: Team abbreviation (e.g., 'LAL', 'BOS')
            season: Season string
            
        Returns:
            List of player dictionaries
        """
        team_id = self.abbrev_to_team_id.get(team_abbrev.upper())
        if not team_id:
            print(f"Unknown team abbreviation: {team_abbrev}")
            return []
        
        return self.get_team_roster(team_id, season)
    
    def get_game_rosters(
        self,
        home_team_id: int,
        away_team_id: int,
        season: str = None
    ) -> Dict[str, List[Dict]]:
        """
        Get rosters for both teams in a game
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            season: Season string
            
        Returns:
            Dictionary with 'home' and 'away' roster lists
        """
        return {
            'home': self.get_team_roster(home_team_id, season),
            'away': self.get_team_roster(away_team_id, season)
        }
    
    def get_all_team_rosters(self, season: str = None) -> Dict[int, List[Dict]]:
        """
        Get rosters for all 30 NBA teams
        
        Args:
            season: Season string
            
        Returns:
            Dictionary mapping team_id to roster list
        """
        season = season or self._get_current_season()
        all_rosters = {}
        
        all_teams = nba_teams.get_teams()
        total = len(all_teams)
        
        print(f"Fetching rosters for {total} teams...")
        
        for idx, team in enumerate(all_teams, 1):
            team_id = team['id']
            team_name = team['full_name']
            
            print(f"  [{idx}/{total}] {team_name}...", end='')
            
            roster = self.get_team_roster(team_id, season)
            all_rosters[team_id] = roster
            
            print(f" {len(roster)} players")
        
        return all_rosters
    
    def get_player_ids_for_team(self, team_id: int, season: str = None) -> List[int]:
        """Get just the player IDs for a team"""
        roster = self.get_team_roster(team_id, season)
        return [p['player_id'] for p in roster]
    
    def find_player_team(self, player_id: int, season: str = None) -> Optional[Dict]:
        """
        Find which team a player is currently on
        
        Args:
            player_id: NBA player ID
            season: Season string
            
        Returns:
            Team info dict or None if not found
        """
        all_rosters = self.get_all_team_rosters(season)
        
        for team_id, roster in all_rosters.items():
            for player in roster:
                if player['player_id'] == player_id:
                    return {
                        'team_id': team_id,
                        'team_abbrev': self.team_id_to_abbrev.get(team_id),
                        'player_info': player
                    }
        
        return None
    
    def to_dataframe(self, roster: List[Dict]) -> pd.DataFrame:
        """Convert roster list to DataFrame"""
        return pd.DataFrame(roster)
    
    def clear_cache(self):
        """Clear all cached roster data"""
        self.roster_cache = {}
        print("Roster cache cleared")
    
    def get_coaches(self, team_id: int, season: str = None) -> List[Dict]:
        """
        Get coaching staff for a team
        
        Args:
            team_id: NBA team ID
            season: Season string
            
        Returns:
            List of coach dictionaries
        """
        season = season or self._get_current_season()
        
        try:
            self._rate_limit()
            
            roster_endpoint = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season
            )
            
            dfs = roster_endpoint.get_data_frames()
            coaches_df = dfs[1]  # Coaches dataset
            
            coaches = []
            for _, row in coaches_df.iterrows():
                coach_dict = {
                    'coach_id': row['COACH_ID'],
                    'coach_name': row['COACH_NAME'],
                    'first_name': row['FIRST_NAME'],
                    'last_name': row['LAST_NAME'],
                    'is_assistant': row.get('IS_ASSISTANT', False),
                    'coach_type': row.get('COACH_TYPE', ''),
                    'team_id': team_id,
                    'season': season
                }
                coaches.append(coach_dict)
            
            return coaches
            
        except Exception as e:
            print(f"Error fetching coaches for team {team_id}: {e}")
            return []


# Convenience function
def get_current_roster(team_abbrev: str) -> List[Dict]:
    """
    Quick function to get a team's current roster
    
    Args:
        team_abbrev: Team abbreviation (e.g., 'LAL')
        
    Returns:
        List of player dictionaries
    """
    fetcher = RosterFetcher()
    return fetcher.get_team_roster_by_abbrev(team_abbrev)


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("NBA ROSTER FETCHER DEMO")
    print("=" * 60)
    
    fetcher = RosterFetcher()
    
    # Get Lakers roster
    print("\n🏀 Los Angeles Lakers Roster:")
    print("-" * 40)
    
    roster = fetcher.get_team_roster_by_abbrev('LAL')
    
    for player in roster:
        print(f"  #{player['number']:>2} {player['player_name']:<25} "
              f"{player['position']:<5} {player['height']}")
    
    print(f"\nTotal: {len(roster)} players")
    
    # Get coaches
    print("\n🏀 Lakers Coaching Staff:")
    print("-" * 40)
    
    lakers_id = fetcher.abbrev_to_team_id['LAL']
    coaches = fetcher.get_coaches(lakers_id)
    
    for coach in coaches:
        role = "Assistant" if coach['is_assistant'] else "Head Coach"
        print(f"  {coach['coach_name']:<25} ({role})")