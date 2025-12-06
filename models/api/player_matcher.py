"""
Player and Team ID Matcher
Maps Sports Game Odds IDs to your NBA database IDs

This module handles the challenge of matching players/teams between
the SGO API and your local database which uses NBA API IDs.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import sys
from pathlib import Path
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from etl.config.settings import DatabaseConfig


@dataclass
class PlayerMatch:
    """Result of player matching attempt"""
    sgo_id: str
    sgo_name: str
    db_player_id: Optional[int]
    db_player_name: Optional[str]
    confidence: float  # 0.0 to 1.0
    match_method: str  # 'exact', 'normalized', 'fuzzy'


@dataclass
class TeamMatch:
    """Result of team matching attempt"""
    sgo_id: str
    sgo_name: str
    db_team_id: Optional[int]
    db_team_abbrev: Optional[str]
    confidence: float


class PlayerTeamMatcher:
    """
    Matches SGO player/team IDs to your database IDs
    
    Strategies:
    1. Exact name match
    2. Normalized name match (remove Jr., Sr., accents, etc.)
    3. Fuzzy match with high threshold
    4. Cache successful matches for future lookups
    """
    
    # Common name variations
    NAME_REPLACEMENTS = {
        'PJ': 'P.J.',
        'CJ': 'C.J.',
        'TJ': 'T.J.',
        'JR': 'Jr.',
        'SR': 'Sr.',
        'III': '',
        'II': '',
        'IV': '',
    }
    
    def __init__(self):
        self.conn = None
        self._connect()
        
        # Caches
        self._db_players: Dict[int, str] = {}  # db_id -> name
        self._db_players_normalized: Dict[str, int] = {}  # normalized_name -> db_id
        self._db_teams: Dict[int, str] = {}  # db_id -> abbrev
        self._db_teams_by_name: Dict[str, int] = {}  # various names -> db_id
        
        # Match caches (persist successful matches)
        self._player_match_cache: Dict[str, int] = {}  # sgo_id -> db_id
        self._team_match_cache: Dict[str, int] = {}  # sgo_id -> db_id
        
        self._load_db_data()
    
    def _connect(self):
        """Establish database connection using psycopg2 directly"""
        try:
            self.conn = psycopg2.connect(
                host=DatabaseConfig.HOST,
                port=DatabaseConfig.PORT,
                dbname=DatabaseConfig.NAME,
                user=DatabaseConfig.USER,
                password=DatabaseConfig.PASSWORD,
                sslmode=DatabaseConfig.SSLMODE
            )
            print(f"✓ PlayerTeamMatcher connected to database")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            self.conn = None
    
    def _load_db_data(self):
        """Load players and teams from database"""
        if not self.conn:
            print("⚠️ No database connection, skipping data load")
            return
            
        try:
            with self.conn.cursor() as cur:
                # Load players - check which columns exist
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'players'
                """)
                player_cols = [row[0] for row in cur.fetchall()]
                
                # Determine name column
                if 'full_name' in player_cols:
                    name_col = 'full_name'
                elif 'display_first_last' in player_cols:
                    name_col = 'display_first_last'
                elif 'name' in player_cols:
                    name_col = 'name'
                else:
                    # Try first_name + last_name
                    name_col = "first_name || ' ' || last_name"
                
                cur.execute(f"""
                    SELECT player_id, {name_col} 
                    FROM players 
                    WHERE {name_col.split('||')[0].strip()} IS NOT NULL
                """)
                for row in cur.fetchall():
                    player_id, name = row
                    if name:
                        self._db_players[player_id] = name
                        normalized = self._normalize_name(name)
                        self._db_players_normalized[normalized] = player_id
                
                # Load teams - check which columns exist
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'teams'
                """)
                team_cols = [row[0] for row in cur.fetchall()]
                
                # Build dynamic query based on available columns
                select_cols = ['team_id']
                if 'abbreviation' in team_cols:
                    select_cols.append('abbreviation')
                else:
                    select_cols.append('NULL as abbreviation')
                    
                if 'full_name' in team_cols:
                    select_cols.append('full_name')
                elif 'name' in team_cols:
                    select_cols.append('name as full_name')
                else:
                    select_cols.append('NULL as full_name')
                    
                if 'nickname' in team_cols:
                    select_cols.append('nickname')
                else:
                    select_cols.append('NULL as nickname')
                    
                if 'city' in team_cols:
                    select_cols.append('city')
                else:
                    select_cols.append('NULL as city')
                
                cur.execute(f"SELECT {', '.join(select_cols)} FROM teams")
                
                for row in cur.fetchall():
                    team_id, abbrev, full_name, nickname, city = row
                    if abbrev:
                        self._db_teams[team_id] = abbrev
                    
                    # Map various name formats
                    if abbrev:
                        self._db_teams_by_name[abbrev.upper()] = team_id
                    if full_name:
                        self._db_teams_by_name[full_name.lower()] = team_id
                    if nickname:
                        self._db_teams_by_name[nickname.lower()] = team_id
                    if city and nickname:
                        self._db_teams_by_name[f"{city} {nickname}".lower()] = team_id
            
            print(f"  Loaded {len(self._db_players)} players and {len(self._db_teams)} teams from DB")
            
        except Exception as e:
            print(f"Error loading DB data: {e}")
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize player name for matching"""
        if not name:
            return ""
        
        # Lowercase
        name = name.lower().strip()
        
        # Remove suffixes
        name = re.sub(r'\s+(jr\.?|sr\.?|iii|ii|iv)$', '', name, flags=re.IGNORECASE)
        
        # Remove periods from initials
        name = re.sub(r'\.', '', name)
        
        # Remove accents (simple version)
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ñ': 'n', 'ü': 'u', 'ö': 'o', 'ä': 'a', 'ć': 'c',
            'č': 'c', 'š': 's', 'ž': 'z', 'đ': 'd', 'ğ': 'g',
            'ı': 'i', 'ş': 's', 'ț': 't', 'ă': 'a',
        }
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        return name
    
    @staticmethod
    def _similarity(s1: str, s2: str) -> float:
        """Calculate string similarity ratio"""
        return SequenceMatcher(None, s1, s2).ratio()
    
    def match_player(
        self, 
        sgo_id: str, 
        sgo_name: str,
        min_confidence: float = 0.85
    ) -> PlayerMatch:
        """
        Match SGO player to database player
        
        Returns PlayerMatch with db_player_id (or None if no match)
        """
        # Check cache first
        if sgo_id in self._player_match_cache:
            db_id = self._player_match_cache[sgo_id]
            return PlayerMatch(
                sgo_id=sgo_id,
                sgo_name=sgo_name,
                db_player_id=db_id,
                db_player_name=self._db_players.get(db_id),
                confidence=1.0,
                match_method='cached'
            )
        
        # Normalize SGO name
        normalized_sgo = self._normalize_name(sgo_name)
        
        # Try exact normalized match
        if normalized_sgo in self._db_players_normalized:
            db_id = self._db_players_normalized[normalized_sgo]
            self._player_match_cache[sgo_id] = db_id
            return PlayerMatch(
                sgo_id=sgo_id,
                sgo_name=sgo_name,
                db_player_id=db_id,
                db_player_name=self._db_players.get(db_id),
                confidence=1.0,
                match_method='normalized'
            )
        
        # Try fuzzy match
        best_match = None
        best_score = 0.0
        best_db_id = None
        
        for db_id, db_name in self._db_players.items():
            normalized_db = self._normalize_name(db_name)
            score = self._similarity(normalized_sgo, normalized_db)
            
            if score > best_score:
                best_score = score
                best_match = db_name
                best_db_id = db_id
        
        if best_score >= min_confidence:
            self._player_match_cache[sgo_id] = best_db_id
            return PlayerMatch(
                sgo_id=sgo_id,
                sgo_name=sgo_name,
                db_player_id=best_db_id,
                db_player_name=best_match,
                confidence=best_score,
                match_method='fuzzy'
            )
        
        # No match found
        return PlayerMatch(
            sgo_id=sgo_id,
            sgo_name=sgo_name,
            db_player_id=None,
            db_player_name=None,
            confidence=best_score,
            match_method='no_match'
        )
    
    def match_team(
        self, 
        sgo_id: str, 
        sgo_name: str
    ) -> TeamMatch:
        """
        Match SGO team to database team
        """
        # Check cache
        if sgo_id in self._team_match_cache:
            db_id = self._team_match_cache[sgo_id]
            return TeamMatch(
                sgo_id=sgo_id,
                sgo_name=sgo_name,
                db_team_id=db_id,
                db_team_abbrev=self._db_teams.get(db_id),
                confidence=1.0
            )
        
        # Try various name formats
        name_lower = sgo_name.lower().strip()
        
        # Check direct match
        if name_lower in self._db_teams_by_name:
            db_id = self._db_teams_by_name[name_lower]
            self._team_match_cache[sgo_id] = db_id
            return TeamMatch(
                sgo_id=sgo_id,
                sgo_name=sgo_name,
                db_team_id=db_id,
                db_team_abbrev=self._db_teams.get(db_id),
                confidence=1.0
            )
        
        # Try abbreviation (if short)
        if len(sgo_name) <= 4:
            upper = sgo_name.upper()
            if upper in self._db_teams_by_name:
                db_id = self._db_teams_by_name[upper]
                self._team_match_cache[sgo_id] = db_id
                return TeamMatch(
                    sgo_id=sgo_id,
                    sgo_name=sgo_name,
                    db_team_id=db_id,
                    db_team_abbrev=self._db_teams.get(db_id),
                    confidence=1.0
                )
        
        # Try partial match (city or nickname)
        for key, db_id in self._db_teams_by_name.items():
            if name_lower in key or key in name_lower:
                self._team_match_cache[sgo_id] = db_id
                return TeamMatch(
                    sgo_id=sgo_id,
                    sgo_name=sgo_name,
                    db_team_id=db_id,
                    db_team_abbrev=self._db_teams.get(db_id),
                    confidence=0.9
                )
        
        # No match
        return TeamMatch(
            sgo_id=sgo_id,
            sgo_name=sgo_name,
            db_team_id=None,
            db_team_abbrev=None,
            confidence=0.0
        )
    
    def match_players_batch(
        self, 
        sgo_players: List[Dict]
    ) -> Dict[str, PlayerMatch]:
        """
        Match a batch of SGO players to database
        
        Args:
            sgo_players: List of dicts with 'playerID' and 'name' keys
            
        Returns:
            Dict mapping sgo_id -> PlayerMatch
        """
        results = {}
        
        for player in sgo_players:
            sgo_id = player.get('playerID', '')
            sgo_name = player.get('name', '')
            
            if sgo_id:
                results[sgo_id] = self.match_player(sgo_id, sgo_name)
        
        # Stats
        matched = sum(1 for m in results.values() if m.db_player_id)
        print(f"Matched {matched}/{len(results)} players")
        
        return results
    
    def get_db_player_id(self, sgo_id: str, sgo_name: str = '') -> Optional[int]:
        """Quick lookup - returns DB player ID or None"""
        match = self.match_player(sgo_id, sgo_name)
        return match.db_player_id
    
    def get_db_team_id(self, sgo_id: str, sgo_name: str = '') -> Optional[int]:
        """Quick lookup - returns DB team ID or None"""
        match = self.match_team(sgo_id, sgo_name)
        return match.db_team_id
    
    def save_match_cache(self, filepath: str = 'sgo_match_cache.json'):
        """Save match caches to file for persistence"""
        import json
        
        cache = {
            'players': self._player_match_cache,
            'teams': self._team_match_cache
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"Saved {len(self._player_match_cache)} player and {len(self._team_match_cache)} team matches")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def load_match_cache(self, filepath: str = 'sgo_match_cache.json'):
        """Load match caches from file"""
        import json
        from pathlib import Path
        
        if not Path(filepath).exists():
            return
        
        try:
            with open(filepath, 'r') as f:
                cache = json.load(f)
            
            self._player_match_cache = {k: int(v) for k, v in cache.get('players', {}).items()}
            self._team_match_cache = {k: int(v) for k, v in cache.get('teams', {}).items()}
            
            print(f"Loaded {len(self._player_match_cache)} player and {len(self._team_match_cache)} team matches from cache")
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    def close(self):
        if self.conn:
            self.conn.close()


# Example usage
if __name__ == '__main__':
    matcher = PlayerTeamMatcher()
    
    # Test player match
    match = matcher.match_player('SGO123', 'LeBron James')
    print(f"LeBron match: {match}")
    
    # Test team match  
    team_match = matcher.match_team('SGO_LAL', 'Los Angeles Lakers')
    print(f"Lakers match: {team_match}")
    
    matcher.close()