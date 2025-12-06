"""
Sports Game Odds API Client for NBA Odds Integration
Fetches player props and game lines from api.sportsgameodds.com/v2

Based on Sports Game Odds API documentation
"""

import requests
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import re


@dataclass
class PlayerPropLine:
    """Player prop betting line"""
    player_name: str
    player_id: Optional[str]  # SGO player ID (string format)
    prop_type: str  # pts, reb, ast, etc.
    line: float
    over_odds: float  # Decimal odds
    under_odds: float
    over_odds_american: int
    under_odds_american: int
    bookmaker: str
    last_update: datetime
    game_id: Optional[str] = None
    team_abbrev: Optional[str] = None
    odd_id: Optional[str] = None  # SGO odd ID string


@dataclass
class GameLine:
    """Game betting lines (spread, total, moneyline)"""
    game_id: str
    game_date: date
    home_team: str
    away_team: str
    home_team_id: Optional[str] = None  # SGO team ID
    away_team_id: Optional[str] = None
    
    # Spread (handicap)
    spread_line: Optional[float] = None
    spread_home_odds: Optional[float] = None
    spread_away_odds: Optional[float] = None
    spread_home_american: Optional[int] = None
    spread_away_american: Optional[int] = None
    
    # Total (over/under)
    total_line: Optional[float] = None
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    over_american: Optional[int] = None
    under_american: Optional[int] = None
    
    # Moneyline
    home_ml: Optional[float] = None
    away_ml: Optional[float] = None
    home_ml_american: Optional[int] = None
    away_ml_american: Optional[int] = None
    
    bookmaker: str = ""
    last_update: Optional[datetime] = None


class SportsGameOddsClient:
    """
    Client for Sports Game Odds API (api.sportsgameodds.com/v2)
    """
    
    BASE_URL = "https://api.sportsgameodds.com/v2"
    
    # Stat ID mappings for player props
    STAT_ID_MAP = {
        'points': 'pts',
        'rebounds': 'reb',
        'assists': 'ast',
        'steals': 'stl',
        'blocks': 'blk',
        'turnovers': 'tov',
        'threePointersMade': 'fg3m',
        'points+rebounds+assists': 'pra',
        'points+rebounds': 'pr',
        'points+assists': 'pa',
        'rebounds+assists': 'ra',
        'blocks+steals': 'bs',
    }
    
    # Reverse mapping
    PROP_TO_STAT = {v: k for k, v in STAT_ID_MAP.items()}
    
    # Team abbreviation mappings (SGO uses full names, we need abbrevs)
    TEAM_NAME_MAP = {
        'Los Angeles Lakers': 'LAL', 'Boston Celtics': 'BOS',
        'Golden State Warriors': 'GSW', 'Phoenix Suns': 'PHX',
        'Milwaukee Bucks': 'MIL', 'Miami Heat': 'MIA',
        'Philadelphia 76ers': 'PHI', 'Denver Nuggets': 'DEN',
        'Memphis Grizzlies': 'MEM', 'Cleveland Cavaliers': 'CLE',
        'Sacramento Kings': 'SAC', 'Brooklyn Nets': 'BKN',
        'New York Knicks': 'NYK', 'Dallas Mavericks': 'DAL',
        'Atlanta Hawks': 'ATL', 'Toronto Raptors': 'TOR',
        'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
        'LA Clippers': 'LAC', 'Chicago Bulls': 'CHI',
        'Utah Jazz': 'UTA', 'Oklahoma City Thunder': 'OKC',
        'Indiana Pacers': 'IND', 'Portland Trail Blazers': 'POR',
        'Orlando Magic': 'ORL', 'Charlotte Hornets': 'CHA',
        'Houston Rockets': 'HOU', 'San Antonio Spurs': 'SAS',
        'Detroit Pistons': 'DET', 'Washington Wizards': 'WAS',
        # Also map abbreviations to themselves
        'LAL': 'LAL', 'BOS': 'BOS', 'GSW': 'GSW', 'PHX': 'PHX',
        'MIL': 'MIL', 'MIA': 'MIA', 'PHI': 'PHI', 'DEN': 'DEN',
        'MEM': 'MEM', 'CLE': 'CLE', 'SAC': 'SAC', 'BKN': 'BKN',
        'NYK': 'NYK', 'DAL': 'DAL', 'ATL': 'ATL', 'TOR': 'TOR',
        'MIN': 'MIN', 'NOP': 'NOP', 'LAC': 'LAC', 'CHI': 'CHI',
        'UTA': 'UTA', 'OKC': 'OKC', 'IND': 'IND', 'POR': 'POR',
        'ORL': 'ORL', 'CHA': 'CHA', 'HOU': 'HOU', 'SAS': 'SAS',
        'DET': 'DET', 'WAS': 'WAS',
    }
    
    def __init__(self, api_key: str, preferred_bookmaker: str = 'fanduel'):
        """
        Initialize Sports Game Odds client
        
        Args:
            api_key: Your SGO API key
            preferred_bookmaker: Bookmaker ID (fanduel, draftkings, etc.)
        """
        self.api_key = api_key
        self.preferred_bookmaker = preferred_bookmaker
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.request_count = 0
        self._last_request_time = 0
        
        # Caches for player/team ID mapping
        self._players_cache: Dict[str, Dict] = {}  # SGO player_id -> player info
        self._teams_cache: Dict[str, Dict] = {}    # SGO team_id -> team info
        self._player_name_to_id: Dict[str, str] = {}  # normalized name -> SGO ID
    
    def _rate_limit(self, cooldown_ms: int = 7000):
        """
        Apply rate limiting - Amateur plan: 10 requests/minute
        Default 7 second cooldown to stay safe (allows ~8.5 req/min)
        """
        cooldown_sec = cooldown_ms / 1000.0
        elapsed = time.time() - self._last_request_time
        if elapsed < cooldown_sec:
            wait_time = cooldown_sec - elapsed
            print(f"    ⏳ Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        self._last_request_time = time.time()
        self.request_count += 1
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Dict = None, 
        cooldown_ms: int = 7000
    ) -> Optional[Any]:
        """Make API request with error handling"""
        self._rate_limit(cooldown_ms)
        
        url = f"{self.BASE_URL}{endpoint}"
        
        if params is None:
            params = {}
        params['apiKey'] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                print(f"⚠️ Rate limited! Waiting 60 seconds...")
                time.sleep(60)
                # Retry once
                response = self.session.get(url, params=params, timeout=30)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if response is not None:
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.text[:500]}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return None
    
    def _fetch_all_paginated(
        self, 
        endpoint: str, 
        params: Dict, 
        max_pages: int = 10
    ) -> List[Dict]:
        """Fetch all results using cursor pagination"""
        all_results = []
        cursor = None
        page = 0
        
        while page < max_pages:
            if cursor:
                params['cursor'] = cursor
            
            result = self._make_request(endpoint, params)
            
            if not result:
                break
            
            # Handle different response structures
            if 'events' in result:
                all_results.extend(result['events'])
            elif 'teams' in result:
                all_results.extend(result['teams'])
            elif 'players' in result:
                all_results.extend(result['players'])
            elif isinstance(result, list):
                all_results.extend(result)
            
            cursor = result.get('nextCursor')
            if not cursor:
                break
            
            page += 1
        
        return all_results
    
    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American odds"""
        if decimal_odds is None:
            return 0
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return 1 + (american_odds / 100)
        else:
            return 1 + (100 / abs(american_odds))
    
    @staticmethod
    def normalize_player_name(name: str) -> str:
        """Normalize player name for matching"""
        if not name:
            return ""
        # Remove Jr., Sr., III, etc.
        name = re.sub(r'\s+(Jr\.?|Sr\.?|III|II|IV)$', '', name, flags=re.IGNORECASE)
        # Lowercase and strip
        return name.lower().strip()
    
    # ==========================================
    # Teams Endpoint
    # ==========================================
    
    def get_teams(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Fetch NBA teams from SGO
        
        Returns dict mapping team_id -> team info
        """
        if self._teams_cache and not force_refresh:
            return self._teams_cache
        
        params = {
            'leagueID': 'NBA',
            'limit': 50
        }
        
        teams = self._fetch_all_paginated('/teams', params, max_pages=2)
        
        for team in teams:
            team_id = team.get('teamID')
            if team_id:
                self._teams_cache[team_id] = team
        
        print(f"Loaded {len(self._teams_cache)} NBA teams")
        return self._teams_cache
    
    # ==========================================
    # Players Endpoint  
    # ==========================================
    
    def get_players(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Fetch NBA players from SGO
        
        Returns dict mapping player_id -> player info
        """
        if self._players_cache and not force_refresh:
            return self._players_cache
        
        params = {
            'leagueID': 'NBA',
            'limit': 100
        }
        
        players = self._fetch_all_paginated('/players', params, max_pages=10)
        
        for player in players:
            player_id = player.get('playerID')
            if player_id:
                self._players_cache[player_id] = player
                # Build name lookup
                name = player.get('name', '')
                if name:
                    normalized = self.normalize_player_name(name)
                    self._player_name_to_id[normalized] = player_id
        
        print(f"Loaded {len(self._players_cache)} NBA players")
        return self._players_cache
    
    def find_player_by_name(self, name: str) -> Optional[Dict]:
        """Find player by name (fuzzy match)"""
        if not self._players_cache:
            self.get_players()
        
        normalized = self.normalize_player_name(name)
        
        # Exact match
        if normalized in self._player_name_to_id:
            player_id = self._player_name_to_id[normalized]
            return self._players_cache.get(player_id)
        
        # Partial match
        for cached_name, player_id in self._player_name_to_id.items():
            if normalized in cached_name or cached_name in normalized:
                return self._players_cache.get(player_id)
        
        return None
    
    # ==========================================
    # Events (Games) Endpoint
    # ==========================================
    
    def get_nba_events(
        self,
        from_date: date = None,
        to_date: date = None,
        include_odds: bool = True,
        event_status: str = None  # 'scheduled', 'live', 'final'
    ) -> List[Dict]:
        """
        Get NBA events/games
        
        Args:
            from_date: Start date filter
            to_date: End date filter  
            include_odds: Whether to include odds in response
            event_status: Filter by status
        """
        params = {
            'leagueID': 'NBA',
            'limit': 50
        }
        
        # Use lowercase string 'true'/'false' for boolean params
        if include_odds:
            params['marketOddsAvailable'] = 'true'
        
        if event_status:
            params['eventStatus'] = event_status
        
        events = self._fetch_all_paginated('/events', params, max_pages=5)
        
        # Filter by date if specified
        if from_date or to_date:
            filtered = []
            for event in events:
                event_time = event.get('startTime')
                if event_time:
                    try:
                        event_date = datetime.fromisoformat(
                            event_time.replace('Z', '+00:00')
                        ).date()
                        
                        if from_date and event_date < from_date:
                            continue
                        if to_date and event_date > to_date:
                            continue
                        
                        filtered.append(event)
                    except:
                        filtered.append(event)
                else:
                    filtered.append(event)
            return filtered
        
        return events
    
    def get_nba_fixtures(
        self,
        from_date: date = None,
        to_date: date = None,
        has_odds: bool = True
    ) -> List[Dict]:
        """Alias for get_nba_events for compatibility"""
        return self.get_nba_events(from_date, to_date, has_odds)
    
    # ==========================================
    # Odds Endpoint
    # ==========================================
    
    def get_event_odds(
        self,
        event_id: str,
        stat_ids: List[str] = None,
        bookmakers: List[str] = None
    ) -> Optional[Dict]:
        """
        Get odds for a specific event
        
        Args:
            event_id: SGO event ID
            stat_ids: Filter by stat types (points, rebounds, etc.)
            bookmakers: Filter by bookmaker IDs
        """
        params = {
            'eventID': event_id,
            'limit': 100
        }
        
        if bookmakers:
            params['sportsbooks'] = ','.join(bookmakers)
        
        result = self._make_request('/odds', params)
        return result
    
    def get_game_lines(
        self, 
        event_id: str, 
        bookmaker: str = None
    ) -> Optional[GameLine]:
        """
        Get game lines (spread, total, moneyline) for an event
        """
        bookmaker = bookmaker or self.preferred_bookmaker
        
        odds_data = self.get_event_odds(event_id, bookmakers=[bookmaker])
        
        if not odds_data:
            return None
        
        return self._parse_game_lines(odds_data, event_id, bookmaker)
    
    def get_player_props(
        self, 
        event_id: str, 
        bookmaker: str = None
    ) -> List[PlayerPropLine]:
        """
        Get player prop lines for an event
        """
        bookmaker = bookmaker or self.preferred_bookmaker
        
        # Ensure players are loaded for name resolution
        if not self._players_cache:
            self.get_players()
        
        odds_data = self.get_event_odds(event_id, bookmakers=[bookmaker])
        
        if not odds_data:
            return []
        
        return self._parse_player_props(odds_data, event_id, bookmaker)
    
    # ==========================================
    # Parsing Helpers
    # ==========================================
    
    def _parse_game_lines(
        self, 
        odds_data: Dict, 
        event_id: str,
        bookmaker: str
    ) -> Optional[GameLine]:
        """Parse game lines from odds response"""
        try:
            # Get event info
            event = odds_data.get('event', {})
            
            home_team = event.get('homeTeam', {})
            away_team = event.get('awayTeam', {})
            
            game_line = GameLine(
                game_id=event_id,
                game_date=date.today(),
                home_team=home_team.get('name', ''),
                away_team=away_team.get('name', ''),
                home_team_id=home_team.get('teamID'),
                away_team_id=away_team.get('teamID'),
                bookmaker=bookmaker
            )
            
            # Parse start time
            start_time = event.get('startTime')
            if start_time:
                try:
                    game_line.game_date = datetime.fromisoformat(
                        start_time.replace('Z', '+00:00')
                    ).date()
                except:
                    pass
            
            # Parse odds - odds_data['odds'] is dict of oddID -> odd object
            odds = odds_data.get('odds', {})
            
            for odd_id, odd_obj in odds.items():
                # Parse oddID format: statID-entityID-periodID-betTypeID-outcomeID
                # e.g., "points-home-game-ml-home"
                
                parts = odd_id.split('-')
                if len(parts) < 5:
                    continue
                
                stat_id = parts[0]
                entity_id = parts[1]
                period_id = parts[2]
                bet_type = parts[3]
                outcome = parts[4] if len(parts) > 4 else ''
                
                # Only care about full game for now
                if period_id != 'game':
                    continue
                
                # Get odds value (use closeOdds or currentOdds)
                odds_value = odd_obj.get('closeOdds') or odd_obj.get('odds')
                line_value = odd_obj.get('line')
                
                if not odds_value:
                    continue
                
                american_odds = int(odds_value) if isinstance(odds_value, (int, float)) else 0
                
                # Moneyline
                if bet_type == 'ml' and stat_id == 'points':
                    if entity_id == 'home' or outcome == 'home':
                        game_line.home_ml_american = american_odds
                        game_line.home_ml = self.american_to_decimal(american_odds)
                    elif entity_id == 'away' or outcome == 'away':
                        game_line.away_ml_american = american_odds
                        game_line.away_ml = self.american_to_decimal(american_odds)
                
                # Spread
                elif bet_type == 'sp' and stat_id == 'points':
                    if entity_id == 'home' or outcome == 'home':
                        game_line.spread_line = line_value
                        game_line.spread_home_american = american_odds
                        game_line.spread_home_odds = self.american_to_decimal(american_odds)
                    elif entity_id == 'away' or outcome == 'away':
                        game_line.spread_away_american = american_odds
                        game_line.spread_away_odds = self.american_to_decimal(american_odds)
                
                # Total (over/under)
                elif bet_type == 'ou' and stat_id == 'points' and entity_id == 'all':
                    if 'over' in outcome:
                        game_line.total_line = line_value
                        game_line.over_american = american_odds
                        game_line.over_odds = self.american_to_decimal(american_odds)
                    elif 'under' in outcome:
                        game_line.under_american = american_odds
                        game_line.under_odds = self.american_to_decimal(american_odds)
            
            game_line.last_update = datetime.now()
            return game_line
            
        except Exception as e:
            print(f"Error parsing game lines: {e}")
            return None
    
    def _parse_player_props(
        self, 
        odds_data: Dict, 
        event_id: str,
        bookmaker: str
    ) -> List[PlayerPropLine]:
        """Parse player props from odds response"""
        props = []
        
        try:
            odds = odds_data.get('odds', {})
            
            # Group odds by player and stat type
            player_odds: Dict[str, Dict[str, Dict]] = {}
            
            for odd_id, odd_obj in odds.items():
                # Parse oddID: statID-playerID-periodID-betTypeID-outcomeID
                # e.g., "points-PLAYER123-game-ou-over"
                
                parts = odd_id.split('-')
                if len(parts) < 5:
                    continue
                
                stat_id = parts[0]
                entity_id = parts[1]
                period_id = parts[2]
                bet_type = parts[3]
                outcome = '-'.join(parts[4:]) if len(parts) > 4 else ''
                
                # Only player props (not home/away/all)
                if entity_id in ('home', 'away', 'all'):
                    continue
                
                # Only full game over/under
                if period_id != 'game' or bet_type != 'ou':
                    continue
                
                # Map stat to our prop type
                prop_type = self.STAT_ID_MAP.get(stat_id)
                if not prop_type:
                    continue
                
                player_id = entity_id
                
                if player_id not in player_odds:
                    player_odds[player_id] = {}
                if prop_type not in player_odds[player_id]:
                    player_odds[player_id][prop_type] = {
                        'over_odds': None,
                        'under_odds': None,
                        'line': None,
                        'odd_id': None
                    }
                
                odds_value = odd_obj.get('closeOdds') or odd_obj.get('odds')
                line_value = odd_obj.get('line')
                
                if 'over' in outcome:
                    player_odds[player_id][prop_type]['over_odds'] = odds_value
                    player_odds[player_id][prop_type]['line'] = line_value
                    player_odds[player_id][prop_type]['odd_id'] = odd_id
                elif 'under' in outcome:
                    player_odds[player_id][prop_type]['under_odds'] = odds_value
                    if line_value and not player_odds[player_id][prop_type]['line']:
                        player_odds[player_id][prop_type]['line'] = line_value
            
            # Build PlayerPropLine objects
            for player_id, prop_types in player_odds.items():
                # Get player name from cache
                player_info = self._players_cache.get(player_id, {})
                player_name = player_info.get('name', f'Player {player_id}')
                team_info = player_info.get('team', {})
                team_abbrev = self.get_team_abbrev(team_info.get('name', ''))
                
                for prop_type, line_data in prop_types.items():
                    if line_data['line'] is None:
                        continue
                    
                    over_odds = line_data['over_odds'] or -110
                    under_odds = line_data['under_odds'] or -110
                    
                    props.append(PlayerPropLine(
                        player_name=player_name,
                        player_id=player_id,
                        prop_type=prop_type,
                        line=float(line_data['line']),
                        over_odds=self.american_to_decimal(over_odds),
                        under_odds=self.american_to_decimal(under_odds),
                        over_odds_american=int(over_odds),
                        under_odds_american=int(under_odds),
                        bookmaker=bookmaker,
                        last_update=datetime.now(),
                        game_id=event_id,
                        team_abbrev=team_abbrev,
                        odd_id=line_data['odd_id']
                    ))
        
        except Exception as e:
            print(f"Error parsing player props: {e}")
        
        return props
    
    def get_team_abbrev(self, team_name: str) -> str:
        """Convert full team name to abbreviation"""
        if not team_name:
            return 'UNK'
        return self.TEAM_NAME_MAP.get(team_name, team_name[:3].upper())
    
    # ==========================================
    # Historical Odds (if available)
    # ==========================================
    
    def get_historical_odds(
        self,
        event_id: str,
        bookmakers: List[str] = None
    ) -> Optional[Dict]:
        """
        Get historical odds - Note: May require higher tier plan
        For now, just returns current odds
        """
        return self.get_event_odds(event_id, bookmakers=bookmakers)
    
    def close(self):
        """Close the session"""
        self.session.close()


# Alias for backward compatibility
OddsPAPIClient = SportsGameOddsClient