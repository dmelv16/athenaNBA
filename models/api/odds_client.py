"""
OddsPAPI Client for NBA Odds Integration
Fetches player props and game lines from api.oddspapi.io

Based on OddsPAPI v4 documentation
"""

import requests
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class PlayerPropLine:
    """Player prop betting line"""
    player_name: str
    player_id: Optional[int]
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
    market_id: Optional[int] = None
    outcome_id: Optional[int] = None


@dataclass
class GameLine:
    """Game betting lines (spread, total, moneyline)"""
    game_id: str
    game_date: date
    home_team: str
    away_team: str
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None
    
    # Spread (handicap)
    spread_line: Optional[float] = None
    spread_home_odds: Optional[float] = None  # Decimal
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
    home_ml: Optional[float] = None  # Decimal
    away_ml: Optional[float] = None
    home_ml_american: Optional[int] = None
    away_ml_american: Optional[int] = None
    
    bookmaker: str = ""
    last_update: Optional[datetime] = None


class OddsPAPIClient:
    """
    Client for OddsPAPI (api.oddspapi.io) v4
    """
    
    BASE_URL = "https://api.oddspapi.io"
    
    # Sport IDs - need to fetch from /v4/sports but basketball is typically 11
    SPORT_ID_BASKETBALL = 11
    
    # NBA Tournament ID - will be fetched dynamically
    NBA_TOURNAMENT_ID = None
    
    # Market IDs for NBA (from /v4/markets endpoint)
    # These are examples - actual IDs need to be fetched
    MARKET_MONEYLINE = None  # Will map from markets endpoint
    MARKET_SPREAD = None
    MARKET_TOTAL = None
    
    # Player prop market mappings (marketId -> our prop_type)
    # Will be populated from /v4/markets where playerProp=true
    PLAYER_PROP_MARKETS: Dict[int, str] = {}
    
    # Team name to abbreviation mapping
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
    }
    
    def __init__(self, api_key: str, preferred_bookmaker: str = 'pinnacle'):
        """
        Initialize OddsPAPI client
        
        Args:
            api_key: Your OddsPAPI API key
            preferred_bookmaker: Bookmaker slug (pinnacle, bet365, etc.)
        """
        self.api_key = api_key
        self.preferred_bookmaker = preferred_bookmaker
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.request_count = 0
        self._last_request_time = 0
        
        # Caches
        self._sports_cache: Dict[str, int] = {}
        self._tournaments_cache: Dict[str, int] = {}
        self._markets_cache: Dict[int, Dict] = {}
        self._participants_cache: Dict[int, str] = {}
        self._bookmakers_cache: List[str] = []
    
    def _rate_limit(self, cooldown_ms: int = 500):
        """Apply rate limiting between requests"""
        cooldown_sec = cooldown_ms / 1000.0
        elapsed = time.time() - self._last_request_time
        if elapsed < cooldown_sec:
            time.sleep(cooldown_sec - elapsed)
        self._last_request_time = time.time()
        self.request_count += 1
    
    def _make_request(self, endpoint: str, params: Dict = None, cooldown_ms: int = 500) -> Optional[Any]:
        """Make API request with error handling"""
        self._rate_limit(cooldown_ms)
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if response:
                print(f"Response: {response.text[:500]}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return None
    
    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American odds"""
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
    
    # ==========================================
    # Reference Data Endpoints
    # ==========================================
    
    def get_sports(self) -> Dict[str, int]:
        """Get available sports and their IDs"""
        if self._sports_cache:
            return self._sports_cache
        
        result = self._make_request('/v4/sports', {'language': 'en'}, cooldown_ms=1000)
        
        if result:
            for sport in result:
                self._sports_cache[sport['slug']] = sport['sportId']
            print(f"Loaded {len(self._sports_cache)} sports")
        
        return self._sports_cache
    
    def get_bookmakers(self) -> List[str]:
        """Get available bookmakers"""
        if self._bookmakers_cache:
            return self._bookmakers_cache
        
        result = self._make_request('/v4/bookmakers', cooldown_ms=1000)
        
        if result:
            self._bookmakers_cache = [b['slug'] for b in result]
            print(f"Loaded {len(self._bookmakers_cache)} bookmakers")
        
        return self._bookmakers_cache
    
    def get_markets(self, sport_id: int = None) -> Dict[int, Dict]:
        """
        Get available markets
        
        Returns dict mapping marketId -> market info
        """
        if self._markets_cache:
            return self._markets_cache
        
        result = self._make_request('/v4/markets', {'language': 'en'}, cooldown_ms=1000)
        
        if result:
            for market in result:
                mid = market['marketId']
                self._markets_cache[mid] = market
                
                # Build player prop mapping
                if market.get('playerProp'):
                    name = market['marketName'].lower()
                    if 'points' in name:
                        self.PLAYER_PROP_MARKETS[mid] = 'pts'
                    elif 'rebound' in name:
                        self.PLAYER_PROP_MARKETS[mid] = 'reb'
                    elif 'assist' in name:
                        self.PLAYER_PROP_MARKETS[mid] = 'ast'
                    elif 'steal' in name:
                        self.PLAYER_PROP_MARKETS[mid] = 'stl'
                    elif 'block' in name:
                        self.PLAYER_PROP_MARKETS[mid] = 'blk'
                    elif 'three' in name or '3-point' in name:
                        self.PLAYER_PROP_MARKETS[mid] = 'fg3m'
                    elif 'turnover' in name:
                        self.PLAYER_PROP_MARKETS[mid] = 'tov'
            
            print(f"Loaded {len(self._markets_cache)} markets, {len(self.PLAYER_PROP_MARKETS)} player props")
        
        return self._markets_cache
    
    def get_participants(self, sport_id: int = 11) -> Dict[int, str]:
        """Get participants (teams) for a sport"""
        cache_key = f"sport_{sport_id}"
        if cache_key in str(self._participants_cache):
            return self._participants_cache
        
        result = self._make_request('/v4/participants', {'sportId': sport_id, 'language': 'en'}, cooldown_ms=1000)
        
        if result:
            # Result is {participantId: name, ...}
            self._participants_cache = {int(k): v for k, v in result.items()}
            print(f"Loaded {len(self._participants_cache)} participants for sport {sport_id}")
        
        return self._participants_cache
    
    # ==========================================
    # Fixtures (Games) Endpoint
    # ==========================================
    
    def get_nba_fixtures(
        self,
        from_date: date = None,
        to_date: date = None,
        tournament_id: int = None,
        has_odds: bool = True
    ) -> List[Dict]:
        """
        Get NBA fixtures/games
        
        Args:
            from_date: Start date
            to_date: End date (max 10 days from start with sportId)
            tournament_id: NBA tournament ID (if known)
            has_odds: Only return fixtures with odds
        """
        # Get basketball sport ID
        sports = self.get_sports()
        sport_id = sports.get('basketball', 11)
        
        from_date = from_date or date.today()
        to_date = to_date or (from_date + timedelta(days=1))
        
        # Ensure max 10 days apart
        if (to_date - from_date).days > 10:
            to_date = from_date + timedelta(days=10)
        
        params = {
            'sportId': sport_id,
            'from': f"{from_date.isoformat()}T00:00:00Z",
            'to': f"{to_date.isoformat()}T23:59:59Z",
            'hasOdds': str(has_odds).lower(),
            'language': 'en'
        }
        
        if tournament_id:
            params['tournamentId'] = tournament_id
        
        result = self._make_request('/v4/fixtures', params, cooldown_ms=1000)
        
        if result:
            # Filter for NBA (look for "NBA" in tournament name)
            nba_fixtures = [
                f for f in result 
                if 'NBA' in f.get('tournamentName', '') or 'nba' in f.get('tournamentName', '').lower()
            ]
            return nba_fixtures
        
        return []
    
    # ==========================================
    # Odds Endpoint
    # ==========================================
    
    def get_fixture_odds(
        self,
        fixture_id: str,
        bookmakers: List[str] = None,
        odds_format: str = 'decimal'
    ) -> Optional[Dict]:
        """
        Get odds for a specific fixture
        
        Args:
            fixture_id: Fixture ID (e.g., "id1000015561098461")
            bookmakers: List of bookmaker slugs to filter
            odds_format: 'decimal', 'american', or 'fractional'
        """
        params = {
            'fixtureId': fixture_id,
            'oddsFormat': odds_format,
            'verbosity': 3,
            'language': 'en'
        }
        
        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)
        
        result = self._make_request('/v4/odds', params, cooldown_ms=500)
        return result
    
    def get_game_lines(self, fixture_id: str, bookmaker: str = None) -> Optional[GameLine]:
        """
        Get game lines (spread, total, moneyline) for a fixture
        """
        bookmaker = bookmaker or self.preferred_bookmaker
        
        odds_data = self.get_fixture_odds(fixture_id, [bookmaker], 'decimal')
        
        if not odds_data:
            return None
        
        return self._parse_game_lines(odds_data, bookmaker)
    
    def get_player_props(self, fixture_id: str, bookmaker: str = None) -> List[PlayerPropLine]:
        """
        Get player prop lines for a fixture
        """
        bookmaker = bookmaker or self.preferred_bookmaker
        
        # Ensure markets are loaded
        if not self._markets_cache:
            self.get_markets()
        
        odds_data = self.get_fixture_odds(fixture_id, [bookmaker], 'decimal')
        
        if not odds_data:
            return []
        
        return self._parse_player_props(odds_data, bookmaker, fixture_id)
    
    # ==========================================
    # Historical Odds
    # ==========================================
    
    def get_historical_odds(
        self,
        fixture_id: str,
        bookmakers: List[str] = None
    ) -> Optional[Dict]:
        """
        Get historical odds for a fixture
        
        Note: 5000ms cooldown on this endpoint
        """
        bookmakers = bookmakers or [self.preferred_bookmaker]
        
        # Max 3 bookmakers
        if len(bookmakers) > 3:
            bookmakers = bookmakers[:3]
        
        params = {
            'fixtureId': fixture_id,
            'bookmakers': ','.join(bookmakers)
        }
        
        result = self._make_request('/v4/historical-odds', params, cooldown_ms=5000)
        return result
    
    # ==========================================
    # Settlements
    # ==========================================
    
    def get_settlements(self, fixture_id: str) -> Optional[Dict]:
        """
        Get settlement results for a fixture
        """
        params = {'fixtureId': fixture_id}
        result = self._make_request('/v4/settlements', params, cooldown_ms=2000)
        return result
    
    # ==========================================
    # Parsing Helpers
    # ==========================================
    
    def _parse_game_lines(self, odds_data: Dict, bookmaker: str) -> Optional[GameLine]:
        """Parse game lines from odds response"""
        try:
            game_line = GameLine(
                game_id=odds_data.get('fixtureId', ''),
                game_date=date.today(),
                home_team=odds_data.get('participant1Name', ''),
                away_team=odds_data.get('participant2Name', ''),
                home_team_id=odds_data.get('participant1Id'),
                away_team_id=odds_data.get('participant2Id'),
                bookmaker=bookmaker
            )
            
            # Parse start time
            start_time = odds_data.get('startTime')
            if start_time:
                try:
                    game_line.game_date = datetime.fromisoformat(
                        start_time.replace('Z', '+00:00')
                    ).date()
                except:
                    pass
            
            # Get bookmaker odds
            bm_odds = odds_data.get('bookmakerOdds', {}).get(bookmaker, {})
            markets = bm_odds.get('markets', {})
            
            for market_id_str, market_data in markets.items():
                market_id = int(market_id_str)
                market_info = self._markets_cache.get(market_id, {})
                market_type = market_info.get('marketType', '')
                market_name = market_info.get('marketName', '').lower()
                
                outcomes = market_data.get('outcomes', {})
                
                # Moneyline (1x2 or h2h)
                if market_type == '1x2' or 'moneyline' in market_name or 'money line' in market_name:
                    if 'fulltime' in market_info.get('period', ''):
                        for oid_str, outcome_data in outcomes.items():
                            players = outcome_data.get('players', {})
                            for pid, player_data in players.items():
                                price = player_data.get('price')
                                if price:
                                    bm_outcome_id = player_data.get('bookmakerOutcomeId', '')
                                    if bm_outcome_id == 'home' or '1' in str(oid_str):
                                        game_line.home_ml = price
                                        game_line.home_ml_american = self.decimal_to_american(price)
                                    elif bm_outcome_id == 'away' or '2' in str(oid_str):
                                        game_line.away_ml = price
                                        game_line.away_ml_american = self.decimal_to_american(price)
                
                # Spread/Handicap
                elif 'spread' in market_type or 'handicap' in market_name:
                    handicap = market_info.get('handicap', 0)
                    for oid_str, outcome_data in outcomes.items():
                        players = outcome_data.get('players', {})
                        for pid, player_data in players.items():
                            price = player_data.get('price')
                            bm_outcome_id = player_data.get('bookmakerOutcomeId', '')
                            if price and 'home' in bm_outcome_id:
                                # Extract spread from bookmakerOutcomeId (e.g., "1.5/home")
                                parts = bm_outcome_id.split('/')
                                if len(parts) >= 2:
                                    try:
                                        game_line.spread_line = float(parts[0])
                                    except:
                                        game_line.spread_line = handicap
                                game_line.spread_home_odds = price
                                game_line.spread_home_american = self.decimal_to_american(price)
                            elif price and 'away' in bm_outcome_id:
                                game_line.spread_away_odds = price
                                game_line.spread_away_american = self.decimal_to_american(price)
                
                # Totals (Over/Under)
                elif market_type == 'totals' or 'over' in market_name:
                    if 'fulltime' in market_info.get('period', '') or game_line.total_line is None:
                        for oid_str, outcome_data in outcomes.items():
                            players = outcome_data.get('players', {})
                            for pid, player_data in players.items():
                                price = player_data.get('price')
                                bm_outcome_id = player_data.get('bookmakerOutcomeId', '')
                                if price and 'over' in bm_outcome_id:
                                    parts = bm_outcome_id.split('/')
                                    if len(parts) >= 2:
                                        try:
                                            game_line.total_line = float(parts[0])
                                        except:
                                            pass
                                    game_line.over_odds = price
                                    game_line.over_american = self.decimal_to_american(price)
                                elif price and 'under' in bm_outcome_id:
                                    game_line.under_odds = price
                                    game_line.under_american = self.decimal_to_american(price)
            
            game_line.last_update = datetime.now()
            return game_line
            
        except Exception as e:
            print(f"Error parsing game lines: {e}")
            return None
    
    def _parse_player_props(self, odds_data: Dict, bookmaker: str, fixture_id: str) -> List[PlayerPropLine]:
        """Parse player props from odds response"""
        props = []
        
        try:
            # Load participants for player names
            if not self._participants_cache:
                sport_id = odds_data.get('sportId', 11)
                self.get_participants(sport_id)
            
            bm_odds = odds_data.get('bookmakerOdds', {}).get(bookmaker, {})
            markets = bm_odds.get('markets', {})
            
            for market_id_str, market_data in markets.items():
                market_id = int(market_id_str)
                market_info = self._markets_cache.get(market_id, {})
                
                # Only process player prop markets
                if not market_info.get('playerProp', False):
                    continue
                
                prop_type = self.PLAYER_PROP_MARKETS.get(market_id)
                if not prop_type:
                    # Try to infer from market name
                    market_name = market_info.get('marketName', '').lower()
                    if 'points' in market_name:
                        prop_type = 'pts'
                    elif 'rebound' in market_name:
                        prop_type = 'reb'
                    elif 'assist' in market_name:
                        prop_type = 'ast'
                    else:
                        continue
                
                outcomes = market_data.get('outcomes', {})
                
                # Group by player
                player_lines: Dict[int, Dict] = {}
                
                for oid_str, outcome_data in outcomes.items():
                    outcome_id = int(oid_str)
                    outcome_info = None
                    for o in market_info.get('outcomes', []):
                        if o['outcomeId'] == outcome_id:
                            outcome_info = o
                            break
                    
                    players_data = outcome_data.get('players', {})
                    
                    for player_id_str, player_data in players_data.items():
                        player_id = int(player_id_str)
                        if player_id == 0:
                            continue  # Skip non-player entries
                        
                        if player_id not in player_lines:
                            player_lines[player_id] = {
                                'over_price': None,
                                'under_price': None,
                                'line': None
                            }
                        
                        price = player_data.get('price')
                        bm_outcome_id = player_data.get('bookmakerOutcomeId', '')
                        
                        # Extract line from bookmakerOutcomeId
                        if '/' in bm_outcome_id:
                            parts = bm_outcome_id.split('/')
                            try:
                                player_lines[player_id]['line'] = float(parts[0])
                            except:
                                pass
                        
                        if 'over' in bm_outcome_id.lower():
                            player_lines[player_id]['over_price'] = price
                        elif 'under' in bm_outcome_id.lower():
                            player_lines[player_id]['under_price'] = price
                
                # Create PlayerPropLine objects
                for player_id, line_data in player_lines.items():
                    if line_data['line'] is not None and line_data['over_price'] is not None:
                        player_name = self._participants_cache.get(player_id, f"Player {player_id}")
                        
                        over_price = line_data['over_price']
                        under_price = line_data['under_price'] or over_price
                        
                        props.append(PlayerPropLine(
                            player_name=player_name,
                            player_id=player_id,
                            prop_type=prop_type,
                            line=line_data['line'],
                            over_odds=over_price,
                            under_odds=under_price,
                            over_odds_american=self.decimal_to_american(over_price),
                            under_odds_american=self.decimal_to_american(under_price),
                            bookmaker=bookmaker,
                            last_update=datetime.now(),
                            game_id=fixture_id,
                            market_id=market_id
                        ))
        
        except Exception as e:
            print(f"Error parsing player props: {e}")
        
        return props
    
    def get_team_abbrev(self, team_name: str) -> str:
        """Convert full team name to abbreviation"""
        return self.TEAM_NAME_MAP.get(team_name, team_name[:3].upper())
    
    def close(self):
        """Close the session"""
        self.session.close()