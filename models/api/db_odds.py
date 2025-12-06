# models/api/db_odds.py
"""
NBA Odds Integration Module
Connects predictions with DraftKings odds data for edge calculation and bet tracking
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import date, datetime, timedelta
from decimal import Decimal
import os
from dotenv import load_dotenv

load_dotenv()

class NBAOddsManager:
    """Manages NBA odds data and prediction comparisons"""
    
    # Prop type mapping between our system and DraftKings
    PROP_TYPE_MAP = {
        'pts': ['points', 'points_ou'],
        'reb': ['rebounds', 'rebounds_ou'],
        'ast': ['assists', 'assists_ou'],
        'pra': ['pts_rebs_asts', 'pts_rebs_asts_ou'],
        'pr': ['pts_rebs', 'pts_rebs_ou'],
        'pa': ['pts_asts', 'pts_asts_ou'],
        'ra': ['rebs_asts_ou'],
        'blk': ['blocks', 'blocks_ou'],
        'stl': ['steals'],
    }
    
    # Minimum edge thresholds for bet recommendations
    EDGE_THRESHOLDS = {
        'EXCEPTIONAL': 0.08,  # 8%+ edge
        'STRONG': 0.05,       # 5-8% edge
        'GOOD': 0.03,         # 3-5% edge
        'MODERATE': 0.02,     # 2-3% edge - minimum for bet
    }
    
    def __init__(self):
        self.conn = self._get_connection()
        self.ensure_tracking_tables()
    
    def _get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "nba_props"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD")
        )
    
    def ensure_tracking_tables(self):
        """Create NBA bet tracking tables if they don't exist"""
        with self.conn.cursor() as cur:
            cur.execute("""
                -- NBA Prop Bet Tracking
                CREATE TABLE IF NOT EXISTS nba_prop_bets (
                    id SERIAL PRIMARY KEY,
                    prediction_id INTEGER,
                    game_id VARCHAR(50),
                    player_id INTEGER,
                    player_name VARCHAR(100),
                    prop_type VARCHAR(20),
                    
                    -- Prediction data
                    predicted_value DECIMAL(6,2),
                    prediction_confidence DECIMAL(5,4),
                    
                    -- Odds data
                    dk_selection_id VARCHAR(100),
                    line DECIMAL(5,1),
                    bet_direction VARCHAR(10),  -- 'over' or 'under'
                    odds_american VARCHAR(10),
                    odds_decimal DECIMAL(6,3),
                    
                    -- Edge calculation
                    implied_probability DECIMAL(5,4),
                    model_probability DECIMAL(5,4),
                    edge DECIMAL(5,4),
                    edge_class VARCHAR(20),
                    
                    -- Bet sizing
                    recommended_bet BOOLEAN DEFAULT FALSE,
                    bet_size DECIMAL(10,2),
                    bet_pct_bankroll DECIMAL(5,4),
                    
                    -- Tracking
                    game_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Results (filled in after game)
                    actual_value DECIMAL(6,2),
                    bet_result VARCHAR(10),  -- 'WIN', 'LOSS', 'PUSH'
                    pnl DECIMAL(10,2),
                    processed_at TIMESTAMP,
                    
                    UNIQUE(game_id, player_id, prop_type, bet_direction)
                );
                
                -- Index for quick lookups
                CREATE INDEX IF NOT EXISTS idx_nba_prop_bets_date ON nba_prop_bets(game_date);
                CREATE INDEX IF NOT EXISTS idx_nba_prop_bets_player ON nba_prop_bets(player_id);
                CREATE INDEX IF NOT EXISTS idx_nba_prop_bets_result ON nba_prop_bets(bet_result);
                
                -- NBA Bankroll History
                CREATE TABLE IF NOT EXISTS nba_bankroll_history (
                    id SERIAL PRIMARY KEY,
                    snapshot_date DATE UNIQUE,
                    starting_bankroll DECIMAL(12,2),
                    ending_bankroll DECIMAL(12,2),
                    daily_pnl DECIMAL(10,2),
                    bets_placed INTEGER,
                    bets_won INTEGER,
                    bets_lost INTEGER,
                    bets_pushed INTEGER,
                    total_staked DECIMAL(12,2),
                    roi_pct DECIMAL(6,3),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.commit()
    
    def get_odds_for_player(self, player_name: str, prop_type: str, game_date: date = None):
        """
        Get DraftKings odds for a specific player and prop type
        Returns both over and under odds if available
        """
        if game_date is None:
            game_date = date.today()
        
        # Get the DK prop types that match our prop type
        dk_prop_types = self.PROP_TYPE_MAP.get(prop_type.lower(), [prop_type])
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    pp.id,
                    pp.player_name,
                    pp.prop_type,
                    pp.label,
                    pp.line,
                    pp.odds_american,
                    pp.odds_decimal,
                    pp.dk_selection_id,
                    pp.dk_market_id,
                    g.name as game_name,
                    g.home_team,
                    g.away_team,
                    g.start_time_mt,
                    pp.updated_at
                FROM player_props pp
                JOIN games g ON pp.game_id = g.id
                WHERE LOWER(pp.player_name) LIKE LOWER(%s)
                AND pp.prop_type = ANY(%s)
                AND DATE(g.start_time_mt) = %s
                ORDER BY pp.prop_type, pp.label
            """, (f"%{player_name}%", dk_prop_types, game_date))
            
            return cur.fetchall()
    
    def get_all_odds_for_date(self, game_date: date = None):
        """Get all player prop odds for a specific date"""
        if game_date is None:
            game_date = date.today()
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    pp.id,
                    pp.player_name,
                    pp.prop_type,
                    pp.label,
                    pp.line,
                    pp.odds_american,
                    pp.odds_decimal,
                    pp.dk_selection_id,
                    g.dk_event_id,
                    g.name as game_name,
                    g.home_team,
                    g.away_team,
                    g.start_time_mt
                FROM player_props pp
                JOIN games g ON pp.game_id = g.id
                WHERE DATE(g.start_time_mt) = %s
                ORDER BY g.start_time_mt, pp.player_name, pp.prop_type
            """, (game_date,))
            
            return cur.fetchall()
    
    def calculate_edge(self, model_prob: float, odds_decimal: float) -> dict:
        """
        Calculate the edge between model probability and implied odds probability
        
        Args:
            model_prob: Our model's probability of the bet winning (0-1)
            odds_decimal: Decimal odds from sportsbook
        
        Returns:
            dict with edge info
        """
        if not odds_decimal or odds_decimal <= 1:
            return {'edge': 0, 'edge_class': 'INVALID', 'implied_prob': 0}
        
        # Calculate implied probability from odds
        implied_prob = 1 / odds_decimal
        
        # Edge = Model Probability - Implied Probability
        edge = model_prob - implied_prob
        
        # Classify edge
        if edge >= self.EDGE_THRESHOLDS['EXCEPTIONAL']:
            edge_class = 'EXCEPTIONAL'
        elif edge >= self.EDGE_THRESHOLDS['STRONG']:
            edge_class = 'STRONG'
        elif edge >= self.EDGE_THRESHOLDS['GOOD']:
            edge_class = 'GOOD'
        elif edge >= self.EDGE_THRESHOLDS['MODERATE']:
            edge_class = 'MODERATE'
        elif edge > 0:
            edge_class = 'MARGINAL'
        else:
            edge_class = 'NEGATIVE'
        
        return {
            'edge': edge,
            'edge_pct': edge * 100,
            'edge_class': edge_class,
            'implied_prob': implied_prob,
            'model_prob': model_prob,
            'is_bet_recommended': edge >= self.EDGE_THRESHOLDS['MODERATE']
        }
    
    def calculate_kelly_bet(self, model_prob: float, odds_decimal: float, 
                           bankroll: float, kelly_fraction: float = 0.25) -> dict:
        """
        Calculate Kelly Criterion bet size
        
        Args:
            model_prob: Model's probability of winning
            odds_decimal: Decimal odds
            bankroll: Current bankroll
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        
        Returns:
            dict with bet sizing info
        """
        if odds_decimal <= 1:
            return {'bet_size': 0, 'bet_pct': 0}
        
        # Kelly formula: f = (bp - q) / b
        # where b = decimal odds - 1, p = win prob, q = lose prob
        b = odds_decimal - 1
        p = model_prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply Kelly fraction and cap
        if kelly <= 0:
            return {'bet_size': 0, 'bet_pct': 0, 'full_kelly': kelly}
        
        adjusted_kelly = kelly * kelly_fraction
        capped_kelly = min(adjusted_kelly, 0.05)  # Max 5% of bankroll
        
        bet_size = bankroll * capped_kelly
        
        return {
            'bet_size': round(bet_size, 2),
            'bet_pct': capped_kelly,
            'bet_pct_display': capped_kelly * 100,
            'full_kelly': kelly,
            'adjusted_kelly': adjusted_kelly
        }
    
    def match_predictions_with_odds(self, predictions: list, game_date: date = None) -> list:
        """
        Match our predictions with DraftKings odds and calculate edges
        
        Args:
            predictions: List of prediction dicts with player_name, prop_type, predicted_value
            game_date: Date to get odds for
        
        Returns:
            List of predictions enriched with odds and edge data
        """
        if game_date is None:
            game_date = date.today()
        
        # Get all odds for the date
        all_odds = self.get_all_odds_for_date(game_date)
        
        # Create lookup by player name (lowercase) and prop type
        odds_lookup = {}
        for odd in all_odds:
            player_key = odd['player_name'].lower().strip()
            prop_type = odd['prop_type']
            
            if player_key not in odds_lookup:
                odds_lookup[player_key] = {}
            if prop_type not in odds_lookup[player_key]:
                odds_lookup[player_key][prop_type] = []
            odds_lookup[player_key][prop_type].append(odd)
        
        enriched = []
        for pred in predictions:
            pred_copy = dict(pred)
            player_key = pred.get('player_name', '').lower().strip()
            our_prop_type = pred.get('prop_type', '').lower()
            
            # Get matching DK prop types
            dk_prop_types = self.PROP_TYPE_MAP.get(our_prop_type, [our_prop_type])
            
            # Find matching odds
            matched_odds = []
            for dk_prop in dk_prop_types:
                if player_key in odds_lookup and dk_prop in odds_lookup[player_key]:
                    matched_odds.extend(odds_lookup[player_key][dk_prop])
            
            # Also try partial name matching
            if not matched_odds:
                for ok_player, ok_props in odds_lookup.items():
                    if player_key in ok_player or ok_player in player_key:
                        for dk_prop in dk_prop_types:
                            if dk_prop in ok_props:
                                matched_odds.extend(ok_props[dk_prop])
                        break
            
            if matched_odds:
                # Get over/under odds
                over_odds = next((o for o in matched_odds if o['label'].lower() == 'over'), None)
                under_odds = next((o for o in matched_odds if o['label'].lower() == 'under'), None)
                
                # Get the line from odds (should be same for over/under)
                line = matched_odds[0]['line']
                pred_value = float(pred.get('predicted_value', 0))
                
                # Determine bet direction based on prediction vs line
                if pred_value > float(line):
                    bet_direction = 'over'
                    relevant_odds = over_odds
                else:
                    bet_direction = 'under'
                    relevant_odds = under_odds
                
                if relevant_odds:
                    odds_decimal = float(relevant_odds['odds_decimal'] or 1.91)
                    
                    # Calculate model probability for the bet
                    # This is a simplified calculation - in production you'd use
                    # your model's actual probability output
                    confidence = float(pred.get('confidence', 0.5))
                    diff = abs(pred_value - float(line))
                    
                    # Estimate probability based on prediction confidence and line diff
                    # Higher confidence + bigger diff = higher probability
                    base_prob = 0.5 + (confidence - 0.5) * 0.5
                    diff_factor = min(diff / 5, 0.15)  # Cap diff boost at 15%
                    model_prob = min(base_prob + diff_factor, 0.85)
                    
                    # Calculate edge
                    edge_info = self.calculate_edge(model_prob, odds_decimal)
                    
                    # Add odds data to prediction
                    pred_copy['has_odds'] = True
                    pred_copy['dk_line'] = float(line)
                    pred_copy['bet_direction'] = bet_direction
                    pred_copy['odds_american'] = relevant_odds['odds_american']
                    pred_copy['odds_decimal'] = odds_decimal
                    pred_copy['dk_selection_id'] = relevant_odds['dk_selection_id']
                    pred_copy['implied_probability'] = edge_info['implied_prob']
                    pred_copy['model_probability'] = model_prob
                    pred_copy['edge'] = edge_info['edge']
                    pred_copy['edge_pct'] = edge_info['edge_pct']
                    pred_copy['edge_class'] = edge_info['edge_class']
                    pred_copy['is_bet_recommended'] = edge_info['is_bet_recommended']
                    
                    # Also include the opposite side odds for reference
                    opposite = under_odds if bet_direction == 'over' else over_odds
                    if opposite:
                        pred_copy['opposite_odds_american'] = opposite['odds_american']
                        pred_copy['opposite_odds_decimal'] = float(opposite['odds_decimal'] or 0)
                else:
                    pred_copy['has_odds'] = False
                    pred_copy['odds_error'] = 'No odds for bet direction'
            else:
                pred_copy['has_odds'] = False
                pred_copy['odds_error'] = 'No matching odds found'
            
            enriched.append(pred_copy)
        
        return enriched
    
    def save_bet_recommendation(self, bet_data: dict, bankroll: float = 1000.0) -> int:
        """
        Save a bet recommendation to the tracking table
        
        Args:
            bet_data: Dict with prediction and odds data
            bankroll: Current bankroll for bet sizing
        
        Returns:
            ID of the saved bet
        """
        # Calculate bet size
        kelly_info = self.calculate_kelly_bet(
            bet_data.get('model_probability', 0.5),
            bet_data.get('odds_decimal', 1.91),
            bankroll
        )
        
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO nba_prop_bets (
                    game_id, player_id, player_name, prop_type,
                    predicted_value, prediction_confidence,
                    dk_selection_id, line, bet_direction,
                    odds_american, odds_decimal,
                    implied_probability, model_probability,
                    edge, edge_class,
                    recommended_bet, bet_size, bet_pct_bankroll,
                    game_date
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s
                )
                ON CONFLICT (game_id, player_id, prop_type, bet_direction)
                DO UPDATE SET
                    predicted_value = EXCLUDED.predicted_value,
                    line = EXCLUDED.line,
                    odds_american = EXCLUDED.odds_american,
                    odds_decimal = EXCLUDED.odds_decimal,
                    edge = EXCLUDED.edge,
                    edge_class = EXCLUDED.edge_class,
                    bet_size = EXCLUDED.bet_size,
                    bet_pct_bankroll = EXCLUDED.bet_pct_bankroll
                RETURNING id
            """, (
                bet_data.get('game_id'),
                bet_data.get('player_id'),
                bet_data.get('player_name'),
                bet_data.get('prop_type'),
                bet_data.get('predicted_value'),
                bet_data.get('confidence'),
                bet_data.get('dk_selection_id'),
                bet_data.get('dk_line'),
                bet_data.get('bet_direction'),
                bet_data.get('odds_american'),
                bet_data.get('odds_decimal'),
                bet_data.get('implied_probability'),
                bet_data.get('model_probability'),
                bet_data.get('edge'),
                bet_data.get('edge_class'),
                bet_data.get('is_bet_recommended', False),
                kelly_info['bet_size'],
                kelly_info['bet_pct'],
                bet_data.get('game_date', date.today())
            ))
            
            result = cur.fetchone()
            self.conn.commit()
            return result[0] if result else None
    
    def get_pending_bets(self, game_date: date = None) -> list:
        """Get bets that haven't been resolved yet"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT * FROM nba_prop_bets
                WHERE bet_result IS NULL
                AND recommended_bet = TRUE
            """
            params = []
            
            if game_date:
                query += " AND game_date = %s"
                params.append(game_date)
            
            query += " ORDER BY game_date DESC, edge DESC"
            
            cur.execute(query, params)
            return cur.fetchall()
    
    def update_bet_result(self, bet_id: int, actual_value: float) -> dict:
        """
        Update a bet with its actual result
        
        Args:
            bet_id: ID of the bet to update
            actual_value: The actual stat value achieved
        
        Returns:
            Updated bet data
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get the bet
            cur.execute("SELECT * FROM nba_prop_bets WHERE id = %s", (bet_id,))
            bet = cur.fetchone()
            
            if not bet:
                return None
            
            line = float(bet['line'])
            direction = bet['bet_direction']
            odds_decimal = float(bet['odds_decimal'])
            bet_size = float(bet['bet_size'])
            
            # Determine result
            if direction == 'over':
                if actual_value > line:
                    result = 'WIN'
                    pnl = bet_size * (odds_decimal - 1)
                elif actual_value == line:
                    result = 'PUSH'
                    pnl = 0
                else:
                    result = 'LOSS'
                    pnl = -bet_size
            else:  # under
                if actual_value < line:
                    result = 'WIN'
                    pnl = bet_size * (odds_decimal - 1)
                elif actual_value == line:
                    result = 'PUSH'
                    pnl = 0
                else:
                    result = 'LOSS'
                    pnl = -bet_size
            
            # Update the bet
            cur.execute("""
                UPDATE nba_prop_bets
                SET actual_value = %s,
                    bet_result = %s,
                    pnl = %s,
                    processed_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING *
            """, (actual_value, result, pnl, bet_id))
            
            self.conn.commit()
            return cur.fetchone()
    
    def get_bet_history(self, days: int = 30, result_filter: str = None) -> list:
        """Get historical bet results"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT * FROM nba_prop_bets
                WHERE game_date >= CURRENT_DATE - INTERVAL '%s days'
                AND recommended_bet = TRUE
            """
            params = [days]
            
            if result_filter:
                query += " AND bet_result = %s"
                params.append(result_filter)
            
            query += " ORDER BY game_date DESC, created_at DESC"
            
            cur.execute(query, params)
            return cur.fetchall()
    
    def get_game_lines_for_date(self, game_date: date = None) -> list:
        """Get all game lines (moneyline, spread, total) for a specific date"""
        if game_date is None:
            game_date = date.today()
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get games for the date
            cur.execute("""
                SELECT 
                    g.id as game_db_id,
                    g.dk_event_id,
                    g.name as game_name,
                    g.home_team,
                    g.away_team,
                    g.start_time_mt,
                    gl.line_type,
                    gl.team,
                    gl.label,
                    gl.line,
                    gl.odds_american,
                    gl.odds_decimal,
                    gl.dk_selection_id
                FROM games g
                LEFT JOIN game_lines gl ON g.id = gl.game_id
                WHERE DATE(g.start_time_mt) = %s
                ORDER BY g.start_time_mt, g.id, gl.line_type, gl.team
            """, (game_date,))
            
            rows = cur.fetchall()
        
        # Group by game
        games = {}
        for row in rows:
            game_id = row['dk_event_id']
            if game_id not in games:
                games[game_id] = {
                    'game_id': game_id,
                    'game_db_id': row['game_db_id'],
                    'game_name': row['game_name'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'start_time': row['start_time_mt'],
                    'moneyline': {'home': None, 'away': None},
                    'spread': {'home': None, 'away': None},
                    'total': {'over': None, 'under': None}
                }
            
            if row['line_type'] == 'moneyline':
                if row['team'] == row['home_team']:
                    games[game_id]['moneyline']['home'] = {
                        'team': row['team'],
                        'odds_american': row['odds_american'],
                        'odds_decimal': float(row['odds_decimal']) if row['odds_decimal'] else None
                    }
                else:
                    games[game_id]['moneyline']['away'] = {
                        'team': row['team'],
                        'odds_american': row['odds_american'],
                        'odds_decimal': float(row['odds_decimal']) if row['odds_decimal'] else None
                    }
            
            elif row['line_type'] == 'spread':
                spread_data = {
                    'team': row['team'],
                    'line': float(row['line']) if row['line'] else None,
                    'odds_american': row['odds_american'],
                    'odds_decimal': float(row['odds_decimal']) if row['odds_decimal'] else None
                }
                if row['team'] == row['home_team']:
                    games[game_id]['spread']['home'] = spread_data
                else:
                    games[game_id]['spread']['away'] = spread_data
            
            elif row['line_type'] == 'total':
                total_data = {
                    'line': float(row['line']) if row['line'] else None,
                    'odds_american': row['odds_american'],
                    'odds_decimal': float(row['odds_decimal']) if row['odds_decimal'] else None
                }
                if row['label'] and 'over' in row['label'].lower():
                    games[game_id]['total']['over'] = total_data
                elif row['label'] and 'under' in row['label'].lower():
                    games[game_id]['total']['under'] = total_data
        
        return list(games.values())
    
    def get_odds_for_game(self, dk_event_id: str) -> dict:
        """Get all odds (game lines + player props) for a specific game"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get game info
            cur.execute("""
                SELECT id, dk_event_id, name, home_team, away_team, start_time_mt
                FROM games WHERE dk_event_id = %s
            """, (dk_event_id,))
            game = cur.fetchone()
            
            if not game:
                return {'error': 'Game not found'}
            
            # Get game lines
            cur.execute("""
                SELECT line_type, team, label, line, odds_american, odds_decimal
                FROM game_lines WHERE game_id = %s
                ORDER BY line_type, team
            """, (game['id'],))
            game_lines = cur.fetchall()
            
            # Get player props
            cur.execute("""
                SELECT player_name, prop_type, label, line, odds_american, odds_decimal
                FROM player_props WHERE game_id = %s
                ORDER BY player_name, prop_type
            """, (game['id'],))
            player_props = cur.fetchall()
        
        # Organize game lines
        organized_lines = {
            'moneyline': {'home': None, 'away': None},
            'spread': {'home': None, 'away': None},
            'total': {'over': None, 'under': None}
        }
        
        for line in game_lines:
            lt = line['line_type']
            if lt == 'moneyline':
                side = 'home' if line['team'] == game['home_team'] else 'away'
                organized_lines['moneyline'][side] = dict(line)
            elif lt == 'spread':
                side = 'home' if line['team'] == game['home_team'] else 'away'
                organized_lines['spread'][side] = dict(line)
            elif lt == 'total':
                side = 'over' if line['label'] and 'over' in line['label'].lower() else 'under'
                organized_lines['total'][side] = dict(line)
        
        # Organize player props by player
        players = {}
        for prop in player_props:
            name = prop['player_name']
            if name not in players:
                players[name] = []
            players[name].append(dict(prop))
        
        return {
            'game': dict(game),
            'game_lines': organized_lines,
            'player_props': players,
            'player_count': len(players),
            'prop_count': len(player_props)
        }

    def get_performance_stats(self, days: int = 30) -> dict:
        """Get overall performance statistics"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN bet_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN bet_result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN bet_result = 'PUSH' THEN 1 ELSE 0 END) as pushes,
                    SUM(COALESCE(pnl, 0)) as total_pnl,
                    SUM(bet_size) as total_staked,
                    AVG(edge) as avg_edge,
                    AVG(model_probability) as avg_model_prob
                FROM nba_prop_bets
                WHERE game_date >= CURRENT_DATE - INTERVAL '%s days'
                AND recommended_bet = TRUE
                AND bet_result IS NOT NULL
            """, (days,))
            
            overall = cur.fetchone()
            
            # By edge class
            cur.execute("""
                SELECT 
                    edge_class,
                    COUNT(*) as bets,
                    SUM(CASE WHEN bet_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(COALESCE(pnl, 0)) as pnl,
                    AVG(edge) as avg_edge
                FROM nba_prop_bets
                WHERE game_date >= CURRENT_DATE - INTERVAL '%s days'
                AND recommended_bet = TRUE
                AND bet_result IS NOT NULL
                GROUP BY edge_class
                ORDER BY avg_edge DESC
            """, (days,))
            
            by_edge = cur.fetchall()
            
            # By prop type
            cur.execute("""
                SELECT 
                    prop_type,
                    COUNT(*) as bets,
                    SUM(CASE WHEN bet_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(COALESCE(pnl, 0)) as pnl
                FROM nba_prop_bets
                WHERE game_date >= CURRENT_DATE - INTERVAL '%s days'
                AND recommended_bet = TRUE
                AND bet_result IS NOT NULL
                GROUP BY prop_type
                ORDER BY pnl DESC
            """, (days,))
            
            by_prop = cur.fetchall()
            
            return {
                'overall': dict(overall) if overall else {},
                'by_edge_class': [dict(r) for r in by_edge],
                'by_prop_type': [dict(r) for r in by_prop]
            }