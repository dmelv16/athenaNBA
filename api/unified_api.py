# api/unified_api.py - Enhanced with Odds Integration
"""
Unified Sports API - Enhanced with DraftKings Odds Integration
Supports NBA props with live odds comparison and edge calculation
"""

import sys
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import date, datetime, timedelta
from decimal import Decimal
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# Initialize modules
NBA_AVAILABLE = False
NHL_AVAILABLE = False
ODDS_AVAILABLE = False
TRACKER_AVAILABLE = False

nba_uploader = None
nhl_uploader = None
odds_manager = None
tracker = None

# NBA module
try:
    from models.api.db_uploader import NBAPredictionUploader
    nba_uploader = NBAPredictionUploader()
    NBA_AVAILABLE = True
    print("✓ NBA module loaded")
except Exception as e:
    print(f"⚠ NBA module not available: {e}")

# NHL module
try:
    from src.connection.db_uploader import PostgresPredictionUploader
    nhl_uploader = PostgresPredictionUploader()
    NHL_AVAILABLE = True
    print("✓ NHL module loaded")
except Exception as e:
    print(f"⚠ NHL module not available: {e}")

# Odds manager (NEW)
try:
    from models.api.db_odds import NBAOddsManager
    odds_manager = NBAOddsManager()
    ODDS_AVAILABLE = True
    print("✓ Odds manager loaded")
except Exception as e:
    print(f"⚠ Odds manager not available: {e}")

# NHL Tracker
try:
    from src.tracking.historical_tracker import HistoricalGameTracker
    MSSQL_CONN_STR = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-J9IV3OH;"
        "DATABASE=nhlDB;"
        "Trusted_Connection=yes;"
    )
    tracker = HistoricalGameTracker(MSSQL_CONN_STR, starting_bankroll=1000.0)
    tracker.connect()
    tracker.ensure_tracking_tables()
    TRACKER_AVAILABLE = True
    print("✓ NHL tracker loaded")
except Exception as e:
    print(f"⚠ NHL tracker not available: {e}")

# Bankroll state
current_bankroll = {"nba": 1000.0, "nhl": 1412.77, "updated_at": datetime.now().isoformat()}

def serialize(data):
    """Recursively serialize data for JSON"""
    if isinstance(data, list):
        return [serialize(item) for item in data]
    if isinstance(data, dict):
        return {k: serialize(v) for k, v in data.items()}
    if isinstance(data, Decimal):
        return float(data)
    if isinstance(data, (datetime, date)):
        return str(data)
    if hasattr(data, 'item'):
        return data.item()
    return data

# ============================================
# Health & General
# ============================================
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'modules': {
            'nba': 'available' if NBA_AVAILABLE else 'unavailable',
            'nhl': 'available' if NHL_AVAILABLE else 'unavailable',
            'odds': 'available' if ODDS_AVAILABLE else 'unavailable',
            'nhl_tracking': 'available' if TRACKER_AVAILABLE else 'unavailable'
        }
    })

@app.route('/api/sports', methods=['GET'])
def list_sports():
    sports = []
    if NBA_AVAILABLE:
        sports.append({'id': 'nba', 'name': 'NBA Basketball', 'icon': '🏀', 'has_odds': ODDS_AVAILABLE})
    if NHL_AVAILABLE:
        sports.append({'id': 'nhl', 'name': 'NHL Hockey', 'icon': '🏒'})
    return jsonify({'sports': sports})

# ============================================
# NBA Predictions with Odds Integration
# ============================================
@app.route('/api/nba/predictions/today', methods=['GET'])
def nba_today():
    """Get today's NBA predictions enriched with DraftKings odds"""
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    
    try:
        # Get base predictions
        data = nba_uploader.get_todays_predictions()
        
        # Enrich with odds if available
        if ODDS_AVAILABLE and data.get('games'):
            for game in data['games']:
                if game.get('player_predictions'):
                    enriched = odds_manager.match_predictions_with_odds(
                        game['player_predictions'],
                        date.today()
                    )
                    game['player_predictions'] = enriched
                    
                    # Calculate game-level stats
                    bets = [p for p in enriched if p.get('is_bet_recommended')]
                    game['recommended_bets'] = len(bets)
                    game['total_edge'] = sum(p.get('edge', 0) for p in bets)
        
        data['odds_available'] = ODDS_AVAILABLE
        data['bankroll'] = current_bankroll['nba']
        
        return jsonify(serialize(data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/predictions/date/<target_date>', methods=['GET'])
def nba_by_date(target_date):
    """Get NBA predictions for a specific date with odds"""
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        parsed = datetime.strptime(target_date, '%Y-%m-%d').date()
        data = nba_uploader.get_predictions_by_date(parsed)
        
        # Enrich with odds if available
        if ODDS_AVAILABLE:
            if isinstance(data, dict) and data.get('games'):
                for game in data['games']:
                    if game.get('player_predictions'):
                        game['player_predictions'] = odds_manager.match_predictions_with_odds(
                            game['player_predictions'], parsed
                        )
            elif isinstance(data, list):
                data = odds_manager.match_predictions_with_odds(data, parsed)
        
        return jsonify(serialize(data))
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/odds/today', methods=['GET'])
def nba_odds_today():
    """Get all DraftKings odds for today"""
    if not ODDS_AVAILABLE:
        return jsonify({'error': 'Odds not available'}), 503
    try:
        odds = odds_manager.get_all_odds_for_date(date.today())
        return jsonify(serialize({'odds': odds, 'count': len(odds)}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/odds/player/<player_name>', methods=['GET'])
def nba_odds_for_player(player_name):
    """Get all odds for a specific player"""
    if not ODDS_AVAILABLE:
        return jsonify({'error': 'Odds not available'}), 503
    try:
        prop_type = request.args.get('prop_type', None)
        target_date = request.args.get('date', date.today().isoformat())
        parsed_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        odds = odds_manager.get_odds_for_player(player_name, prop_type or 'points', parsed_date)
        return jsonify(serialize({'player': player_name, 'odds': odds}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/predictions/with-edge', methods=['GET'])
def nba_predictions_with_edge():
    """Get all predictions that have positive edge"""
    if not NBA_AVAILABLE or not ODDS_AVAILABLE:
        return jsonify({'error': 'NBA or Odds not available'}), 503
    
    try:
        min_edge = request.args.get('min_edge', 0.02, type=float)
        target_date = request.args.get('date', date.today().isoformat())
        parsed_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        # Get predictions
        if parsed_date == date.today():
            data = nba_uploader.get_todays_predictions()
        else:
            data = nba_uploader.get_predictions_by_date(parsed_date)
        
        # Collect all player predictions
        all_predictions = []
        if isinstance(data, dict) and data.get('games'):
            for game in data['games']:
                for pred in game.get('player_predictions', []):
                    pred['game_info'] = {
                        'home_team': game.get('home_team_abbrev'),
                        'away_team': game.get('away_team_abbrev'),
                        'game_time': game.get('game_time')
                    }
                    all_predictions.append(pred)
        
        # Enrich with odds
        enriched = odds_manager.match_predictions_with_odds(all_predictions, parsed_date)
        
        # Filter by edge
        bets = [p for p in enriched if p.get('edge', 0) >= min_edge]
        
        # Sort by edge descending
        bets.sort(key=lambda x: x.get('edge', 0), reverse=True)
        
        # Calculate bet sizes
        bankroll = current_bankroll['nba']
        for bet in bets:
            kelly = odds_manager.calculate_kelly_bet(
                bet.get('model_probability', 0.5),
                bet.get('odds_decimal', 1.91),
                bankroll
            )
            bet['bet_size'] = kelly['bet_size']
            bet['bet_pct'] = kelly['bet_pct_display']
        
        return jsonify(serialize({
            'date': target_date,
            'min_edge': min_edge,
            'total_bets': len(bets),
            'total_edge': sum(b.get('edge', 0) for b in bets),
            'bankroll': bankroll,
            'bets': bets
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/bets/save', methods=['POST'])
def nba_save_bet():
    """Save a bet recommendation for tracking"""
    if not ODDS_AVAILABLE:
        return jsonify({'error': 'Odds not available'}), 503
    
    try:
        data = request.get_json()
        bankroll = data.get('bankroll', current_bankroll['nba'])
        
        bet_id = odds_manager.save_bet_recommendation(data, bankroll)
        
        return jsonify({'success': True, 'bet_id': bet_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/bets/save-all', methods=['POST'])
def nba_save_all_bets():
    """Save all recommended bets for today"""
    if not NBA_AVAILABLE or not ODDS_AVAILABLE:
        return jsonify({'error': 'NBA or Odds not available'}), 503
    
    try:
        data = request.get_json() or {}
        min_edge = data.get('min_edge', 0.02)
        bankroll = data.get('bankroll', current_bankroll['nba'])
        
        # Get today's predictions with odds
        predictions = nba_uploader.get_todays_predictions()
        
        all_preds = []
        if predictions.get('games'):
            for game in predictions['games']:
                for pred in game.get('player_predictions', []):
                    pred['game_date'] = date.today()
                    all_preds.append(pred)
        
        # Enrich with odds
        enriched = odds_manager.match_predictions_with_odds(all_preds, date.today())
        
        # Save bets that meet edge threshold
        saved_count = 0
        for pred in enriched:
            if pred.get('edge', 0) >= min_edge and pred.get('has_odds'):
                pred['is_bet_recommended'] = True
                odds_manager.save_bet_recommendation(pred, bankroll)
                saved_count += 1
        
        return jsonify({
            'success': True,
            'bets_saved': saved_count,
            'min_edge': min_edge
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/bets/pending', methods=['GET'])
def nba_pending_bets():
    """Get all pending (unresolved) bets"""
    if not ODDS_AVAILABLE:
        return jsonify({'error': 'Odds not available'}), 503
    
    try:
        target_date = request.args.get('date', None)
        parsed_date = None
        if target_date:
            parsed_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        bets = odds_manager.get_pending_bets(parsed_date)
        return jsonify(serialize({'bets': bets, 'count': len(bets)}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/bets/history', methods=['GET'])
def nba_bet_history():
    """Get NBA bet history with results"""
    if not ODDS_AVAILABLE:
        return jsonify({'error': 'Odds not available'}), 503
    
    try:
        days = request.args.get('days', 30, type=int)
        result_filter = request.args.get('result', None)
        
        bets = odds_manager.get_bet_history(days, result_filter)
        return jsonify(serialize({'bets': bets, 'days': days, 'count': len(bets)}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/bets/result', methods=['POST'])
def nba_update_bet_result():
    """Update a bet with its result"""
    if not ODDS_AVAILABLE:
        return jsonify({'error': 'Odds not available'}), 503
    
    try:
        data = request.get_json()
        bet_id = data.get('bet_id')
        actual_value = data.get('actual_value')
        
        if not bet_id or actual_value is None:
            return jsonify({'error': 'bet_id and actual_value required'}), 400
        
        result = odds_manager.update_bet_result(bet_id, actual_value)
        
        if result:
            return jsonify(serialize({'success': True, 'bet': result}))
        else:
            return jsonify({'error': 'Bet not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/bets/performance', methods=['GET'])
def nba_bet_performance():
    """Get NBA betting performance stats"""
    if not ODDS_AVAILABLE:
        return jsonify({'error': 'Odds not available'}), 503
    
    try:
        days = request.args.get('days', 30, type=int)
        stats = odds_manager.get_performance_stats(days)
        return jsonify(serialize(stats))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Keep existing NBA endpoints for backward compatibility
@app.route('/api/nba/predictions/history', methods=['GET'])
def nba_history():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        data = nba_uploader.get_recent_predictions(days=days)
        return jsonify(serialize({'predictions': data, 'days': days}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/games/<game_id>', methods=['GET'])
def nba_game_detail(game_id):
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        with nba_uploader.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM nba_player_predictions WHERE game_id = %s ORDER BY confidence DESC",
                (game_id,)
            )
            cols = [d[0] for d in cur.description]
            player_preds = [dict(zip(cols, r)) for r in cur.fetchall()]
            
            # Enrich with odds
            if ODDS_AVAILABLE and player_preds:
                game_date = player_preds[0].get('game_date', date.today())
                if isinstance(game_date, str):
                    game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
                player_preds = odds_manager.match_predictions_with_odds(player_preds, game_date)
            
            cur.execute(
                "SELECT * FROM nba_team_predictions WHERE game_id = %s",
                (game_id,)
            )
            cols = [d[0] for d in cur.description]
            team_preds = [dict(zip(cols, r)) for r in cur.fetchall()]
        
        return jsonify(serialize({
            'game_id': game_id,
            'player_predictions': player_preds,
            'team_predictions': team_preds,
            'odds_available': ODDS_AVAILABLE
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/players/<int:player_id>/history', methods=['GET'])
def nba_player_history(player_id):
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        limit = request.args.get('limit', 5, type=int)
        prop_type = request.args.get('prop_type', None)
        
        query = """
            SELECT 
                p.prediction_date, p.game_id, p.game_date,
                p.player_name, p.team_abbrev, p.opponent_abbrev,
                p.prop_type, p.predicted_value, p.line, p.confidence,
                p.recommended_bet,
                pgl.pts as actual_pts, pgl.reb as actual_reb,
                pgl.ast as actual_ast,
                pgl.pts + pgl.reb + pgl.ast as actual_pra
            FROM nba_player_predictions p
            LEFT JOIN player_game_logs pgl 
                ON p.player_id = pgl.player_id AND p.game_id = pgl.game_id
            WHERE p.player_id = %s
        """
        params = [player_id]
        
        if prop_type:
            query += " AND p.prop_type = %s"
            params.append(prop_type)
        
        query += " ORDER BY p.prediction_date DESC LIMIT %s"
        params.append(limit)
        
        with nba_uploader.conn.cursor() as cur:
            cur.execute(query, tuple(params))
            cols = [d[0] for d in cur.description]
            results = [dict(zip(cols, r)) for r in cur.fetchall()]
        
        for r in results:
            prop = r['prop_type']
            if prop == 'pts': r['actual_value'] = r.get('actual_pts')
            elif prop == 'reb': r['actual_value'] = r.get('actual_reb')
            elif prop == 'ast': r['actual_value'] = r.get('actual_ast')
            elif prop == 'pra': r['actual_value'] = r.get('actual_pra')
            elif prop == 'pr': r['actual_value'] = (r.get('actual_pts') or 0) + (r.get('actual_reb') or 0)
            elif prop == 'pa': r['actual_value'] = (r.get('actual_pts') or 0) + (r.get('actual_ast') or 0)
            elif prop == 'ra': r['actual_value'] = (r.get('actual_reb') or 0) + (r.get('actual_ast') or 0)
            else: r['actual_value'] = None
            
            if r['actual_value'] is not None and r.get('line'):
                if r['recommended_bet'] == 'over':
                    r['hit'] = r['actual_value'] > r['line']
                elif r['recommended_bet'] == 'under':
                    r['hit'] = r['actual_value'] < r['line']
                else:
                    r['hit'] = None
            else:
                r['hit'] = None
        
        return jsonify(serialize({'player_id': player_id, 'history': results}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/accuracy', methods=['GET'])
def nba_accuracy():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        since_date = date.today() - timedelta(days=days)
        
        query = """
            WITH prediction_results AS (
                SELECT p.prop_type, p.predicted_value, p.line, p.confidence,
                    p.prediction_date, p.recommended_bet,
                    CASE p.prop_type
                        WHEN 'pts' THEN pgl.pts
                        WHEN 'reb' THEN pgl.reb
                        WHEN 'ast' THEN pgl.ast
                        WHEN 'pra' THEN pgl.pts + pgl.reb + pgl.ast
                        WHEN 'pr' THEN pgl.pts + pgl.reb
                        WHEN 'pa' THEN pgl.pts + pgl.ast
                        WHEN 'ra' THEN pgl.reb + pgl.ast
                    END as actual_value
                FROM nba_player_predictions p
                JOIN player_game_logs pgl 
                    ON p.player_id = pgl.player_id AND p.game_id = pgl.game_id
                WHERE p.prediction_date >= %s
                    AND p.prop_type IN ('pts', 'reb', 'ast', 'pra', 'pr', 'pa', 'ra')
            )
            SELECT prop_type, COUNT(*) as total_predictions,
                AVG(ABS(predicted_value - actual_value)) as avg_error,
                AVG(confidence) as avg_confidence,
                SUM(CASE WHEN line IS NOT NULL AND (
                    (recommended_bet = 'over' AND actual_value > line) OR
                    (recommended_bet = 'under' AND actual_value < line)
                ) THEN 1 ELSE 0 END)::float / 
                NULLIF(SUM(CASE WHEN line IS NOT NULL THEN 1 ELSE 0 END), 0) as line_accuracy
            FROM prediction_results
            WHERE actual_value IS NOT NULL
            GROUP BY prop_type ORDER BY prop_type
        """
        
        with nba_uploader.conn.cursor() as cur:
            cur.execute(query, (since_date,))
            cols = [d[0] for d in cur.description]
            results = [dict(zip(cols, r)) for r in cur.fetchall()]
        
        return jsonify(serialize({'days': days, 'accuracy_by_prop': results}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/game-lines/today', methods=['GET'])
def nba_game_lines_today():
    """Get all game lines (moneyline, spread, totals) for today's games"""
    if not ODDS_AVAILABLE:
        return jsonify({'error': 'Odds not available'}), 503
    try:
        game_lines = odds_manager.get_game_lines_for_date(date.today())
        return jsonify(serialize({
            'date': str(date.today()),
            'games': game_lines,
            'count': len(game_lines)
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nba/game-lines/date/<target_date>', methods=['GET'])
def nba_game_lines_by_date(target_date):
    """Get game lines for a specific date"""
    if not ODDS_AVAILABLE:
        return jsonify({'error': 'Odds not available'}), 503
    try:
        parsed = datetime.strptime(target_date, '%Y-%m-%d').date()
        game_lines = odds_manager.get_game_lines_for_date(parsed)
        return jsonify(serialize({
            'date': target_date,
            'games': game_lines,
            'count': len(game_lines)
        }))
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nba/games/<game_id>/odds', methods=['GET'])
def nba_game_odds(game_id):
    """Get all odds (game lines + player props) for a specific game"""
    if not ODDS_AVAILABLE:
        return jsonify({'error': 'Odds not available'}), 503
    try:
        odds = odds_manager.get_odds_for_game(game_id)
        return jsonify(serialize(odds))
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/nba/game-predictions/today', methods=['GET'])
def nba_game_predictions_today():
    """Get model's team predictions (spread, total) with DraftKings lines and edge calculation"""
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    
    try:
        today = date.today()
        result = get_game_predictions_with_odds(today)
        return jsonify(serialize(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nba/game-predictions/date/<target_date>', methods=['GET'])
def nba_game_predictions_by_date(target_date):
    """Get model's team predictions for a specific date with DK lines"""
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    
    try:
        parsed = datetime.strptime(target_date, '%Y-%m-%d').date()
        result = get_game_predictions_with_odds(parsed)
        return jsonify(serialize(result))
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def normalize_team_name(name):
    """Normalize team names for matching between different sources"""
    if not name:
        return ''
    # Remove common suffixes and normalize
    name = name.upper().strip()
    # Map full names to abbreviations
    team_map = {
        'ATL HAWKS': 'ATL', 'HAWKS': 'ATL', 'ATLANTA': 'ATL',
        'BOS CELTICS': 'BOS', 'CELTICS': 'BOS', 'BOSTON': 'BOS',
        'BKN NETS': 'BKN', 'NETS': 'BKN', 'BROOKLYN': 'BKN',
        'CHA HORNETS': 'CHA', 'HORNETS': 'CHA', 'CHARLOTTE': 'CHA',
        'CHI BULLS': 'CHI', 'BULLS': 'CHI', 'CHICAGO': 'CHI',
        'CLE CAVALIERS': 'CLE', 'CAVALIERS': 'CLE', 'CAVS': 'CLE', 'CLEVELAND': 'CLE',
        'DAL MAVERICKS': 'DAL', 'MAVERICKS': 'DAL', 'MAVS': 'DAL', 'DALLAS': 'DAL',
        'DEN NUGGETS': 'DEN', 'NUGGETS': 'DEN', 'DENVER': 'DEN',
        'DET PISTONS': 'DET', 'PISTONS': 'DET', 'DETROIT': 'DET',
        'GS WARRIORS': 'GSW', 'WARRIORS': 'GSW', 'GOLDEN STATE': 'GSW', 'GSW': 'GSW',
        'HOU ROCKETS': 'HOU', 'ROCKETS': 'HOU', 'HOUSTON': 'HOU',
        'IND PACERS': 'IND', 'PACERS': 'IND', 'INDIANA': 'IND',
        'LA CLIPPERS': 'LAC', 'CLIPPERS': 'LAC', 'LAC': 'LAC',
        'LA LAKERS': 'LAL', 'LAKERS': 'LAL', 'LAL': 'LAL', 'LOS ANGELES LAKERS': 'LAL',
        'MEM GRIZZLIES': 'MEM', 'GRIZZLIES': 'MEM', 'MEMPHIS': 'MEM',
        'MIA HEAT': 'MIA', 'HEAT': 'MIA', 'MIAMI': 'MIA',
        'MIL BUCKS': 'MIL', 'BUCKS': 'MIL', 'MILWAUKEE': 'MIL',
        'MIN TIMBERWOLVES': 'MIN', 'TIMBERWOLVES': 'MIN', 'WOLVES': 'MIN', 'MINNESOTA': 'MIN',
        'NO PELICANS': 'NOP', 'PELICANS': 'NOP', 'NEW ORLEANS': 'NOP', 'NOP': 'NOP',
        'NY KNICKS': 'NYK', 'KNICKS': 'NYK', 'NEW YORK': 'NYK', 'NYK': 'NYK',
        'OKC THUNDER': 'OKC', 'THUNDER': 'OKC', 'OKLAHOMA CITY': 'OKC',
        'ORL MAGIC': 'ORL', 'MAGIC': 'ORL', 'ORLANDO': 'ORL',
        'PHI 76ERS': 'PHI', '76ERS': 'PHI', 'SIXERS': 'PHI', 'PHILADELPHIA': 'PHI',
        'PHX SUNS': 'PHX', 'SUNS': 'PHX', 'PHOENIX': 'PHX',
        'POR TRAIL BLAZERS': 'POR', 'TRAIL BLAZERS': 'POR', 'BLAZERS': 'POR', 'PORTLAND': 'POR',
        'SAC KINGS': 'SAC', 'KINGS': 'SAC', 'SACRAMENTO': 'SAC',
        'SA SPURS': 'SAS', 'SPURS': 'SAS', 'SAN ANTONIO': 'SAS', 'SAS': 'SAS',
        'TOR RAPTORS': 'TOR', 'RAPTORS': 'TOR', 'TORONTO': 'TOR',
        'UTA JAZZ': 'UTA', 'JAZZ': 'UTA', 'UTAH': 'UTA',
        'WAS WIZARDS': 'WAS', 'WIZARDS': 'WAS', 'WASHINGTON': 'WAS',
    }
    return team_map.get(name, name[:3] if len(name) >= 3 else name)


def get_game_predictions_with_odds(target_date):
    """
    Combines model team predictions with DraftKings game lines
    Returns games with predicted spread/total, actual lines, and edge
    """
    games = {}
    
    # 1. Get our model's team predictions
    with nba_uploader.conn.cursor() as cur:
        cur.execute("""
            SELECT 
                game_id, game_date,
                home_team_id, home_team_abbrev,
                away_team_id, away_team_abbrev,
                prop_type, predicted_value, confidence,
                lower_bound, upper_bound
            FROM nba_team_predictions
            WHERE prediction_date = %s OR game_date = %s
            ORDER BY game_id, prop_type
        """, (target_date, target_date))
        
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            pred = dict(zip(cols, row))
            game_id = pred['game_id']
            
            if game_id not in games:
                games[game_id] = {
                    'game_id': game_id,
                    'game_date': str(pred['game_date']),
                    'home_team': pred['home_team_abbrev'],
                    'away_team': pred['away_team_abbrev'],
                    'home_team_id': pred['home_team_id'],
                    'away_team_id': pred['away_team_id'],
                    'predictions': {},
                    'dk_lines': {},
                    'edges': {}
                }
            
            prop_type = pred['prop_type']
            games[game_id]['predictions'][prop_type] = {
                'value': float(pred['predicted_value']),
                'confidence': float(pred['confidence']) if pred['confidence'] else None,
                'lower': float(pred['lower_bound']) if pred['lower_bound'] else None,
                'upper': float(pred['upper_bound']) if pred['upper_bound'] else None
            }
    
    # 2. Get DraftKings game lines directly from the database
    # Query the games and game_lines tables directly
    with nba_uploader.conn.cursor() as cur:
        cur.execute("""
            SELECT 
                g.id as games_table_id,
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
                gl.odds_decimal
            FROM games g
            LEFT JOIN game_lines gl ON g.id = gl.game_id
            WHERE DATE(g.start_time_mt) = %s
            ORDER BY g.start_time_mt, g.id, gl.line_type
        """, (target_date,))
        
        cols = [d[0] for d in cur.description]
        dk_rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    
    # Group DK data by game
    dk_games = {}
    for row in dk_rows:
        dk_id = row['dk_event_id']
        if dk_id not in dk_games:
            dk_games[dk_id] = {
                'dk_event_id': dk_id,
                'game_name': row['game_name'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'start_time': row['start_time_mt'],
                'moneyline': {'home': None, 'away': None},
                'spread': {'home': None, 'away': None},
                'total': {'over': None, 'under': None}
            }
        
        dk_game = dk_games[dk_id]
        line_type = row['line_type']
        
        if line_type == 'moneyline':
            # Check if this team is home or away
            team_abbrev = normalize_team_name(row['team'])
            home_abbrev = normalize_team_name(dk_game['home_team'])
            if team_abbrev == home_abbrev:
                dk_game['moneyline']['home'] = {
                    'team': row['team'],
                    'odds_american': row['odds_american'],
                    'odds_decimal': float(row['odds_decimal']) if row['odds_decimal'] else None
                }
            else:
                dk_game['moneyline']['away'] = {
                    'team': row['team'],
                    'odds_american': row['odds_american'],
                    'odds_decimal': float(row['odds_decimal']) if row['odds_decimal'] else None
                }
        
        elif line_type == 'spread':
            team_abbrev = normalize_team_name(row['team'])
            home_abbrev = normalize_team_name(dk_game['home_team'])
            spread_data = {
                'team': row['team'],
                'line': float(row['line']) if row['line'] is not None else None,
                'odds_american': row['odds_american'],
                'odds_decimal': float(row['odds_decimal']) if row['odds_decimal'] else None
            }
            if team_abbrev == home_abbrev:
                dk_game['spread']['home'] = spread_data
            else:
                dk_game['spread']['away'] = spread_data
        
        elif line_type == 'total':
            total_data = {
                'line': float(row['line']) if row['line'] is not None else None,
                'odds_american': row['odds_american'],
                'odds_decimal': float(row['odds_decimal']) if row['odds_decimal'] else None
            }
            if row['label'] and 'over' in row['label'].lower():
                dk_game['total']['over'] = total_data
            elif row['label'] and 'under' in row['label'].lower():
                dk_game['total']['under'] = total_data
    
    # 3. Match DK games to our predictions by team names
    for dk_id, dk_game in dk_games.items():
        dk_home = normalize_team_name(dk_game['home_team'])
        dk_away = normalize_team_name(dk_game['away_team'])
        
        # Find matching game in our predictions
        matched_game_id = None
        for gid, game in games.items():
            pred_home = normalize_team_name(game['home_team'])
            pred_away = normalize_team_name(game['away_team'])
            
            if (pred_home == dk_home and pred_away == dk_away) or \
               (pred_home == dk_home) or (pred_away == dk_away):
                matched_game_id = gid
                break
        
        if matched_game_id:
            game = games[matched_game_id]
            game['dk_event_id'] = dk_id
            game['start_time'] = str(dk_game['start_time']) if dk_game['start_time'] else None
            
            # Store DK lines
            if dk_game['spread']['home']:
                game['dk_lines']['spread'] = {
                    'home_line': dk_game['spread']['home'].get('line'),
                    'home_odds': dk_game['spread']['home'].get('odds_american'),
                    'away_line': dk_game['spread']['away'].get('line') if dk_game['spread']['away'] else None,
                    'away_odds': dk_game['spread']['away'].get('odds_american') if dk_game['spread']['away'] else None
                }
            
            if dk_game['total']['over']:
                game['dk_lines']['total'] = {
                    'line': dk_game['total']['over'].get('line'),
                    'over_odds': dk_game['total']['over'].get('odds_american'),
                    'under_odds': dk_game['total']['under'].get('odds_american') if dk_game['total']['under'] else None
                }
            
            if dk_game['moneyline']['home']:
                game['dk_lines']['moneyline'] = {
                    'home_odds': dk_game['moneyline']['home'].get('odds_american'),
                    'away_odds': dk_game['moneyline']['away'].get('odds_american') if dk_game['moneyline']['away'] else None
                }
        else:
            # No prediction for this DK game, but we should still show it
            # Create a new entry just with DK data
            games[dk_id] = {
                'game_id': dk_id,
                'game_date': str(target_date),
                'home_team': dk_home,
                'away_team': dk_away,
                'dk_event_id': dk_id,
                'start_time': str(dk_game['start_time']) if dk_game['start_time'] else None,
                'predictions': {},
                'dk_lines': {},
                'edges': {}
            }
            game = games[dk_id]
            
            if dk_game['spread']['home']:
                game['dk_lines']['spread'] = {
                    'home_line': dk_game['spread']['home'].get('line'),
                    'home_odds': dk_game['spread']['home'].get('odds_american'),
                    'away_line': dk_game['spread']['away'].get('line') if dk_game['spread']['away'] else None,
                    'away_odds': dk_game['spread']['away'].get('odds_american') if dk_game['spread']['away'] else None
                }
            
            if dk_game['total']['over']:
                game['dk_lines']['total'] = {
                    'line': dk_game['total']['over'].get('line'),
                    'over_odds': dk_game['total']['over'].get('odds_american'),
                    'under_odds': dk_game['total']['under'].get('odds_american') if dk_game['total']['under'] else None
                }
            
            if dk_game['moneyline']['home']:
                game['dk_lines']['moneyline'] = {
                    'home_odds': dk_game['moneyline']['home'].get('odds_american'),
                    'away_odds': dk_game['moneyline']['away'].get('odds_american') if dk_game['moneyline']['away'] else None
                }
    
    # 4. Calculate edges for games that have both predictions and DK lines
    # 
    # IMPORTANT: Sign conventions differ between our model and DraftKings!
    # 
    # Our model's spread prediction:
    #   - Positive (+13.5) = Home team wins by 13.5 points
    #   - Negative (-5.0) = Home team loses by 5 points (away wins)
    #
    # DraftKings spread convention:
    #   - Negative (-8.5) = Team is FAVORED to win by 8.5
    #   - Positive (+8.5) = Team is UNDERDOG, expected to lose by 8.5
    #
    # So if our model says Home wins by +13.5 and DK has Home at -8.5:
    #   - Both agree home wins, but model thinks by MORE
    #   - Model spread in DK terms would be -13.5
    #   - Edge = (-13.5) - (-8.5) = -5 points (model thinks home covers by 5 more)
    #   - Recommendation: Bet HOME to cover -8.5
    
    for gid, game in games.items():
        # Spread edge
        if 'spread' in game.get('predictions', {}) and 'spread' in game.get('dk_lines', {}):
            pred_spread = game['predictions']['spread']['value']  # Positive = home wins
            dk_spread = game['dk_lines']['spread'].get('home_line')  # Negative = home favored
            
            if dk_spread is not None:
                dk_spread_float = float(dk_spread)
                
                # Convert our prediction to DK convention (flip sign)
                # Our +13.5 (home wins by 13.5) = DK -13.5 (home favored by 13.5)
                model_spread_dk_convention = -pred_spread
                
                # Edge = how much better the line is than our prediction
                # If model says -13.5 and DK offers -8.5, that's +5 points of value for home
                spread_edge = dk_spread_float - model_spread_dk_convention
                
                # Positive edge = bet HOME (line is better than model expects)
                # Negative edge = bet AWAY (model thinks away will cover)
                if spread_edge > 1.5:
                    recommendation = 'home'
                elif spread_edge < -1.5:
                    recommendation = 'away'
                else:
                    recommendation = 'no_bet'
                
                game['edges']['spread'] = {
                    'edge_points': round(spread_edge, 1),
                    'model_spread_dk': round(model_spread_dk_convention, 1),
                    'recommendation': recommendation,
                    'confidence': 'high' if abs(spread_edge) > 3 else ('medium' if abs(spread_edge) > 1.5 else 'low'),
                    'explanation': f"Model: {game['home_team']} {model_spread_dk_convention:+.1f}, DK: {dk_spread_float:+.1f}"
                }
        
        # Total edge (this one is straightforward - both use same convention)
        if 'total' in game.get('predictions', {}) and 'total' in game.get('dk_lines', {}):
            pred_total = game['predictions']['total']['value']
            dk_total = game['dk_lines']['total'].get('line')
            if dk_total is not None:
                total_edge = pred_total - float(dk_total)
                game['edges']['total'] = {
                    'edge_points': round(total_edge, 1),
                    'recommendation': 'over' if total_edge > 3 else ('under' if total_edge < -3 else 'no_bet'),
                    'confidence': 'high' if abs(total_edge) > 5 else ('medium' if abs(total_edge) > 3 else 'low')
                }
    
    # Convert to list and sort
    games_list = list(games.values())
    games_list.sort(key=lambda x: x.get('start_time') or x.get('game_id', ''))
    
    # Count recommendations
    spread_bets = sum(1 for g in games_list 
                      if g.get('edges', {}).get('spread', {}).get('recommendation') 
                      and g['edges']['spread']['recommendation'] != 'no_bet')
    total_bets = sum(1 for g in games_list 
                     if g.get('edges', {}).get('total', {}).get('recommendation') 
                     and g['edges']['total']['recommendation'] != 'no_bet')
    
    return {
        'date': str(target_date),
        'games': games_list,
        'total_games': len(games_list),
        'spread_recommendations': spread_bets,
        'total_recommendations': total_bets,
        'odds_available': True,
        'dk_games_found': len(dk_games)
    }

# ============================================
# NHL Endpoints
# ============================================
@app.route('/api/nhl/predictions/today', methods=['GET'])
def nhl_today():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        data = nhl_uploader.get_recent_predictions(today_only=True)
        for game in data:
            if 'bet_pct_bankroll' in game:
                game['bet_pct_bankroll'] = float(game['bet_pct_bankroll'] or 0)
            if 'bet_size' in game:
                game['bet_size'] = float(game['bet_size'] or 0)
        return jsonify(serialize({'games': data, 'date': str(date.today()), 'bankroll': current_bankroll['nhl']}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nhl/predictions/date/<target_date>', methods=['GET'])
def nhl_by_date(target_date):
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        parsed = datetime.strptime(target_date, '%Y-%m-%d').date()
        predictions = nhl_uploader.get_predictions_by_date(parsed)
        return jsonify(serialize(predictions))
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nhl/predictions/history', methods=['GET'])
def nhl_history():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        data = nhl_uploader.get_recent_predictions(days=days, today_only=False)
        return jsonify(serialize({'predictions': data, 'days': days}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nhl/accuracy', methods=['GET'])
def nhl_accuracy():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        accuracy = nhl_uploader.get_accuracy_stats(days=days)
        return jsonify(serialize(accuracy))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# NHL Tracking endpoints
# ============================================
@app.route('/api/tracking/stats', methods=['GET'])
def get_tracking_stats():
    if not TRACKER_AVAILABLE:
        return jsonify({'error': 'Tracking not available'}), 503
    try:
        cursor = tracker.pg_conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as total_bets,
                SUM(CASE WHEN bet_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN bet_result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl, SUM(bet_size) as total_staked,
                AVG(edge) as avg_edge, AVG(model_probability) as avg_probability,
                MIN(game_date) as first_bet_date, MAX(game_date) as last_bet_date
            FROM bet_results
        """)
        overall = cursor.fetchone()
        
        cursor.execute("""
            SELECT CASE 
                    WHEN edge >= 0.08 THEN 'EXCEPTIONAL'
                    WHEN edge >= 0.05 THEN 'STRONG'
                    WHEN edge >= 0.03 THEN 'GOOD'
                    WHEN edge >= 0.01 THEN 'MODERATE'
                    ELSE 'MARGINAL'
                END as edge_class,
                COUNT(*) as bets,
                SUM(CASE WHEN bet_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(pnl) as pnl, AVG(edge) as avg_edge
            FROM bet_results GROUP BY 1 ORDER BY avg_edge DESC
        """)
        by_edge = cursor.fetchall()
        
        cursor.execute("""
            SELECT predicted_team, COUNT(*) as bets,
                SUM(CASE WHEN bet_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(pnl) as pnl
            FROM bet_results GROUP BY predicted_team HAVING COUNT(*) >= 3
            ORDER BY SUM(pnl) DESC LIMIT 10
        """)
        by_team = cursor.fetchall()
        
        cursor.execute("SELECT bet_result FROM bet_results ORDER BY game_date DESC, processed_at DESC LIMIT 10")
        recent = [row[0] for row in cursor.fetchall()]
        
        current_br = tracker.get_current_bankroll()
        
        def calc_streak(results):
            if not results: return {'type': 'NONE', 'length': 0}
            streak_type, length = results[0], 0
            for r in results:
                if r == streak_type: length += 1
                else: break
            return {'type': streak_type, 'length': length}
        
        return jsonify(serialize({
            'current_bankroll': current_br,
            'starting_bankroll': tracker.starting_bankroll,
            'total_return_pct': ((current_br - tracker.starting_bankroll) / tracker.starting_bankroll) * 100,
            'overall': {
                'total_bets': overall[0] or 0, 'wins': overall[1] or 0, 'losses': overall[2] or 0,
                'win_rate': (overall[1] / overall[0]) if overall[0] else 0,
                'total_pnl': float(overall[3]) if overall[3] else 0,
                'total_staked': float(overall[4]) if overall[4] else 0,
                'roi_pct': (float(overall[3]) / float(overall[4]) * 100) if overall[4] else 0,
                'avg_edge': float(overall[5]) if overall[5] else 0,
                'avg_probability': float(overall[6]) if overall[6] else 0,
            },
            'by_edge_class': [{'edge_class': r[0], 'bets': r[1], 'wins': r[2],
                'win_rate': r[2]/r[1] if r[1] else 0, 'pnl': float(r[3]) if r[3] else 0,
                'avg_edge': float(r[4]) if r[4] else 0} for r in by_edge],
            'top_teams': [{'team': r[0], 'bets': r[1], 'wins': r[2],
                'win_rate': r[2]/r[1] if r[1] else 0, 'pnl': float(r[3]) if r[3] else 0} for r in by_team],
            'recent_results': recent,
            'current_streak': calc_streak(recent)
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tracking/bets', methods=['GET'])
def get_bet_results():
    if not TRACKER_AVAILABLE:
        return jsonify({'error': 'Tracking not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        limit = request.args.get('limit', 100, type=int)
        result_filter = request.args.get('result', None)
        
        cursor = tracker.pg_conn.cursor()
        query = """
            SELECT game_id, prediction_date, game_date, home_team, away_team,
                predicted_team, predicted_winner, bet_size, decimal_odds, american_odds,
                model_probability, edge, home_score, away_score, actual_winner,
                bet_result, pnl, bankroll_after
            FROM bet_results WHERE game_date >= CURRENT_DATE - INTERVAL '%s days'
        """
        params = [days]
        if result_filter:
            query += " AND bet_result = %s"
            params.append(result_filter)
        query += " ORDER BY game_date DESC, processed_at DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, tuple(params))
        cols = [d[0] for d in cursor.description]
        results = [dict(zip(cols, row)) for row in cursor.fetchall()]
        
        return jsonify(serialize({'days': days, 'total_results': len(results), 'bets': results}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tracking/performance', methods=['GET'])
def get_tracking_performance():
    if not TRACKER_AVAILABLE:
        return jsonify({'error': 'Tracking not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        data = tracker.get_performance_data(days=days)
        return jsonify(serialize(data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# Bankroll
# ============================================
@app.route('/api/bankroll/status', methods=['GET'])
def get_bankroll_status():
    return jsonify({
        'nba': current_bankroll['nba'],
        'nhl': current_bankroll['nhl'],
        'updated_at': current_bankroll['updated_at']
    })

@app.route('/api/bankroll/update', methods=['POST'])
def update_bankroll():
    try:
        data = request.get_json()
        sport = data.get('sport', 'nhl')
        new_amount = float(data.get('bankroll', current_bankroll.get(sport, 1000)))
        current_bankroll[sport] = new_amount
        current_bankroll['updated_at'] = datetime.now().isoformat()
        return jsonify({'success': True, 'bankroll': current_bankroll})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# Main
# ============================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🎯 UNIFIED SPORTS PREDICTIONS API - Enhanced")
    print("=" * 60)
    print(f"🏀 NBA: {'Available' if NBA_AVAILABLE else 'Not Available'}")
    print(f"🏒 NHL: {'Available' if NHL_AVAILABLE else 'Not Available'}")
    print(f"💰 Odds: {'Available' if ODDS_AVAILABLE else 'Not Available'}")
    print(f"📊 Tracking: {'Available' if TRACKER_AVAILABLE else 'Not Available'}")
    print("\nEndpoints:")
    print("  GET  /api/health")
    print("  GET  /api/nba/predictions/today")
    print("  GET  /api/nba/odds/today")
    print("  GET  /api/nba/predictions/with-edge")
    print("  POST /api/nba/bets/save-all")
    print("  GET  /api/nba/bets/pending")
    print("  GET  /api/nba/bets/history")
    print("  POST /api/nba/bets/result")
    print("  GET  /api/nba/bets/performance")
    print("\n🌐 Running on http://localhost:5001")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)