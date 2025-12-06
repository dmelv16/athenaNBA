"""
Unified Sports API Package - Enhanced Version
Supports NBA and NHL predictions with full historical data, odds, and results tracking
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

# Initialize Uploaders
NBA_AVAILABLE = False
NHL_AVAILABLE = False
nba_uploader = None
nhl_uploader = None

try:
    from models.api.db_uploader import NBAPredictionUploader
    nba_uploader = NBAPredictionUploader()
    NBA_AVAILABLE = True
    print("✓ NBA module loaded")
except Exception as e:
    print(f"⚠ NBA module not available: {e}")

try:
    from src.connection.db_uploader import PostgresPredictionUploader
    nhl_uploader = PostgresPredictionUploader()
    NHL_AVAILABLE = True
    print("✓ NHL module loaded")
except Exception as e:
    print(f"⚠ NHL module not available: {e}")

def serialize_predictions(data):
    if isinstance(data, list):
        return [serialize_predictions(item) for item in data]
    if isinstance(data, dict):
        return {k: serialize_predictions(v) for k, v in data.items()}
    if isinstance(data, Decimal):
        return float(data)
    if isinstance(data, (datetime, date)):
        return str(data)
    if hasattr(data, 'item'):
        return data.item()
    return data

# ============================================
# General Endpoints
# ============================================
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'sports': {
            'nba': 'available' if NBA_AVAILABLE else 'unavailable',
            'nhl': 'available' if NHL_AVAILABLE else 'unavailable'
        }
    })

@app.route('/api/sports', methods=['GET'])
def list_sports():
    sports = []
    if NBA_AVAILABLE:
        sports.append({'id': 'nba', 'name': 'NBA Basketball', 'icon': '🏀'})
    if NHL_AVAILABLE:
        sports.append({'id': 'nhl', 'name': 'NHL Hockey', 'icon': '🏒'})
    return jsonify({'sports': sports})

# ============================================
# NBA Endpoints
# ============================================
@app.route('/api/nba/predictions/today', methods=['GET'])
def nba_today():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        data = nba_uploader.get_todays_predictions()
        return jsonify(serialize_predictions(data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/predictions/date/<target_date>', methods=['GET'])
def nba_by_date(target_date):
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        parsed = datetime.strptime(target_date, '%Y-%m-%d').date()
        data = nba_uploader.get_predictions_by_date(parsed)
        return jsonify(serialize_predictions(data))
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/predictions/history', methods=['GET'])
def nba_history():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        data = nba_uploader.get_recent_predictions(days=days)
        return jsonify(serialize_predictions({'predictions': data, 'days': days}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/predictions/best-bets', methods=['GET'])
def nba_best_bets():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        min_conf = request.args.get('min_confidence', 0.65, type=float)
        min_edge = request.args.get('min_edge', 2.0, type=float)
        data = nba_uploader.get_best_bets(min_conf, min_edge)
        return jsonify(serialize_predictions({'predictions': data}))
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
            
            cur.execute(
                "SELECT * FROM nba_team_predictions WHERE game_id = %s",
                (game_id,)
            )
            cols = [d[0] for d in cur.description]
            team_preds = [dict(zip(cols, r)) for r in cur.fetchall()]
        
        return jsonify(serialize_predictions({
            'game_id': game_id,
            'player_predictions': player_preds,
            'team_predictions': team_preds
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# NEW: Player historical predictions endpoint
@app.route('/api/nba/players/<int:player_id>/history', methods=['GET'])
def nba_player_history(player_id):
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        limit = request.args.get('limit', 5, type=int)
        prop_type = request.args.get('prop_type', None)
        
        query = """
            SELECT 
                p.prediction_date,
                p.game_id,
                p.game_date,
                p.player_name,
                p.team_abbrev,
                p.opponent_abbrev,
                p.prop_type,
                p.predicted_value,
                p.line,
                p.confidence,
                p.recommended_bet,
                pgl.pts as actual_pts,
                pgl.reb as actual_reb,
                pgl.ast as actual_ast,
                pgl.pts + pgl.reb + pgl.ast as actual_pra
            FROM nba_player_predictions p
            LEFT JOIN player_game_logs pgl 
                ON p.player_id = pgl.player_id 
                AND p.game_id = pgl.game_id
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
        
        # Calculate actual value based on prop type and hit/miss
        for r in results:
            prop = r['prop_type']
            if prop == 'pts':
                r['actual_value'] = r.get('actual_pts')
            elif prop == 'reb':
                r['actual_value'] = r.get('actual_reb')
            elif prop == 'ast':
                r['actual_value'] = r.get('actual_ast')
            elif prop == 'pra':
                r['actual_value'] = r.get('actual_pra')
            elif prop == 'pr':
                r['actual_value'] = (r.get('actual_pts') or 0) + (r.get('actual_reb') or 0)
            elif prop == 'pa':
                r['actual_value'] = (r.get('actual_pts') or 0) + (r.get('actual_ast') or 0)
            elif prop == 'ra':
                r['actual_value'] = (r.get('actual_reb') or 0) + (r.get('actual_ast') or 0)
            else:
                r['actual_value'] = None
            
            # Determine hit/miss
            if r['actual_value'] is not None and r.get('line'):
                if r['recommended_bet'] == 'over':
                    r['hit'] = r['actual_value'] > r['line']
                elif r['recommended_bet'] == 'under':
                    r['hit'] = r['actual_value'] < r['line']
                else:
                    r['hit'] = None
            else:
                r['hit'] = None
        
        return jsonify(serialize_predictions({
            'player_id': player_id,
            'history': results
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nba/accuracy', methods=['GET'])
def nba_accuracy():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        
        query = """
            WITH prediction_results AS (
                SELECT 
                    p.prop_type,
                    p.predicted_value,
                    p.line,
                    p.confidence,
                    p.prediction_date,
                    p.recommended_bet,
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
                    ON p.player_id = pgl.player_id 
                    AND p.game_id = pgl.game_id
                WHERE p.prediction_date >= %s
                    AND p.prop_type IN ('pts', 'reb', 'ast', 'pra', 'pr', 'pa', 'ra')
            )
            SELECT 
                prop_type,
                COUNT(*) as total_predictions,
                AVG(ABS(predicted_value - actual_value)) as avg_error,
                AVG(confidence) as avg_confidence,
                SUM(CASE 
                    WHEN line IS NOT NULL AND (
                        (recommended_bet = 'over' AND actual_value > line) OR
                        (recommended_bet = 'under' AND actual_value < line)
                    ) THEN 1 ELSE 0 
                END)::float / NULLIF(SUM(CASE WHEN line IS NOT NULL THEN 1 ELSE 0 END), 0) as line_accuracy
            FROM prediction_results
            WHERE actual_value IS NOT NULL
            GROUP BY prop_type
            ORDER BY prop_type
        """
        
        since_date = date.today() - timedelta(days=days)
        
        with nba_uploader.conn.cursor() as cur:
            cur.execute(query, (since_date,))
            cols = [d[0] for d in cur.description]
            results = [dict(zip(cols, r)) for r in cur.fetchall()]
        
        return jsonify(serialize_predictions({
            'days': days,
            'accuracy_by_prop': results
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# NHL Endpoints - Enhanced
# ============================================
@app.route('/api/nhl/predictions/today', methods=['GET'])
def nhl_today():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        data = nhl_uploader.get_recent_predictions(today_only=True)
        return jsonify(serialize_predictions({'games': data, 'date': str(date.today())}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nhl/predictions/date/<target_date>', methods=['GET'])
def nhl_by_date(target_date):
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        parsed = datetime.strptime(target_date, '%Y-%m-%d').date()
        predictions = nhl_uploader.get_predictions_by_date(parsed)
        return jsonify(serialize_predictions(predictions))
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nhl/predictions/history', methods=['GET'])
def nhl_history():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        data = nhl_uploader.get_recent_predictions(days=days, today_only=False)
        return jsonify(serialize_predictions({'predictions': data, 'days': days}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/nhl/predictions/summary', methods=['GET'])
def nhl_summary():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        data = nhl_uploader.get_betting_summary(days=days)
        return jsonify(serialize_predictions(data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# NEW: NHL Results tracking
@app.route('/api/nhl/results/history', methods=['GET'])
def nhl_results_history():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        results = nhl_uploader.get_prediction_results(days=days)
        return jsonify(serialize_predictions(results))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# NEW: NHL Accuracy tracking
@app.route('/api/nhl/accuracy', methods=['GET'])
def nhl_accuracy():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        accuracy = nhl_uploader.get_accuracy_stats(days=days)
        return jsonify(serialize_predictions(accuracy))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# NEW: NHL Best bets
@app.route('/api/nhl/predictions/best-bets', methods=['GET'])
def nhl_best_bets():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        min_edge = request.args.get('min_edge', 0.05, type=float)
        min_prob = request.args.get('min_probability', 0.55, type=float)
        data = nhl_uploader.get_best_bets(min_edge=min_edge, min_probability=min_prob)
        return jsonify(serialize_predictions({'predictions': data}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🎯 UNIFIED SPORTS PREDICTIONS API")
    print("=" * 60)
    print(f"🏀 NBA: {'Available' if NBA_AVAILABLE else 'Not Available'}")
    print(f"🏒 NHL: {'Available' if NHL_AVAILABLE else 'Not Available'}")
    print("\nEndpoints:")
    print("  GET /api/health")
    print("  GET /api/sports")
    print("  GET /api/nba/predictions/today")
    print("  GET /api/nba/predictions/date/<YYYY-MM-DD>")
    print("  GET /api/nba/predictions/history?days=30")
    print("  GET /api/nba/predictions/best-bets")
    print("  GET /api/nba/players/<player_id>/history")
    print("  GET /api/nba/accuracy?days=30")
    print("  GET /api/nhl/predictions/today")
    print("  GET /api/nhl/predictions/date/<YYYY-MM-DD>")
    print("  GET /api/nhl/predictions/history?days=30")
    print("  GET /api/nhl/predictions/best-bets")
    print("  GET /api/nhl/accuracy?days=30")
    print("  GET /api/nhl/results/history?days=30")
    print("=" * 60)
    print("🌐 Running on http://localhost:5001")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)