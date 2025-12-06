import sys
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import date, datetime, timedelta
from decimal import Decimal
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ============================================
# Custom JSON Encoder for Decimal/Date types
# ============================================
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# ============================================
# Initialize Uploaders
# ============================================
NBA_AVAILABLE = False
NHL_AVAILABLE = False
nba_uploader = None
nhl_uploader = None

# Try NBA
try:
    from models.api.db_uploader import NBAPredictionUploader
    nba_uploader = NBAPredictionUploader()
    NBA_AVAILABLE = True
    print("✓ NBA module loaded")
except Exception as e:
    print(f"⚠ NBA module not available: {e}")

# Try NHL
try:
    from src.connection.db_uploader import PostgresPredictionUploader
    nhl_uploader = PostgresPredictionUploader()
    NHL_AVAILABLE = True
    print("✓ NHL module loaded")
except Exception as e:
    print(f"⚠ NHL module not available: {e}")

# ============================================
# Helper Functions
# ============================================
def serialize_predictions(data):
    """Ensure all data is JSON serializable"""
    if isinstance(data, list):
        return [serialize_predictions(item) for item in data]
    if isinstance(data, dict):
        return {k: serialize_predictions(v) for k, v in data.items()}
    if isinstance(data, Decimal):
        return float(data)
    if isinstance(data, (datetime, date)):
        return str(data)
    if hasattr(data, 'item'):  # numpy types
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

@app.route('/api/nba/predictions/parlays', methods=['GET'])
def nba_parlays():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        target = request.args.get('date')
        if target:
            parsed = datetime.strptime(target, '%Y-%m-%d').date()
        else:
            parsed = date.today()
        data = nba_uploader.get_parlays(parsed)
        return jsonify(serialize_predictions({'parlays': data}))
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

@app.route('/api/nba/accuracy', methods=['GET'])
def nba_accuracy():
    """Get accuracy stats for NBA predictions"""
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        
        # Query to calculate accuracy by comparing predictions to actual results
        query = """
            WITH prediction_results AS (
                SELECT 
                    p.prop_type,
                    p.predicted_value,
                    p.line,
                    p.confidence,
                    p.prediction_date,
                    pgl.pts, pgl.reb, pgl.ast,
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
                        (predicted_value > line AND actual_value > line) OR
                        (predicted_value < line AND actual_value < line)
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
# NHL Endpoints
# ============================================
@app.route('/api/nhl/predictions/today', methods=['GET'])
def nhl_today():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        data = nhl_uploader.get_recent_predictions(today_only=True)
        return jsonify(serialize_predictions({'games': data}))
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

# ============================================
# Run Server
# ============================================
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
    print("  GET /api/nba/accuracy?days=30")
    print("  GET /api/nhl/predictions/today")
    print("  GET /api/nhl/predictions/history?days=30")
    print("=" * 60)
    print("🌐 Running on http://localhost:5001")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)