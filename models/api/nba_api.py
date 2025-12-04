"""
Multi-Sport Predictions API
Serves NBA and NHL predictions to React frontend

Run: python api/sports_api.py
"""

from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
from datetime import date, datetime, timedelta
import sys
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ============================================
# NBA Blueprint
# ============================================
nba_bp = Blueprint('nba', __name__, url_prefix='/api/nba')

# Import NBA uploader
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.api.db_uploader import NBAPredictionUploader
    nba_uploader = NBAPredictionUploader()
    NBA_AVAILABLE = True
except Exception as e:
    print(f"⚠️  NBA module not available: {e}")
    NBA_AVAILABLE = False
    nba_uploader = None


@nba_bp.route('/predictions/today', methods=['GET'])
def nba_today():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA predictions not available'}), 503
    try:
        predictions = nba_uploader.get_todays_predictions()
        return jsonify(predictions), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nba_bp.route('/predictions/date/<target_date>', methods=['GET'])
def nba_by_date(target_date: str):
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA predictions not available'}), 503
    try:
        parsed_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        predictions = nba_uploader.get_predictions_by_date(parsed_date)
        return jsonify(predictions), 200
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nba_bp.route('/predictions/best-bets', methods=['GET'])
def nba_best_bets():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA predictions not available'}), 503
    try:
        min_conf = request.args.get('min_confidence', default=0.65, type=float)
        min_edge = request.args.get('min_edge', default=2.0, type=float)
        best_bets = nba_uploader.get_best_bets(min_conf, min_edge)
        return jsonify({
            'date': str(date.today()),
            'total': len(best_bets),
            'predictions': best_bets
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nba_bp.route('/predictions/parlays', methods=['GET'])
def nba_parlays():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA predictions not available'}), 503
    try:
        parlays = nba_uploader.get_parlays()
        return jsonify({
            'date': str(date.today()),
            'total': len(parlays),
            'parlays': parlays
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nba_bp.route('/predictions/history', methods=['GET'])
def nba_history():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA predictions not available'}), 503
    try:
        days = request.args.get('days', default=7, type=int)
        predictions = nba_uploader.get_recent_predictions(days=days)
        return jsonify({
            'days': days,
            'total': len(predictions),
            'predictions': predictions
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nba_bp.route('/games/<game_id>', methods=['GET'])
def nba_game(game_id: str):
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA predictions not available'}), 503
    try:
        # Get player and team predictions for game
        player_query = """
            SELECT * FROM nba_player_predictions
            WHERE game_id = %s ORDER BY confidence DESC
        """
        team_query = """
            SELECT * FROM nba_team_predictions
            WHERE game_id = %s
        """
        
        with nba_uploader.conn.cursor() as cur:
            cur.execute(player_query, (game_id,))
            cols = [d[0] for d in cur.description]
            player_preds = [dict(zip(cols, r)) for r in cur.fetchall()]
            
            cur.execute(team_query, (game_id,))
            cols = [d[0] for d in cur.description]
            team_preds = [dict(zip(cols, r)) for r in cur.fetchall()]
        
        # Serialize
        for p in player_preds + team_preds:
            for k, v in p.items():
                if isinstance(v, (date, datetime)):
                    p[k] = str(v)
                elif hasattr(v, 'item'):
                    p[k] = float(v)
        
        return jsonify({
            'game_id': game_id,
            'team_predictions': team_preds,
            'player_predictions': player_preds
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nba_bp.route('/players/search', methods=['GET'])
def nba_player_search():
    if not NBA_AVAILABLE:
        return jsonify({'error': 'NBA predictions not available'}), 503
    try:
        q = request.args.get('q', '', type=str)
        if len(q) < 2:
            return jsonify({'error': 'Query must be at least 2 characters'}), 400
        
        query = """
            SELECT DISTINCT player_id, player_name, team_abbrev
            FROM nba_player_predictions
            WHERE LOWER(player_name) LIKE LOWER(%s)
            LIMIT 20
        """
        
        with nba_uploader.conn.cursor() as cur:
            cur.execute(query, (f'%{q}%',))
            results = [{'id': r[0], 'name': r[1], 'team': r[2]} for r in cur.fetchall()]
        
        return jsonify({'query': q, 'results': results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nba_bp.route('/health', methods=['GET'])
def nba_health():
    return jsonify({
        'status': 'healthy' if NBA_AVAILABLE else 'unavailable',
        'sport': 'nba',
        'timestamp': datetime.now().isoformat()
    }), 200 if NBA_AVAILABLE else 503


# ============================================
# NHL Blueprint (integrate your existing NHL)
# ============================================
nhl_bp = Blueprint('nhl', __name__, url_prefix='/api/nhl')

# Import NHL uploader
try:
    from src.connection.db_uploader import PostgresPredictionUploader
    nhl_uploader = PostgresPredictionUploader()
    NHL_AVAILABLE = True
except Exception as e:
    print(f"⚠️  NHL module not available: {e}")
    NHL_AVAILABLE = False
    nhl_uploader = None


@nhl_bp.route('/predictions/today', methods=['GET'])
def nhl_today():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL predictions not available'}), 503
    try:
        predictions = nhl_uploader.get_recent_predictions(today_only=True)
        return jsonify({
            'date': str(date.today()),
            'total_games': len(predictions),
            'games': predictions
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nhl_bp.route('/predictions/history', methods=['GET'])
def nhl_history():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL predictions not available'}), 503
    try:
        days = request.args.get('days', default=7, type=int)
        predictions = nhl_uploader.get_recent_predictions(days=days, today_only=False)
        return jsonify({
            'days': days,
            'total': len(predictions),
            'predictions': predictions
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nhl_bp.route('/predictions/summary', methods=['GET'])
def nhl_summary():
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL predictions not available'}), 503
    try:
        days = request.args.get('days', default=7, type=int)
        summary = nhl_uploader.get_betting_summary(days=days)
        return jsonify(summary), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@nhl_bp.route('/health', methods=['GET'])
def nhl_health():
    return jsonify({
        'status': 'healthy' if NHL_AVAILABLE else 'unavailable',
        'sport': 'nhl',
        'timestamp': datetime.now().isoformat()
    }), 200 if NHL_AVAILABLE else 503


# ============================================
# General Endpoints
# ============================================

@app.route('/api/health', methods=['GET'])
def global_health():
    """Overall API health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'date': str(date.today()),
        'sports': {
            'nba': 'available' if NBA_AVAILABLE else 'unavailable',
            'nhl': 'available' if NHL_AVAILABLE else 'unavailable'
        }
    }), 200


@app.route('/api/sports', methods=['GET'])
def list_sports():
    """List available sports"""
    sports = []
    
    if NBA_AVAILABLE:
        sports.append({
            'id': 'nba',
            'name': 'NBA Basketball',
            'icon': '🏀',
            'endpoints': '/api/nba/*'
        })
    
    if NHL_AVAILABLE:
        sports.append({
            'id': 'nhl', 
            'name': 'NHL Hockey',
            'icon': '🏒',
            'endpoints': '/api/nhl/*'
        })
    
    return jsonify({
        'total': len(sports),
        'sports': sports
    }), 200


@app.route('/api/predictions/today', methods=['GET'])
def all_predictions_today():
    """Get today's predictions for all sports"""
    result = {
        'date': str(date.today()),
        'sports': {}
    }
    
    if NBA_AVAILABLE:
        try:
            result['sports']['nba'] = nba_uploader.get_todays_predictions()
        except:
            result['sports']['nba'] = {'error': 'Failed to load'}
    
    if NHL_AVAILABLE:
        try:
            preds = nhl_uploader.get_recent_predictions(today_only=True)
            result['sports']['nhl'] = {'games': preds, 'total': len(preds)}
        except:
            result['sports']['nhl'] = {'error': 'Failed to load'}
    
    return jsonify(result), 200


# ============================================
# Register Blueprints
# ============================================
app.register_blueprint(nba_bp)
app.register_blueprint(nhl_bp)


# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================
# Run Server
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("🎯 MULTI-SPORT PREDICTIONS API")
    print("=" * 60)
    print(f"\n📅 Date: {date.today()}")
    print(f"🏀 NBA: {'Available' if NBA_AVAILABLE else 'Not Available'}")
    print(f"🏒 NHL: {'Available' if NHL_AVAILABLE else 'Not Available'}")
    
    print("\n📊 Available Endpoints:")
    print("\n  General:")
    print("    GET /api/health")
    print("    GET /api/sports")
    print("    GET /api/predictions/today")
    
    if NBA_AVAILABLE:
        print("\n  NBA:")
        print("    GET /api/nba/predictions/today")
        print("    GET /api/nba/predictions/date/<YYYY-MM-DD>")
        print("    GET /api/nba/predictions/best-bets")
        print("    GET /api/nba/predictions/parlays")
        print("    GET /api/nba/predictions/history?days=7")
        print("    GET /api/nba/games/<game_id>")
        print("    GET /api/nba/players/search?q=<name>")
        print("    GET /api/nba/health")
    
    if NHL_AVAILABLE:
        print("\n  NHL:")
        print("    GET /api/nhl/predictions/today")
        print("    GET /api/nhl/predictions/history?days=7")
        print("    GET /api/nhl/predictions/summary")
        print("    GET /api/nhl/health")
    
    print("\n" + "=" * 60)
    print("🌐 Server running on http://localhost:5001")
    print("=" * 60 + "\n")
    print("\n📋 Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.rule} -> {rule.endpoint}")
    app.run(debug=True, host='0.0.0.0', port=5001)