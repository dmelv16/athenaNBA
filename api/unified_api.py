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

from src.tracking.historical_tracker import HistoricalGameTracker

# ============================================
# Initialize Tracker (add after other initializations)
# ============================================
TRACKER_AVAILABLE = False
tracker = None

try:
    MSSQL_CONN_STR = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-J9IV3OH;"
        "DATABASE=nhlDB;"
        "Trusted_Connection=yes;"
    )
    
    tracker = HistoricalGameTracker(
        mssql_connection_string=MSSQL_CONN_STR,
        starting_bankroll=1000.0
    )
    tracker.connect()
    tracker.ensure_tracking_tables()
    TRACKER_AVAILABLE = True
    print("✓ Historical tracker loaded")
except Exception as e:
    print(f"⚠ Historical tracker not available: {e}")

# ============================================
# Historical Tracking Endpoints
# ============================================

@app.route('/api/tracking/performance', methods=['GET'])
def get_tracking_performance():
    """Get overall betting performance metrics"""
    if not TRACKER_AVAILABLE:
        return jsonify({'error': 'Tracking not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        data = tracker.get_performance_data(days=days)
        return jsonify(serialize_predictions(data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tracking/bankroll', methods=['GET'])
def get_bankroll_history():
    """Get bankroll history over time"""
    if not TRACKER_AVAILABLE:
        return jsonify({'error': 'Tracking not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        
        cursor = tracker.pg_conn.cursor()
        cursor.execute("""
            SELECT 
                snapshot_date,
                starting_bankroll,
                ending_bankroll,
                daily_pnl,
                bets_placed,
                bets_won,
                bets_lost,
                win_rate,
                total_staked,
                roi_pct
            FROM bankroll_history
            WHERE snapshot_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY snapshot_date DESC
        """, (days,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify(serialize_predictions({
            'days': days,
            'history': results,
            'current_bankroll': tracker.get_current_bankroll()
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tracking/bets', methods=['GET'])
def get_bet_results():
    """Get individual bet results"""
    if not TRACKER_AVAILABLE:
        return jsonify({'error': 'Tracking not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        limit = request.args.get('limit', 100, type=int)
        result_filter = request.args.get('result', None)  # 'WIN', 'LOSS', or None
        
        cursor = tracker.pg_conn.cursor()
        
        query = """
            SELECT 
                game_id, prediction_date, game_date,
                home_team, away_team, predicted_team, predicted_winner,
                bet_size, decimal_odds, american_odds,
                model_probability, edge,
                home_score, away_score, actual_winner,
                bet_result, pnl, bankroll_after
            FROM bet_results
            WHERE game_date >= CURRENT_DATE - INTERVAL '%s days'
        """
        params = [days]
        
        if result_filter:
            query += " AND bet_result = %s"
            params.append(result_filter)
        
        query += " ORDER BY game_date DESC, processed_at DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, tuple(params))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify(serialize_predictions({
            'days': days,
            'total_results': len(results),
            'bets': results
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tracking/update', methods=['POST'])
def run_tracking_update():
    """Manually trigger tracking update for recent games"""
    if not TRACKER_AVAILABLE:
        return jsonify({'error': 'Tracking not available'}), 503
    try:
        tracker.run_daily_update()
        return jsonify({
            'status': 'success',
            'message': 'Tracking update completed',
            'current_bankroll': tracker.get_current_bankroll()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tracking/backfill', methods=['POST'])
def run_tracking_backfill():
    """Run historical backfill (use with caution)"""
    if not TRACKER_AVAILABLE:
        return jsonify({'error': 'Tracking not available'}), 503
    try:
        data = request.get_json() or {}
        days = data.get('days', 30)
        reset = data.get('reset_bankroll', False)
        starting_bankroll = data.get('starting_bankroll', 1000.0)
        
        if reset:
            tracker.starting_bankroll = starting_bankroll
        
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)
        
        tracker.run_backfill(
            start_date=start_date,
            end_date=end_date,
            reset_bankroll=reset
        )
        
        return jsonify({
            'status': 'success',
            'message': f'Backfill completed for {days} days',
            'current_bankroll': tracker.get_current_bankroll()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tracking/stats', methods=['GET'])
def get_tracking_stats():
    """Get detailed tracking statistics"""
    if not TRACKER_AVAILABLE:
        return jsonify({'error': 'Tracking not available'}), 503
    try:
        cursor = tracker.pg_conn.cursor()
        
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_bets,
                SUM(CASE WHEN bet_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN bet_result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                SUM(bet_size) as total_staked,
                AVG(edge) as avg_edge,
                AVG(model_probability) as avg_probability,
                MIN(game_date) as first_bet_date,
                MAX(game_date) as last_bet_date
            FROM bet_results
        """)
        overall = cursor.fetchone()
        
        # Stats by edge class (approximated by edge ranges)
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN edge >= 0.08 THEN 'EXCEPTIONAL'
                    WHEN edge >= 0.05 THEN 'STRONG'
                    WHEN edge >= 0.03 THEN 'GOOD'
                    WHEN edge >= 0.01 THEN 'MODERATE'
                    ELSE 'MARGINAL'
                END as edge_class,
                COUNT(*) as bets,
                SUM(CASE WHEN bet_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(pnl) as pnl,
                AVG(edge) as avg_edge
            FROM bet_results
            GROUP BY 
                CASE 
                    WHEN edge >= 0.08 THEN 'EXCEPTIONAL'
                    WHEN edge >= 0.05 THEN 'STRONG'
                    WHEN edge >= 0.03 THEN 'GOOD'
                    WHEN edge >= 0.01 THEN 'MODERATE'
                    ELSE 'MARGINAL'
                END
            ORDER BY avg_edge DESC
        """)
        by_edge = cursor.fetchall()
        
        # Stats by team
        cursor.execute("""
            SELECT 
                predicted_team,
                COUNT(*) as bets,
                SUM(CASE WHEN bet_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(pnl) as pnl
            FROM bet_results
            GROUP BY predicted_team
            HAVING COUNT(*) >= 3
            ORDER BY SUM(pnl) DESC
            LIMIT 10
        """)
        by_team = cursor.fetchall()
        
        # Recent streak
        cursor.execute("""
            SELECT bet_result
            FROM bet_results
            ORDER BY game_date DESC, processed_at DESC
            LIMIT 10
        """)
        recent = [row[0] for row in cursor.fetchall()]
        
        current_bankroll = tracker.get_current_bankroll()
        
        return jsonify(serialize_predictions({
            'current_bankroll': current_bankroll,
            'starting_bankroll': tracker.starting_bankroll,
            'total_return_pct': ((current_bankroll - tracker.starting_bankroll) / tracker.starting_bankroll) * 100,
            'overall': {
                'total_bets': overall[0] or 0,
                'wins': overall[1] or 0,
                'losses': overall[2] or 0,
                'win_rate': (overall[1] / overall[0]) if overall[0] else 0,
                'total_pnl': float(overall[3]) if overall[3] else 0,
                'total_staked': float(overall[4]) if overall[4] else 0,
                'roi_pct': (float(overall[3]) / float(overall[4]) * 100) if overall[4] else 0,
                'avg_edge': float(overall[5]) if overall[5] else 0,
                'avg_probability': float(overall[6]) if overall[6] else 0,
                'first_bet': str(overall[7]) if overall[7] else None,
                'last_bet': str(overall[8]) if overall[8] else None
            },
            'by_edge_class': [
                {
                    'edge_class': row[0],
                    'bets': row[1],
                    'wins': row[2],
                    'win_rate': row[2] / row[1] if row[1] else 0,
                    'pnl': float(row[3]) if row[3] else 0,
                    'avg_edge': float(row[4]) if row[4] else 0
                }
                for row in by_edge
            ],
            'top_teams': [
                {
                    'team': row[0],
                    'bets': row[1],
                    'wins': row[2],
                    'win_rate': row[2] / row[1] if row[1] else 0,
                    'pnl': float(row[3]) if row[3] else 0
                }
                for row in by_team
            ],
            'recent_results': recent,
            'current_streak': calculate_streak(recent)
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def calculate_streak(results: list) -> dict:
    """Calculate current win/loss streak"""
    if not results:
        return {'type': 'NONE', 'length': 0}
    
    streak_type = results[0]
    streak_length = 0
    
    for result in results:
        if result == streak_type:
            streak_length += 1
        else:
            break
    
    return {'type': streak_type, 'length': streak_length}
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
        },
        'tracking': 'available' if TRACKER_AVAILABLE else 'unavailable'
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
# Bankroll Management Endpoints
# ============================================

# Store bankroll in memory (in production, use database)
current_bankroll = {"amount": 1412.77, "updated_at": datetime.now().isoformat()}

@app.route('/api/bankroll/status', methods=['GET'])
def get_bankroll_status():
    """Get current bankroll status"""
    try:
        # Try to get from NHL uploader's summary
        if NHL_AVAILABLE:
            summary = nhl_uploader.get_betting_summary(days=30)
            return jsonify({
                'bankroll': current_bankroll['amount'],
                'updated_at': current_bankroll['updated_at'],
                'summary': serialize_predictions(summary)
            })
        return jsonify({
            'bankroll': current_bankroll['amount'],
            'updated_at': current_bankroll['updated_at']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bankroll/update', methods=['POST'])
def update_bankroll():
    """Update current bankroll"""
    try:
        data = request.get_json()
        new_amount = float(data.get('bankroll', current_bankroll['amount']))
        current_bankroll['amount'] = new_amount
        current_bankroll['updated_at'] = datetime.now().isoformat()
        return jsonify({
            'success': True,
            'bankroll': current_bankroll['amount'],
            'updated_at': current_bankroll['updated_at']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    """Get today's NHL predictions with full bet details"""
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        data = nhl_uploader.get_recent_predictions(today_only=True)
        
        # Ensure bet_pct_bankroll is properly included
        for game in data:
            # Ensure numeric fields are properly formatted
            if 'bet_pct_bankroll' in game:
                game['bet_pct_bankroll'] = float(game['bet_pct_bankroll'] or 0)
            if 'bet_size' in game:
                game['bet_size'] = float(game['bet_size'] or 0)
            if 'expected_value' in game:
                game['expected_value'] = float(game['expected_value'] or 0)
            if 'edge' in game:
                game['edge'] = float(game['edge'] or 0)
            if 'model_probability' in game:
                game['model_probability'] = float(game['model_probability'] or 0)
        
        return jsonify(serialize_predictions({
            'games': data, 
            'date': str(date.today()),
            'bankroll': current_bankroll['amount']
        }))
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
    """Get NHL betting summary with bankroll info"""
    if not NHL_AVAILABLE:
        return jsonify({'error': 'NHL not available'}), 503
    try:
        days = request.args.get('days', 30, type=int)
        data = nhl_uploader.get_betting_summary(days=days)
        
        # Add current bankroll to summary
        data['current_bankroll'] = current_bankroll['amount']
        data['bankroll_updated_at'] = current_bankroll['updated_at']
        
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