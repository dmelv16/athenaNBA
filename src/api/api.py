"""
Flask REST API for NHL Predictions
Serves data to React frontend
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from src.connection.db_uploader import PostgresPredictionUploader
from datetime import date, datetime, timedelta
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for React dev server

uploader = PostgresPredictionUploader()

# ============================================
# Predictions Endpoints
# ============================================

@app.route('/api/predictions/today', methods=['GET'])
def get_todays_predictions():
    """Get today's predictions only"""
    try:
        # Get only today's predictions from database
        predictions = uploader.get_recent_predictions(today_only=True)
        
        if predictions and len(predictions) > 0:
            return jsonify({
                'date': str(date.today()),
                'total_games': len(predictions),
                'games': predictions
            }), 200
        
        # Fallback to JSON file if no database results
        json_path = Path(r'D:\NHLapi\predictions_json\games_latest.json')
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Filter JSON to only include today's date
            today_str = str(date.today())
            if data.get('date') == today_str:
                return jsonify(data), 200
            else:
                return jsonify({
                    'date': today_str,
                    'total_games': 0,
                    'games': [],
                    'message': 'No predictions for today yet. Run predictions first.'
                }), 200
        
        return jsonify({
            'date': str(date.today()),
            'total_games': 0,
            'games': [],
            'message': 'No predictions available'
        }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/history', methods=['GET'])
def get_predictions_history():
    """Get historical predictions"""
    try:
        days = request.args.get('days', default=7, type=int)
        predictions = uploader.get_recent_predictions(days=days, today_only=False)
        
        # Group by date
        from collections import defaultdict
        grouped = defaultdict(list)
        
        for pred in predictions:
            pred_date = str(pred['prediction_date'].date() if hasattr(pred['prediction_date'], 'date') else pred['prediction_date'])
            grouped[pred_date].append(pred)
        
        return jsonify({
            'days_requested': days,
            'total_predictions': len(predictions),
            'by_date': dict(grouped)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/predictions/summary', methods=['GET'])
def get_summary():
    """Get summary statistics"""
    try:
        json_path = Path('predictions_json/summary_latest.json')
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            return jsonify(data), 200
        else:
            summary = uploader.get_betting_summary(days=7)
            return jsonify(summary), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/betting', methods=['GET'])
def get_betting_recommendations():
    """Get betting recommendations sorted by value"""
    try:
        json_path = Path('predictions_json/betting_latest.json')
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            return jsonify(data), 200
        else:
            return jsonify({'error': 'No betting data available'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/analytics', methods=['GET'])
def get_analytics():
    """Get detailed analytics"""
    try:
        json_path = Path('predictions_json/analytics_latest.json')
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            return jsonify(data), 200
        else:
            return jsonify({'error': 'No analytics data available'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/portfolio', methods=['GET'])
def get_portfolio():
    """Get portfolio metrics"""
    try:
        json_path = Path('predictions_json/portfolio_latest.json')
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            return jsonify(data), 200
        else:
            return jsonify({'error': 'No portfolio data available'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# Historical Data Endpoints
# ============================================

@app.route('/api/history/recent', methods=['GET'])
def get_recent_history():
    """Get recent prediction history"""
    try:
        days = request.args.get('days', default=7, type=int)
        predictions = uploader.get_recent_predictions(days=days)
        
        return jsonify({
            'days': days,
            'total_predictions': len(predictions),
            'predictions': predictions
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/summary', methods=['GET'])
def get_history_summary():
    """Get historical betting summary"""
    try:
        days = request.args.get('days', default=30, type=int)
        summary = uploader.get_betting_summary(days=days)
        
        return jsonify(summary), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# Utility Endpoints
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'date': str(date.today())
    }), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        # Check if predictions exist for today
        json_path = Path('predictions_json/games_latest.json')
        has_todays_predictions = json_path.exists()
        
        # Check database connection
        try:
            recent = uploader.get_recent_predictions(days=1)
            db_connected = True
            db_prediction_count = len(recent)
        except:
            db_connected = False
            db_prediction_count = 0
        
        return jsonify({
            'status': 'operational',
            'date': str(date.today()),
            'predictions_available': has_todays_predictions,
            'database_connected': db_connected,
            'database_prediction_count': db_prediction_count
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

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
    print("🚀 Starting NHL Predictions API...")
    print("📊 Available endpoints:")
    print("   GET /api/predictions/today")
    print("   GET /api/predictions/summary")
    print("   GET /api/predictions/betting")
    print("   GET /api/predictions/analytics")
    print("   GET /api/predictions/portfolio")
    print("   GET /api/history/recent?days=7")
    print("   GET /api/history/summary?days=30")
    print("   GET /api/health")
    print("   GET /api/status")
    print("\n🌐 Server running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)