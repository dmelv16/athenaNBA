"""
Database uploader for NBA predictions
Handles storing and retrieving predictions from PostgreSQL
"""

import psycopg2
from psycopg2.extras import execute_values, Json
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from etl.config.settings import DatabaseConfig


class NBAPredictionUploader:
    """Upload and retrieve NBA predictions from database"""
    
    def __init__(self):
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=DatabaseConfig.HOST,
                port=DatabaseConfig.PORT,
                dbname=DatabaseConfig.NAME,
                user=DatabaseConfig.USER,
                password=DatabaseConfig.PASSWORD,
                sslmode=DatabaseConfig.SSLMODE
            )
            print(f"✓ Connected to database: {DatabaseConfig.HOST}:{DatabaseConfig.PORT}/{DatabaseConfig.NAME}")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def init_schema(self):
        """Initialize prediction tables"""
        schema_sql = """
        -- Player prop predictions
        CREATE TABLE IF NOT EXISTS nba_player_predictions (
            id SERIAL PRIMARY KEY,
            prediction_date DATE NOT NULL,
            game_id VARCHAR(20) NOT NULL,
            game_date DATE NOT NULL,
            player_id INTEGER NOT NULL,
            player_name VARCHAR(100) NOT NULL,
            team_abbrev VARCHAR(10),
            opponent_abbrev VARCHAR(10),
            is_home BOOLEAN,
            prop_type VARCHAR(20) NOT NULL,
            predicted_value DECIMAL(10,2) NOT NULL,
            confidence DECIMAL(5,4),
            lower_bound DECIMAL(10,2),
            upper_bound DECIMAL(10,2),
            line DECIMAL(10,2),
            edge DECIMAL(10,2),
            recommended_bet VARCHAR(10),
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(prediction_date, game_id, player_id, prop_type)
        );

        -- Team prop predictions
        CREATE TABLE IF NOT EXISTS nba_team_predictions (
            id SERIAL PRIMARY KEY,
            prediction_date DATE NOT NULL,
            game_id VARCHAR(20) NOT NULL,
            game_date DATE NOT NULL,
            home_team_id INTEGER NOT NULL,
            home_team_abbrev VARCHAR(10),
            away_team_id INTEGER NOT NULL,
            away_team_abbrev VARCHAR(10),
            prop_type VARCHAR(20) NOT NULL,
            predicted_value DECIMAL(10,2) NOT NULL,
            confidence DECIMAL(5,4),
            lower_bound DECIMAL(10,2),
            upper_bound DECIMAL(10,2),
            line DECIMAL(10,2),
            edge DECIMAL(10,2),
            recommended_bet VARCHAR(10),
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(prediction_date, game_id, prop_type)
        );

        -- Parlay recommendations
        CREATE TABLE IF NOT EXISTS nba_parlay_predictions (
            id SERIAL PRIMARY KEY,
            prediction_date DATE NOT NULL,
            parlay_type VARCHAR(20),
            num_legs INTEGER,
            legs JSONB NOT NULL,
            combined_odds DECIMAL(10,2),
            american_odds INTEGER,
            avg_confidence DECIMAL(5,4),
            min_confidence DECIMAL(5,4),
            avg_edge DECIMAL(10,2),
            expected_value DECIMAL(5,4),
            score DECIMAL(10,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_nba_player_pred_date ON nba_player_predictions(prediction_date);
        CREATE INDEX IF NOT EXISTS idx_nba_player_pred_game ON nba_player_predictions(game_id);
        CREATE INDEX IF NOT EXISTS idx_nba_player_pred_player ON nba_player_predictions(player_id);
        CREATE INDEX IF NOT EXISTS idx_nba_team_pred_date ON nba_team_predictions(prediction_date);
        CREATE INDEX IF NOT EXISTS idx_nba_team_pred_game ON nba_team_predictions(game_id);
        CREATE INDEX IF NOT EXISTS idx_nba_parlay_date ON nba_parlay_predictions(prediction_date);
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(schema_sql)
                self.conn.commit()
            print("✓ Schema initialized")
        except Exception as e:
            print(f"✗ Schema initialization error: {e}")
            self.conn.rollback()
    
    # ==========================================
    # Upload Methods
    # ==========================================
    
    def upload_player_predictions(
        self,
        predictions: List[Dict],
        prediction_date: date = None,
        model_version: str = "v1.0"
    ) -> int:
        """
        Upload player prop predictions
        
        Args:
            predictions: List of prediction dictionaries
            prediction_date: Date predictions are for
            model_version: Model version string
            
        Returns:
            Number of predictions uploaded
        """
        if not predictions:
            return 0
        
        prediction_date = prediction_date or date.today()
        
        query = """
            INSERT INTO nba_player_predictions (
                prediction_date, game_id, game_date,
                player_id, player_name, team_abbrev, opponent_abbrev, is_home,
                prop_type, predicted_value, confidence, lower_bound, upper_bound,
                line, edge, recommended_bet, model_version
            ) VALUES %s
            ON CONFLICT (prediction_date, game_id, player_id, prop_type) 
            DO UPDATE SET
                predicted_value = EXCLUDED.predicted_value,
                confidence = EXCLUDED.confidence,
                lower_bound = EXCLUDED.lower_bound,
                upper_bound = EXCLUDED.upper_bound,
                line = EXCLUDED.line,
                edge = EXCLUDED.edge,
                recommended_bet = EXCLUDED.recommended_bet,
                model_version = EXCLUDED.model_version,
                created_at = CURRENT_TIMESTAMP
        """
        
        values = []
        for p in predictions:
            values.append((
                prediction_date,
                p.get('game_id'),
                p.get('game_date'),
                p.get('player_id'),
                p.get('player_name'),
                p.get('team_abbrev'),
                p.get('opponent_abbrev'),
                p.get('is_home'),
                p.get('prop_type'),
                p.get('predicted_value'),
                p.get('confidence'),
                p.get('lower_bound'),
                p.get('upper_bound'),
                p.get('line'),
                p.get('edge'),
                p.get('recommended_bet'),
                model_version
            ))
        
        with self.conn.cursor() as cur:
            execute_values(cur, query, values)
            self.conn.commit()
        
        print(f"✓ Uploaded {len(values)} player predictions")
        return len(values)
    
    def upload_team_predictions(
        self,
        predictions: List[Dict],
        prediction_date: date = None,
        model_version: str = "v1.0"
    ) -> int:
        """Upload team prop predictions (spread, total)"""
        if not predictions:
            return 0
        
        prediction_date = prediction_date or date.today()
        
        query = """
            INSERT INTO nba_team_predictions (
                prediction_date, game_id, game_date,
                home_team_id, home_team_abbrev, away_team_id, away_team_abbrev,
                prop_type, predicted_value, confidence, lower_bound, upper_bound,
                line, edge, recommended_bet, model_version
            ) VALUES %s
            ON CONFLICT (prediction_date, game_id, prop_type)
            DO UPDATE SET
                predicted_value = EXCLUDED.predicted_value,
                confidence = EXCLUDED.confidence,
                lower_bound = EXCLUDED.lower_bound,
                upper_bound = EXCLUDED.upper_bound,
                line = EXCLUDED.line,
                edge = EXCLUDED.edge,
                recommended_bet = EXCLUDED.recommended_bet,
                model_version = EXCLUDED.model_version,
                created_at = CURRENT_TIMESTAMP
        """
        
        values = []
        for p in predictions:
            values.append((
                prediction_date,
                p.get('game_id'),
                p.get('game_date'),
                p.get('home_team_id'),
                p.get('home_team_abbrev'),
                p.get('away_team_id'),
                p.get('away_team_abbrev'),
                p.get('prop_type'),
                p.get('predicted_value'),
                p.get('confidence'),
                p.get('lower_bound'),
                p.get('upper_bound'),
                p.get('line'),
                p.get('edge'),
                p.get('recommended_bet'),
                model_version
            ))
        
        with self.conn.cursor() as cur:
            execute_values(cur, query, values)
            self.conn.commit()
        
        print(f"✓ Uploaded {len(values)} team predictions")
        return len(values)
    
    def upload_parlays(
        self,
        parlays: List[Dict],
        prediction_date: date = None
    ) -> int:
        """Upload parlay recommendations"""
        if not parlays:
            return 0
        
        prediction_date = prediction_date or date.today()
        
        query = """
            INSERT INTO nba_parlay_predictions (
                prediction_date, parlay_type, num_legs, legs,
                combined_odds, american_odds, avg_confidence, min_confidence,
                avg_edge, expected_value, score
            ) VALUES %s
        """
        
        values = []
        for p in parlays:
            values.append((
                prediction_date,
                p.get('parlay_type'),
                p.get('num_legs'),
                Json(p.get('legs', [])),
                p.get('combined_odds'),
                p.get('american_odds'),
                p.get('avg_confidence'),
                p.get('min_confidence'),
                p.get('avg_edge'),
                p.get('expected_value'),
                p.get('score')
            ))
        
        with self.conn.cursor() as cur:
            execute_values(cur, query, values)
            self.conn.commit()
        
        print(f"✓ Uploaded {len(values)} parlays")
        return len(values)
    
    # ==========================================
    # Retrieval Methods
    # ==========================================
    
    def get_todays_predictions(self) -> Dict:
        """Get all predictions for today"""
        today = date.today()
        
        # Get player predictions
        player_query = """
            SELECT * FROM nba_player_predictions
            WHERE prediction_date = %s
            ORDER BY confidence DESC
        """
        
        # Get team predictions
        team_query = """
            SELECT * FROM nba_team_predictions
            WHERE prediction_date = %s
            ORDER BY game_id
        """
        
        # Get parlays
        parlay_query = """
            SELECT * FROM nba_parlay_predictions
            WHERE prediction_date = %s
            ORDER BY score DESC
        """
        
        with self.conn.cursor() as cur:
            # Player predictions
            cur.execute(player_query, (today,))
            columns = [desc[0] for desc in cur.description]
            player_preds = [dict(zip(columns, row)) for row in cur.fetchall()]
            
            # Team predictions
            cur.execute(team_query, (today,))
            columns = [desc[0] for desc in cur.description]
            team_preds = [dict(zip(columns, row)) for row in cur.fetchall()]
            
            # Parlays
            cur.execute(parlay_query, (today,))
            columns = [desc[0] for desc in cur.description]
            parlays = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        # Convert dates to strings for JSON serialization
        for p in player_preds + team_preds + parlays:
            for key, value in p.items():
                if isinstance(value, (date, datetime)):
                    p[key] = str(value)
                elif hasattr(value, 'item'):  # numpy/decimal types
                    p[key] = float(value)
        
        # Group player predictions by game
        games = {}
        for pred in player_preds:
            game_id = pred['game_id']
            if game_id not in games:
                games[game_id] = {
                    'game_id': game_id,
                    'game_date': pred['game_date'],
                    'team_predictions': [],
                    'player_predictions': []
                }
            games[game_id]['player_predictions'].append(pred)
        
        # Add team predictions to games
        for pred in team_preds:
            game_id = pred['game_id']
            if game_id not in games:
                games[game_id] = {
                    'game_id': game_id,
                    'game_date': pred['game_date'],
                    'home_team': pred['home_team_abbrev'],
                    'away_team': pred['away_team_abbrev'],
                    'team_predictions': [],
                    'player_predictions': []
                }
            games[game_id]['team_predictions'].append(pred)
            games[game_id]['home_team'] = pred['home_team_abbrev']
            games[game_id]['away_team'] = pred['away_team_abbrev']
        
        return {
            'date': str(today),
            'total_games': len(games),
            'total_player_predictions': len(player_preds),
            'total_team_predictions': len(team_preds),
            'games': list(games.values()),
            'parlays': parlays
        }
    
    def get_predictions_by_date(self, target_date: date) -> Dict:
        """Get predictions for a specific date"""
        player_query = """
            SELECT * FROM nba_player_predictions
            WHERE prediction_date = %s
            ORDER BY confidence DESC
        """
        
        team_query = """
            SELECT * FROM nba_team_predictions
            WHERE prediction_date = %s
        """
        
        with self.conn.cursor() as cur:
            cur.execute(player_query, (target_date,))
            columns = [desc[0] for desc in cur.description]
            player_preds = [dict(zip(columns, row)) for row in cur.fetchall()]
            
            cur.execute(team_query, (target_date,))
            columns = [desc[0] for desc in cur.description]
            team_preds = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        # Convert types
        for p in player_preds + team_preds:
            for key, value in p.items():
                if isinstance(value, (date, datetime)):
                    p[key] = str(value)
                elif hasattr(value, 'item'):
                    p[key] = float(value)
        
        return {
            'date': str(target_date),
            'player_predictions': player_preds,
            'team_predictions': team_preds
        }
    
    def get_recent_predictions(
        self,
        days: int = 7,
        today_only: bool = False
    ) -> List[Dict]:
        """Get recent predictions"""
        if today_only:
            query = """
                SELECT * FROM nba_player_predictions
                WHERE prediction_date = %s
                ORDER BY confidence DESC
            """
            params = (date.today(),)
        else:
            query = """
                SELECT * FROM nba_player_predictions
                WHERE prediction_date >= %s
                ORDER BY prediction_date DESC, confidence DESC
            """
            params = (date.today() - timedelta(days=days),)
        
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        for r in results:
            for key, value in r.items():
                if isinstance(value, (date, datetime)):
                    r[key] = str(value)
                elif hasattr(value, 'item'):
                    r[key] = float(value)
        
        return results
    
    def get_best_bets(self, min_confidence: float = 0.65, min_edge: float = 2.0) -> List[Dict]:
        """Get highest confidence predictions for today"""
        query = """
            SELECT * FROM nba_player_predictions
            WHERE prediction_date = %s
              AND confidence >= %s
              AND (edge IS NULL OR ABS(edge) >= %s)
            ORDER BY confidence DESC, ABS(edge) DESC
            LIMIT 20
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (date.today(), min_confidence, min_edge))
            columns = [desc[0] for desc in cur.description]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        for r in results:
            for key, value in r.items():
                if isinstance(value, (date, datetime)):
                    r[key] = str(value)
                elif hasattr(value, 'item'):
                    r[key] = float(value)
        
        return results
    
    def get_parlays(self, prediction_date: date = None) -> List[Dict]:
        """Get parlay recommendations"""
        prediction_date = prediction_date or date.today()
        
        query = """
            SELECT * FROM nba_parlay_predictions
            WHERE prediction_date = %s
            ORDER BY score DESC
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (prediction_date,))
            columns = [desc[0] for desc in cur.description]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        for r in results:
            for key, value in r.items():
                if isinstance(value, (date, datetime)):
                    r[key] = str(value)
                elif hasattr(value, 'item'):
                    r[key] = float(value)
        
        return results
    
    def get_prediction_stats(self, days: int = 30) -> Dict:
        """Get prediction statistics"""
        query = """
            SELECT 
                prop_type,
                COUNT(*) as total_predictions,
                AVG(confidence) as avg_confidence,
                AVG(ABS(edge)) as avg_edge
            FROM nba_player_predictions
            WHERE prediction_date >= %s
            GROUP BY prop_type
            ORDER BY total_predictions DESC
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (date.today() - timedelta(days=days),))
            columns = [desc[0] for desc in cur.description]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        return {
            'days': days,
            'by_prop_type': results
        }