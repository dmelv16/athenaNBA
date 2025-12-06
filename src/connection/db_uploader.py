"""
Database Uploader for NHL Predictions - Enhanced Version
Includes odds tracking, results verification, and historical accuracy
"""

import psycopg2
from psycopg2.extras import Json, execute_batch
from psycopg2 import pool
import json
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()


class PostgresPredictionUploader:
    """Handle uploading and retrieving NHL predictions from PostgreSQL database"""
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or os.getenv('DATABASE_URL')
        
        if not self.connection_string:
            raise ValueError("No database connection string provided.")
        
        try:
            self.connection_pool = pool.SimpleConnectionPool(
                minconn=1, maxconn=5, dsn=self.connection_string
            )
            logger.info("✅ PostgreSQL connection pool created")
            self._ensure_tables()
        except Exception as e:
            logger.error(f"❌ Failed to create connection pool: {e}")
            raise
    
    def _ensure_tables(self):
        """Ensure required tables exist"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Enhanced predictions table with odds
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nhl_predictions (
                    id SERIAL PRIMARY KEY,
                    prediction_date DATE NOT NULL,
                    game_id BIGINT NOT NULL,
                    game_time TIMESTAMP,
                    home_team VARCHAR(50) NOT NULL,
                    away_team VARCHAR(50) NOT NULL,
                    venue VARCHAR(100),
                    predicted_winner VARCHAR(50),
                    predicted_team VARCHAR(10),
                    home_win_probability DECIMAL(5,4),
                    away_win_probability DECIMAL(5,4),
                    model_probability DECIMAL(5,4),
                    calibrated_accuracy DECIMAL(5,4),
                    action VARCHAR(20),
                    edge DECIMAL(5,4),
                    edge_class VARCHAR(20),
                    expected_value DECIMAL(10,4),
                    expected_roi DECIMAL(5,4),
                    sharpe_ratio DECIMAL(5,4),
                    bet_size DECIMAL(10,2),
                    bet_pct_bankroll DECIMAL(5,4),
                    kelly_fraction DECIMAL(5,4),
                    decimal_odds DECIMAL(6,3),
                    american_odds INTEGER,
                    implied_probability DECIMAL(5,4),
                    prediction_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(prediction_date, game_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_nhl_pred_date ON nhl_predictions(prediction_date);
                CREATE INDEX IF NOT EXISTS idx_nhl_pred_game ON nhl_predictions(game_id);
            """)
            
            # Game results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nhl_game_results (
                    id SERIAL PRIMARY KEY,
                    game_id BIGINT UNIQUE NOT NULL,
                    game_date DATE NOT NULL,
                    home_team VARCHAR(50) NOT NULL,
                    away_team VARCHAR(50) NOT NULL,
                    home_score INTEGER,
                    away_score INTEGER,
                    winner VARCHAR(50),
                    went_to_ot BOOLEAN DEFAULT FALSE,
                    went_to_so BOOLEAN DEFAULT FALSE,
                    final_period VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_nhl_results_date ON nhl_game_results(game_date);
            """)
            
            # Prediction results tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nhl_prediction_results (
                    id SERIAL PRIMARY KEY,
                    prediction_id INTEGER REFERENCES nhl_predictions(id),
                    game_id BIGINT NOT NULL,
                    predicted_winner VARCHAR(50),
                    actual_winner VARCHAR(50),
                    predicted_probability DECIMAL(5,4),
                    was_correct BOOLEAN,
                    bet_placed BOOLEAN DEFAULT FALSE,
                    bet_result VARCHAR(20),
                    profit_loss DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_nhl_pred_results_game ON nhl_prediction_results(game_id);
            """)
            
            conn.commit()
            cursor.close()
            logger.info("✅ Database tables verified/created")
        except Exception as e:
            logger.error(f"Error ensuring tables: {e}")
            conn.rollback()
        finally:
            self.return_connection(conn)
    
    def get_connection(self):
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        self.connection_pool.putconn(conn)
    
    def close_pool(self):
        if self.connection_pool:
            self.connection_pool.closeall()
    
    def upload_predictions(self, predictions_df, prediction_date: Optional[date] = None) -> Dict[str, int]:
        """Upload predictions DataFrame to database"""
        if len(predictions_df) == 0:
            return {'inserted': 0, 'updated': 0, 'failed': 0}
        
        prediction_date = prediction_date or date.today()
        conn = self.get_connection()
        stats = {'inserted': 0, 'updated': 0, 'failed': 0}
        
        try:
            cursor = conn.cursor()
            
            insert_query = """
                INSERT INTO nhl_predictions (
                    prediction_date, game_id, game_time, home_team, away_team, venue,
                    predicted_winner, predicted_team, home_win_probability, away_win_probability,
                    model_probability, calibrated_accuracy, action, edge, edge_class,
                    expected_value, expected_roi, sharpe_ratio, bet_size, bet_pct_bankroll,
                    kelly_fraction, decimal_odds, american_odds, implied_probability, prediction_data
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (prediction_date, game_id) 
                DO UPDATE SET
                    predicted_winner = EXCLUDED.predicted_winner,
                    home_win_probability = EXCLUDED.home_win_probability,
                    away_win_probability = EXCLUDED.away_win_probability,
                    model_probability = EXCLUDED.model_probability,
                    edge = EXCLUDED.edge,
                    bet_size = EXCLUDED.bet_size,
                    prediction_data = EXCLUDED.prediction_data,
                    updated_at = NOW();
            """
            
            batch_data = []
            for _, pred in predictions_df.iterrows():
                try:
                    game_time = pred.get('game_time')
                    if isinstance(game_time, str):
                        game_time = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
                    
                    pred_dict = pred.to_dict()
                    for key, value in pred_dict.items():
                        if isinstance(value, float) and (value != value):
                            pred_dict[key] = None
                    
                    batch_data.append((
                        prediction_date,
                        int(pred['game_id']),
                        game_time,
                        pred['home_team'],
                        pred['away_team'],
                        pred.get('venue', ''),
                        pred['predicted_winner'],
                        pred.get('predicted_team', ''),
                        float(pred.get('home_win_prob', 0.5)),
                        float(pred.get('away_win_prob', 0.5)),
                        float(pred.get('model_probability', 0.5)),
                        float(pred.get('empirical_accuracy', 0.5)),
                        pred.get('action', 'HOLD'),
                        float(pred.get('edge', 0)),
                        pred.get('edge_class', 'none'),
                        float(pred.get('expected_value', 0)),
                        float(pred.get('expected_roi', 0)),
                        float(pred.get('sharpe_ratio', 0)),
                        float(pred.get('bet_size', 0)),
                        float(pred.get('bet_pct_bankroll', 0)),
                        float(pred.get('kelly_fraction', 0)),
                        float(pred.get('decimal_odds', 2.0)),
                        int(pred.get('american_odds', 100)),
                        float(pred.get('implied_probability', 0.5)),
                        Json(pred_dict)
                    ))
                except Exception as e:
                    logger.error(f"Failed to prepare prediction: {e}")
                    stats['failed'] += 1
            
            if batch_data:
                execute_batch(cursor, insert_query, batch_data, page_size=100)
                stats['inserted'] = len(batch_data)
                conn.commit()
            
            cursor.close()
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            conn.rollback()
            raise
        finally:
            self.return_connection(conn)
        
        return stats
    
    def upload_game_result(self, game_id: int, home_team: str, away_team: str,
                          home_score: int, away_score: int, game_date: date,
                          went_to_ot: bool = False, went_to_so: bool = False) -> bool:
        """Upload actual game result for tracking accuracy"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            winner = home_team if home_score > away_score else away_team
            
            cursor.execute("""
                INSERT INTO nhl_game_results (
                    game_id, game_date, home_team, away_team, home_score, away_score,
                    winner, went_to_ot, went_to_so
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id) DO UPDATE SET
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    winner = EXCLUDED.winner,
                    went_to_ot = EXCLUDED.went_to_ot,
                    went_to_so = EXCLUDED.went_to_so;
            """, (game_id, game_date, home_team, away_team, home_score, away_score,
                  winner, went_to_ot, went_to_so))
            
            # Update prediction results
            cursor.execute("""
                INSERT INTO nhl_prediction_results (prediction_id, game_id, predicted_winner, 
                    actual_winner, predicted_probability, was_correct)
                SELECT p.id, p.game_id, p.predicted_winner, %s,
                    p.model_probability, (p.predicted_winner = %s)
                FROM nhl_predictions p
                WHERE p.game_id = %s
                ON CONFLICT DO NOTHING;
            """, (winner, winner, game_id))
            
            conn.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to upload game result: {e}")
            conn.rollback()
            return False
        finally:
            self.return_connection(conn)
    
    def get_recent_predictions(self, days: int = 7, today_only: bool = False) -> List[Dict]:
        """Get recent predictions with results if available"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            if today_only:
                query = """
                    SELECT p.*, r.home_score, r.away_score, r.winner as actual_winner,
                           r.went_to_ot, r.went_to_so,
                           CASE WHEN r.winner IS NOT NULL 
                                THEN (p.predicted_winner = r.winner) 
                                ELSE NULL END as was_correct
                    FROM nhl_predictions p
                    LEFT JOIN nhl_game_results r ON p.game_id = r.game_id
                    WHERE p.prediction_date = CURRENT_DATE
                    ORDER BY p.game_time, p.edge DESC;
                """
                cursor.execute(query)
            else:
                query = """
                    SELECT p.*, r.home_score, r.away_score, r.winner as actual_winner,
                           r.went_to_ot, r.went_to_so,
                           CASE WHEN r.winner IS NOT NULL 
                                THEN (p.predicted_winner = r.winner) 
                                ELSE NULL END as was_correct
                    FROM nhl_predictions p
                    LEFT JOIN nhl_game_results r ON p.game_id = r.game_id
                    WHERE p.prediction_date >= CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY p.prediction_date DESC, p.edge DESC;
                """
                cursor.execute(query, (days,))
            
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve predictions: {e}")
            return []
        finally:
            self.return_connection(conn)
    
    def get_predictions_by_date(self, target_date: date) -> Dict:
        """Get predictions for a specific date"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            query = """
                SELECT p.*, r.home_score, r.away_score, r.winner as actual_winner,
                       r.went_to_ot, r.went_to_so,
                       CASE WHEN r.winner IS NOT NULL 
                            THEN (p.predicted_winner = r.winner) 
                            ELSE NULL END as was_correct
                FROM nhl_predictions p
                LEFT JOIN nhl_game_results r ON p.game_id = r.game_id
                WHERE p.prediction_date = %s
                ORDER BY p.game_time, p.edge DESC;
            """
            cursor.execute(query, (target_date,))
            
            columns = [desc[0] for desc in cursor.description]
            predictions = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            
            return {
                'date': str(target_date),
                'total_games': len(predictions),
                'predictions': predictions
            }
        except Exception as e:
            logger.error(f"Failed to get predictions by date: {e}")
            return {'date': str(target_date), 'total_games': 0, 'predictions': []}
        finally:
            self.return_connection(conn)
    
    def get_prediction_results(self, days: int = 30) -> Dict:
        """Get prediction results with win/loss tracking"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    p.prediction_date,
                    p.game_id,
                    p.home_team,
                    p.away_team,
                    p.predicted_winner,
                    p.model_probability,
                    p.edge,
                    p.bet_size,
                    p.action,
                    r.home_score,
                    r.away_score,
                    r.winner as actual_winner,
                    r.went_to_ot,
                    (p.predicted_winner = r.winner) as was_correct,
                    CASE 
                        WHEN p.action = 'BET' AND p.predicted_winner = r.winner 
                        THEN p.bet_size * (p.decimal_odds - 1)
                        WHEN p.action = 'BET' AND p.predicted_winner != r.winner 
                        THEN -p.bet_size
                        ELSE 0 
                    END as profit_loss
                FROM nhl_predictions p
                JOIN nhl_game_results r ON p.game_id = r.game_id
                WHERE p.prediction_date >= CURRENT_DATE - INTERVAL '%s days'
                ORDER BY p.prediction_date DESC;
            """
            cursor.execute(query, (days,))
            
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            # Calculate summary stats
            total_bets = sum(1 for r in results if r['action'] == 'BET')
            wins = sum(1 for r in results if r['action'] == 'BET' and r['was_correct'])
            total_profit = sum(r['profit_loss'] or 0 for r in results)
            
            cursor.close()
            
            return {
                'days': days,
                'total_predictions': len(results),
                'total_bets': total_bets,
                'wins': wins,
                'losses': total_bets - wins,
                'win_rate': wins / total_bets if total_bets > 0 else 0,
                'total_profit': total_profit,
                'results': results
            }
        except Exception as e:
            logger.error(f"Failed to get results: {e}")
            return {}
        finally:
            self.return_connection(conn)
    
    def get_accuracy_stats(self, days: int = 30) -> Dict:
        """Get accuracy statistics"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            query = """
                WITH prediction_results AS (
                    SELECT 
                        p.prediction_date,
                        p.model_probability,
                        p.edge,
                        p.action,
                        (p.predicted_winner = r.winner) as was_correct
                    FROM nhl_predictions p
                    JOIN nhl_game_results r ON p.game_id = r.game_id
                    WHERE p.prediction_date >= CURRENT_DATE - INTERVAL '%s days'
                )
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct_predictions,
                    AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) as overall_accuracy,
                    AVG(CASE WHEN action = 'BET' AND was_correct THEN 1.0 
                             WHEN action = 'BET' THEN 0.0 
                             ELSE NULL END) as bet_accuracy,
                    AVG(model_probability) as avg_confidence,
                    AVG(edge) FILTER (WHERE action = 'BET') as avg_edge_on_bets
                FROM prediction_results;
            """
            cursor.execute(query, (days,))
            
            row = cursor.fetchone()
            columns = [desc[0] for desc in cursor.description]
            stats = dict(zip(columns, row)) if row else {}
            
            # Get accuracy by confidence bucket
            cursor.execute("""
                WITH prediction_results AS (
                    SELECT 
                        CASE 
                            WHEN p.model_probability >= 0.7 THEN 'high'
                            WHEN p.model_probability >= 0.6 THEN 'medium'
                            ELSE 'low'
                        END as confidence_bucket,
                        (p.predicted_winner = r.winner) as was_correct
                    FROM nhl_predictions p
                    JOIN nhl_game_results r ON p.game_id = r.game_id
                    WHERE p.prediction_date >= CURRENT_DATE - INTERVAL '%s days'
                )
                SELECT 
                    confidence_bucket,
                    COUNT(*) as count,
                    AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) as accuracy
                FROM prediction_results
                GROUP BY confidence_bucket
                ORDER BY confidence_bucket;
            """, (days,))
            
            by_confidence = []
            for row in cursor.fetchall():
                by_confidence.append({
                    'bucket': row[0],
                    'count': row[1],
                    'accuracy': float(row[2]) if row[2] else 0
                })
            
            cursor.close()
            
            stats['by_confidence'] = by_confidence
            stats['days'] = days
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get accuracy stats: {e}")
            return {}
        finally:
            self.return_connection(conn)
    
    def get_best_bets(self, min_edge: float = 0.05, min_probability: float = 0.55) -> List[Dict]:
        """Get best betting opportunities for today"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM nhl_predictions
                WHERE prediction_date = CURRENT_DATE
                  AND edge >= %s
                  AND model_probability >= %s
                  AND action = 'BET'
                ORDER BY edge DESC, model_probability DESC
                LIMIT 10;
            """
            cursor.execute(query, (min_edge, min_probability))
            
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            return results
        except Exception as e:
            logger.error(f"Failed to get best bets: {e}")
            return []
        finally:
            self.return_connection(conn)
    
    def get_betting_summary(self, days: int = 30) -> Dict:
        """Get betting summary statistics"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    COUNT(*) as total_games,
                    COUNT(*) FILTER (WHERE action = 'BET') as games_bet,
                    COALESCE(SUM(bet_size), 0) as total_staked,
                    COALESCE(SUM(expected_value), 0) as total_ev,
                    AVG(edge) FILTER (WHERE action = 'BET') as avg_edge,
                    AVG(model_probability) FILTER (WHERE action = 'BET') as avg_probability
                FROM nhl_predictions
                WHERE prediction_date >= CURRENT_DATE - INTERVAL '%s days';
            """
            cursor.execute(query, (days,))
            
            row = cursor.fetchone()
            cursor.close()
            
            return {
                'days': days,
                'total_games': row[0] or 0,
                'games_bet': row[1] or 0,
                'total_staked': float(row[2]) if row[2] else 0,
                'total_ev': float(row[3]) if row[3] else 0,
                'avg_edge': float(row[4]) if row[4] else 0,
                'avg_probability': float(row[5]) if row[5] else 0
            }
        except Exception as e:
            logger.error(f"Failed to get betting summary: {e}")
            return {}
        finally:
            self.return_connection(conn)