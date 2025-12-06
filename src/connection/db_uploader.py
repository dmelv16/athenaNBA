"""
Database Uploader for NHL Predictions
Uploads JSON predictions to PostgreSQL database
"""

import psycopg2
from psycopg2.extras import Json, execute_batch
from psycopg2 import pool
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class PostgresPredictionUploader:
    """Handle uploading predictions to PostgreSQL database"""
    
    def __init__(self, connection_string: str = None):
        """
        Initialize database connection
        
        Args:
            connection_string: PostgreSQL connection string
                              If None, reads from DATABASE_URL env variable
        """
        self.connection_string = connection_string or os.getenv('DATABASE_URL')
        
        if not self.connection_string:
            raise ValueError(
                "No database connection string provided. "
                "Set DATABASE_URL environment variable or pass connection_string"
            )
        
        # Create connection pool for better performance
        try:
            self.connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=self.connection_string
            )
            logger.info("✅ PostgreSQL connection pool created")
        except Exception as e:
            logger.error(f"❌ Failed to create connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool"""
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool"""
        self.connection_pool.putconn(conn)
    
    def close_pool(self):
        """Close all connections in pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Connection pool closed")
    
    def upload_predictions(self, 
                          predictions_df,
                          prediction_date: Optional[date] = None) -> Dict[str, int]:
        """
        Upload predictions DataFrame to database
        
        Args:
            predictions_df: DataFrame with predictions
            prediction_date: Date of predictions (defaults to today)
            
        Returns:
            Dict with upload statistics
        """
        if len(predictions_df) == 0:
            logger.warning("No predictions to upload")
            return {'inserted': 0, 'updated': 0, 'failed': 0}
        
        if prediction_date is None:
            prediction_date = date.today()
        
        conn = self.get_connection()
        stats = {'inserted': 0, 'updated': 0, 'failed': 0}
        
        try:
            cursor = conn.cursor()
            
            logger.info(f"Uploading {len(predictions_df)} predictions for {prediction_date}")
            
            # Prepare batch insert
            insert_query = """
                INSERT INTO predictions (
                    prediction_date, game_id, game_time,
                    home_team, away_team, venue,
                    predicted_winner, predicted_team,
                    home_win_probability, away_win_probability,
                    model_probability, calibrated_accuracy,
                    action, edge, edge_class,
                    expected_value, expected_roi, sharpe_ratio,
                    bet_size, bet_pct_bankroll, kelly_fraction,
                    decimal_odds, american_odds, implied_probability,
                    prediction_data
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (prediction_date, game_id) 
                DO UPDATE SET
                    game_time = EXCLUDED.game_time,
                    predicted_winner = EXCLUDED.predicted_winner,
                    predicted_team = EXCLUDED.predicted_team,
                    home_win_probability = EXCLUDED.home_win_probability,
                    away_win_probability = EXCLUDED.away_win_probability,
                    model_probability = EXCLUDED.model_probability,
                    calibrated_accuracy = EXCLUDED.calibrated_accuracy,
                    action = EXCLUDED.action,
                    edge = EXCLUDED.edge,
                    edge_class = EXCLUDED.edge_class,
                    expected_value = EXCLUDED.expected_value,
                    expected_roi = EXCLUDED.expected_roi,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    bet_size = EXCLUDED.bet_size,
                    bet_pct_bankroll = EXCLUDED.bet_pct_bankroll,
                    kelly_fraction = EXCLUDED.kelly_fraction,
                    decimal_odds = EXCLUDED.decimal_odds,
                    american_odds = EXCLUDED.american_odds,
                    implied_probability = EXCLUDED.implied_probability,
                    prediction_data = EXCLUDED.prediction_data,
                    updated_at = NOW()
                RETURNING (xmax = 0) AS inserted;
            """
            
            # Prepare data for batch insert
            batch_data = []
            for _, pred in predictions_df.iterrows():
                try:
                    # Parse game_time if it's a string
                    game_time = pred['game_time']
                    if isinstance(game_time, str):
                        game_time = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
                    
                    # Convert prediction to dict for JSON storage
                    pred_dict = pred.to_dict()
                    
                    # Handle NaN values
                    for key, value in pred_dict.items():
                        if isinstance(value, float) and (value != value):  # NaN check
                            pred_dict[key] = None
                    
                    batch_data.append((
                        prediction_date,
                        int(pred['game_id']),
                        game_time,
                        pred['home_team'],
                        pred['away_team'],
                        pred.get('venue', ''),
                        pred['predicted_winner'],
                        pred['predicted_team'],
                        float(pred['home_win_prob']),
                        float(pred['away_win_prob']),
                        float(pred['model_probability']),
                        float(pred['empirical_accuracy']),
                        pred['action'],
                        float(pred['edge']),
                        pred['edge_class'],
                        float(pred['expected_value']),
                        float(pred['expected_roi']),
                        float(pred['sharpe_ratio']),
                        float(pred['bet_size']),
                        float(pred['bet_pct_bankroll']),
                        float(pred['kelly_fraction']),
                        float(pred['decimal_odds']),
                        int(pred['american_odds']),
                        float(pred['implied_probability']),
                        Json(pred_dict)  # Store full prediction as JSONB
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to prepare prediction {pred.get('game_id', 'unknown')}: {e}")
                    stats['failed'] += 1
            
            # Execute batch insert
            if batch_data:
                execute_batch(
                    cursor, 
                    insert_query, 
                    batch_data,
                    page_size=100
                )
                
                # Since we can't fetch results with execute_batch,
                # assume all were successful
                stats['inserted'] = len(batch_data)
                stats['updated'] = 0
                
                conn.commit()
                logger.info(
                    f"✅ Upload complete: {stats['inserted']} inserted, "
                    f"{stats['updated']} updated, {stats['failed']} failed"
                )
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            conn.rollback()
            raise
        
        finally:
            self.return_connection(conn)
        
        return stats
    
    def upload_portfolio_summary(self, 
                                portfolio_data: Dict,
                                summary_date: Optional[date] = None) -> bool:
        """
        Upload portfolio summary for the day
        
        Args:
            portfolio_data: Dict with portfolio metrics
            summary_date: Date of summary (defaults to today)
            
        Returns:
            True if successful
        """
        if summary_date is None:
            summary_date = date.today()
        
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor()
            
            # Convert numpy types to Python native types
            def convert_value(val):
                """Convert numpy types to Python native types"""
                import numpy as np
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    return int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    return float(val)
                elif isinstance(val, np.ndarray):
                    return val.tolist()
                elif pd.isna(val):
                    return 0
                return val
            
            insert_query = """
                INSERT INTO portfolio_summary (
                    summary_date,
                    total_games,
                    games_with_bets,
                    total_stake,
                    total_expected_value,
                    average_edge,
                    portfolio_exposure_pct,
                    portfolio_sharpe,
                    portfolio_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (summary_date)
                DO UPDATE SET
                    total_games = EXCLUDED.total_games,
                    games_with_bets = EXCLUDED.games_with_bets,
                    total_stake = EXCLUDED.total_stake,
                    total_expected_value = EXCLUDED.total_expected_value,
                    average_edge = EXCLUDED.average_edge,
                    portfolio_exposure_pct = EXCLUDED.portfolio_exposure_pct,
                    portfolio_sharpe = EXCLUDED.portfolio_sharpe,
                    portfolio_data = EXCLUDED.portfolio_data;
            """
            
            # Convert all values in portfolio_data
            clean_portfolio_data = {k: convert_value(v) for k, v in portfolio_data.items()}
            
            cursor.execute(insert_query, (
                summary_date,
                convert_value(portfolio_data.get('total_games', 0)),
                convert_value(portfolio_data.get('games_with_bets', 0)),
                convert_value(portfolio_data.get('total_stake', 0)),
                convert_value(portfolio_data.get('total_expected_value', 0)),
                convert_value(portfolio_data.get('average_edge', 0)),
                convert_value(portfolio_data.get('portfolio_exposure_pct', 0)),
                convert_value(portfolio_data.get('portfolio_sharpe', 0)),
                Json(clean_portfolio_data)
            ))
            
            conn.commit()
            logger.info(f"✅ Portfolio summary uploaded for {summary_date}")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload portfolio summary: {e}")
            conn.rollback()
            return False
        
        finally:
            self.return_connection(conn)
    
    def upload_from_json_files(self, json_dir: str = 'predictions_json') -> Dict:
        """
        Upload predictions from JSON files directory
        
        Args:
            json_dir: Directory containing JSON files
            
        Returns:
            Dict with upload statistics
        """
        json_path = Path(json_dir)
        
        if not json_path.exists():
            logger.error(f"JSON directory not found: {json_dir}")
            return {}
        
        stats = {}
        
        # Look for the latest games JSON file
        games_file = json_path / 'games_latest.json'
        portfolio_file = json_path / 'portfolio_latest.json'
        
        if games_file.exists():
            logger.info(f"Loading predictions from {games_file}")
            
            with open(games_file, 'r') as f:
                games_data = json.load(f)
            
            # Convert to DataFrame-like structure for upload
            import pandas as pd
            
            games_list = []
            for game in games_data.get('games', []):
                flat_game = {
                    'game_id': game['game_id'],
                    'game_time': game['matchup']['game_time_utc'],
                    'home_team': game['matchup']['home_team'],
                    'away_team': game['matchup']['away_team'],
                    'venue': game['matchup']['venue'],
                    'matchup': game['matchup']['display_text'],
                    
                    'predicted_winner': game['prediction']['predicted_winner'],
                    'predicted_team': game['prediction']['predicted_team'],
                    'home_win_prob': game['prediction']['home_win_probability'],
                    'away_win_prob': game['prediction']['away_win_probability'],
                    'model_probability': game['prediction']['model_confidence'],
                    'empirical_accuracy': game['prediction']['calibrated_accuracy'],
                    
                    'action': game['recommendation']['action'],
                    'bet_size': game['recommendation']['bet_size'],
                    'bet_pct_bankroll': game['recommendation']['bet_percentage'],
                    'kelly_fraction': game['recommendation']['kelly_fraction'],
                    
                    'edge': game['value_analysis']['edge'],
                    'edge_class': game['value_analysis']['edge_classification'],
                    'expected_value': game['value_analysis']['expected_value'],
                    'expected_roi': game['value_analysis']['expected_roi'],
                    'sharpe_ratio': game['value_analysis']['sharpe_ratio'],
                    
                    'decimal_odds': game['odds']['decimal'],
                    'american_odds': game['odds']['american'],
                    'implied_probability': game['odds']['implied_probability'],
                    
                    'portfolio_correlation': game['risk_metrics']['portfolio_correlation'],
                    'current_portfolio_exposure': game['risk_metrics']['current_exposure'],
                    'accuracy_std_error': game['risk_metrics']['accuracy_std_error'],
                    
                    'reasoning': game['recommendation']['reasoning']
                }
                games_list.append(flat_game)
            
            predictions_df = pd.DataFrame(games_list)
            
            # Upload predictions
            prediction_date = datetime.fromisoformat(games_data['date']).date()
            upload_stats = self.upload_predictions(predictions_df, prediction_date)
            stats['predictions'] = upload_stats
        
        # Upload portfolio summary
        if portfolio_file.exists():
            logger.info(f"Loading portfolio summary from {portfolio_file}")
            
            with open(portfolio_file, 'r') as f:
                portfolio_data = json.load(f)
            
            summary_date = datetime.fromisoformat(portfolio_data['date']).date()
            
            # Extract summary metrics
            summary = {
                'total_games': len(games_list) if games_list else 0,
                'games_with_bets': portfolio_data['portfolio_metrics']['total_positions'],
                'total_stake': portfolio_data['portfolio_metrics']['total_exposure'],
                'total_expected_value': portfolio_data['expected_performance']['portfolio_expected_value'],
                'average_edge': portfolio_data['expected_performance']['average_edge'],
                'portfolio_exposure_pct': portfolio_data['portfolio_metrics']['exposure_percentage'],
                'portfolio_sharpe': portfolio_data['expected_performance']['portfolio_sharpe_ratio'],
                **portfolio_data  # Include full portfolio data
            }
            
            success = self.upload_portfolio_summary(summary, summary_date)
            stats['portfolio_uploaded'] = success
        
        return stats
    
    def get_recent_predictions(self, days: int = 7, today_only: bool = False) -> List[Dict]:
        """
        Retrieve recent predictions from database
        
        Args:
            days: Number of days to look back
            today_only: If True, only return today's predictions
            
        Returns:
            List of prediction dicts
        """
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor()
            
            if today_only:
                query = """
                    SELECT 
                        prediction_date,
                        home_team,
                        away_team,
                        predicted_team,
                        predicted_winner,
                        model_probability,
                        edge,
                        edge_class,
                        bet_size,
                        expected_value,
                        expected_roi,
                        action,
                        decimal_odds,
                        american_odds,
                        implied_probability
                    FROM predictions
                    WHERE prediction_date = CURRENT_DATE
                    ORDER BY bet_size DESC;
                """
                cursor.execute(query)
            else:
                query = """
                    SELECT 
                        prediction_date,
                        home_team,
                        away_team,
                        predicted_team,
                        predicted_winner,
                        model_probability,
                        edge,
                        edge_class,
                        bet_size,
                        expected_value,
                        expected_roi,
                        action,
                        decimal_odds,
                        american_odds,
                        implied_probability
                    FROM predictions
                    WHERE prediction_date >= CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY prediction_date DESC, bet_size DESC;
                """
                cursor.execute(query, (days,))
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            logger.info(f"Retrieved {len(results)} predictions")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve predictions: {e}")
            return []
        
        finally:
            self.return_connection(conn)
    
    def get_betting_summary(self, days: int = 30) -> Dict:
        """
        Get betting summary statistics
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with summary statistics
        """
        conn = self.get_connection()
        
        try:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    COUNT(*) as total_games,
                    COUNT(*) FILTER (WHERE action = 'BET') as games_bet,
                    SUM(bet_size) as total_staked,
                    SUM(expected_value) as total_ev,
                    AVG(edge) FILTER (WHERE action = 'BET') as avg_edge,
                    AVG(model_probability) FILTER (WHERE action = 'BET') as avg_probability
                FROM predictions
                WHERE prediction_date >= CURRENT_DATE - INTERVAL '%s days';
            """
            
            cursor.execute(query, (days,))
            result = cursor.fetchone()
            
            cursor.close()
            
            return {
                'total_games': result[0] or 0,
                'games_bet': result[1] or 0,
                'total_staked': float(result[2]) if result[2] else 0,
                'total_ev': float(result[3]) if result[3] else 0,
                'avg_edge': float(result[4]) if result[4] else 0,
                'avg_probability': float(result[5]) if result[5] else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get betting summary: {e}")
            return {}
        
        finally:
            self.return_connection(conn)


def main():
    """Example usage of the uploader"""
    
    # Initialize uploader
    uploader = PostgresPredictionUploader()
    
    try:
        # Option 1: Upload from JSON files
        logger.info("\n" + "="*80)
        logger.info("UPLOADING PREDICTIONS FROM JSON FILES")
        logger.info("="*80)
        
        stats = uploader.upload_from_json_files('predictions_json')
        
        if stats:
            logger.info("\n📊 UPLOAD SUMMARY:")
            logger.info(f"  Predictions: {stats.get('predictions', {})}")
            logger.info(f"  Portfolio: {stats.get('portfolio_uploaded', False)}")
        
        # Verify upload
        logger.info("\n" + "="*80)
        logger.info("VERIFYING UPLOAD")
        logger.info("="*80)
        
        recent = uploader.get_recent_predictions(days=1)
        logger.info(f"\n✅ Found {len(recent)} predictions from today")
        
        if recent:
            logger.info("\nSample predictions:")
            for pred in recent[:3]:
                logger.info(
                    f"  {pred['away_team']} @ {pred['home_team']}: "
                    f"{pred['predicted_team']} (Edge: {pred['edge']:.2%}, "
                    f"Bet: ${pred['bet_size']:.2f})"
                )
        
        # Get betting summary
        summary = uploader.get_betting_summary(days=7)
        logger.info("\n📈 LAST 7 DAYS SUMMARY:")
        logger.info(f"  Total Games: {summary['total_games']}")
        logger.info(f"  Games Bet: {summary['games_bet']}")
        logger.info(f"  Total Staked: ${summary['total_staked']:.2f}")
        logger.info(f"  Expected Value: ${summary['total_ev']:.2f}")
        logger.info(f"  Average Edge: {summary['avg_edge']:.2%}")
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        uploader.close_pool()


if __name__ == "__main__":
    main()