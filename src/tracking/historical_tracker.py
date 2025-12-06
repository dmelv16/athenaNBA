"""
Historical NHL Game Tracker
Matches predictions with actual results, calculates P&L, and tracks bankroll over time
"""

import pyodbc
import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalGameTracker:
    """Track historical game results against predictions and update bankroll"""
    
    def __init__(self, 
                 mssql_connection_string: str,
                 postgres_connection_string: str = None,
                 starting_bankroll: float = 1000.0):
        """
        Initialize the tracker
        
        Args:
            mssql_connection_string: Connection string for MSSQL (game results)
            postgres_connection_string: Connection string for PostgreSQL (predictions)
            starting_bankroll: Starting bankroll amount
        """
        self.mssql_conn_str = mssql_connection_string
        self.postgres_conn_str = postgres_connection_string or os.getenv('DATABASE_URL')
        self.starting_bankroll = starting_bankroll
        
        self.mssql_conn = None
        self.pg_conn = None
        
    def connect(self):
        """Establish database connections"""
        try:
            self.mssql_conn = pyodbc.connect(self.mssql_conn_str)
            logger.info("✅ Connected to MSSQL (game results)")
        except Exception as e:
            logger.error(f"❌ MSSQL connection failed: {e}")
            raise
        
        try:
            self.pg_conn = psycopg2.connect(self.postgres_conn_str)
            logger.info("✅ Connected to PostgreSQL (predictions)")
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connections"""
        if self.mssql_conn:
            self.mssql_conn.close()
        if self.pg_conn:
            self.pg_conn.close()
        logger.info("Database connections closed")
    
    def ensure_tracking_tables(self):
        """Create tracking tables in PostgreSQL if they don't exist"""
        cursor = self.pg_conn.cursor()
        
        # Table for individual bet results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bet_results (
                id SERIAL PRIMARY KEY,
                game_id BIGINT NOT NULL,
                prediction_date DATE NOT NULL,
                game_date DATE NOT NULL,
                home_team VARCHAR(10) NOT NULL,
                away_team VARCHAR(10) NOT NULL,
                predicted_team VARCHAR(10) NOT NULL,
                predicted_winner VARCHAR(10) NOT NULL,
                bet_size DECIMAL(10, 2) NOT NULL,
                decimal_odds DECIMAL(6, 3) NOT NULL,
                american_odds INTEGER NOT NULL,
                model_probability DECIMAL(5, 4) NOT NULL,
                edge DECIMAL(6, 4) NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                actual_winner VARCHAR(10),
                bet_result VARCHAR(10),
                pnl DECIMAL(10, 2),
                bankroll_before DECIMAL(12, 2),
                bankroll_after DECIMAL(12, 2),
                processed_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(game_id, prediction_date)
            );
            
            CREATE INDEX IF NOT EXISTS idx_bet_results_game_date 
            ON bet_results(game_date);
            
            CREATE INDEX IF NOT EXISTS idx_bet_results_processed 
            ON bet_results(processed_at);
        """)
        
        # Table for daily bankroll snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bankroll_history (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL UNIQUE,
                starting_bankroll DECIMAL(12, 2) NOT NULL,
                ending_bankroll DECIMAL(12, 2) NOT NULL,
                daily_pnl DECIMAL(10, 2) NOT NULL,
                bets_placed INTEGER NOT NULL,
                bets_won INTEGER NOT NULL,
                bets_lost INTEGER NOT NULL,
                bets_pushed INTEGER NOT NULL,
                win_rate DECIMAL(5, 4),
                total_staked DECIMAL(10, 2),
                roi_pct DECIMAL(6, 2),
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Table for cumulative statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracking_stats (
                id SERIAL PRIMARY KEY,
                stat_name VARCHAR(50) UNIQUE NOT NULL,
                stat_value DECIMAL(12, 4) NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW()
            );
            
            INSERT INTO tracking_stats (stat_name, stat_value)
            VALUES ('current_bankroll', %s)
            ON CONFLICT (stat_name) DO NOTHING;
        """, (self.starting_bankroll,))
        
        self.pg_conn.commit()
        logger.info("✅ Tracking tables ready")
    
    def get_current_bankroll(self) -> float:
        """Get current bankroll from database"""
        cursor = self.pg_conn.cursor()
        cursor.execute("""
            SELECT stat_value FROM tracking_stats 
            WHERE stat_name = 'current_bankroll'
        """)
        result = cursor.fetchone()
        return float(result[0]) if result else self.starting_bankroll
    
    def update_bankroll(self, new_bankroll: float):
        """Update current bankroll in database"""
        cursor = self.pg_conn.cursor()
        cursor.execute("""
            UPDATE tracking_stats 
            SET stat_value = %s, updated_at = NOW()
            WHERE stat_name = 'current_bankroll'
        """, (new_bankroll,))
        self.pg_conn.commit()
    
    def fetch_completed_games(self, 
                              start_date: date = None, 
                              end_date: date = None) -> pd.DataFrame:
        """
        Fetch completed games from MSSQL schedule table
        
        Args:
            start_date: Start date for game lookup
            end_date: End date for game lookup (defaults to yesterday)
        """
        if end_date is None:
            end_date = date.today() - timedelta(days=1)
        
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        query = """
            SELECT 
                game_id,
                season,
                gameType,
                CAST(CAST(startTimeUTC AS datetimeoffset) AT TIME ZONE 'Mountain Standard Time' AS DATE) as gameDate,
                venue,
                gameState,
                awayTeam_id,
                awayTeam_abbrev,
                awayTeam_score,
                homeTeam_id,
                homeTeam_abbrev,
                homeTeam_score,
                periodDescriptor_json,
                gameOutcome_json
            FROM [nhlDB].[schedule].[schedule]
            WHERE gameState IN ('OFF', 'FINAL')
                AND gameType = 2
                AND CAST(CAST(startTimeUTC AS datetimeoffset) AT TIME ZONE 'Mountain Standard Time' AS DATE) BETWEEN ? AND ?
            ORDER BY gameDate ASC
        """
        
        df = pd.read_sql(query, self.mssql_conn, params=[start_date, end_date])
        logger.info(f"Fetched {len(df)} completed games from {start_date} to {end_date}")
        return df
    
    def fetch_predictions_for_games(self, game_ids: List[int]) -> pd.DataFrame:
        """Fetch predictions from PostgreSQL for specific game IDs"""
        if not game_ids:
            return pd.DataFrame()
        
        cursor = self.pg_conn.cursor()
        
        # Convert to tuple for SQL IN clause
        game_ids_tuple = tuple(game_ids)
        
        query = """
            SELECT 
                game_id,
                prediction_date,
                home_team,
                away_team,
                predicted_winner,
                predicted_team,
                home_win_probability,
                away_win_probability,
                model_probability,
                action,
                edge,
                bet_size,
                decimal_odds,
                american_odds,
                implied_probability,
                expected_value
            FROM predictions
            WHERE game_id IN %s
                AND action = 'BET'
        """
        
        cursor.execute(query, (game_ids_tuple,))
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        
        df = pd.DataFrame(results, columns=columns)
        logger.info(f"Found {len(df)} predictions with bets for {len(game_ids)} games")
        return df
    
    def fetch_unprocessed_predictions(self) -> pd.DataFrame:
        """Fetch predictions that haven't been processed yet"""
        cursor = self.pg_conn.cursor()
        
        query = """
            SELECT 
                p.game_id,
                p.prediction_date,
                p.home_team,
                p.away_team,
                p.predicted_winner,
                p.predicted_team,
                p.home_win_probability,
                p.away_win_probability,
                p.model_probability,
                p.action,
                p.edge,
                p.bet_size,
                p.decimal_odds,
                p.american_odds,
                p.implied_probability,
                p.expected_value
            FROM predictions p
            LEFT JOIN bet_results br ON p.game_id = br.game_id 
                AND p.prediction_date = br.prediction_date
            WHERE p.action = 'BET'
                AND br.id IS NULL
                AND p.prediction_date < CURRENT_DATE
            ORDER BY p.prediction_date ASC
        """
        
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        
        df = pd.DataFrame(results, columns=columns)
        logger.info(f"Found {len(df)} unprocessed predictions")
        return df
    
    def determine_winner(self, home_score: int, away_score: int) -> str:
        """Determine the winner of a game"""
        if home_score > away_score:
            return 'HOME'
        elif away_score > home_score:
            return 'AWAY'
        else:
            return 'TIE'  # Shouldn't happen in NHL regulation
    
    def calculate_bet_result(self, 
                            predicted_winner: str,
                            actual_winner: str,
                            bet_size: float,
                            decimal_odds: float) -> Tuple[str, float]:
        """
        Calculate the result and P&L of a bet
        
        Returns:
            Tuple of (result_string, pnl_amount)
        """
        if predicted_winner == actual_winner:
            # Win - payout is (decimal_odds - 1) * bet_size
            pnl = bet_size * (decimal_odds - 1)
            return 'WIN', pnl
        else:
            # Loss - lose the bet size
            return 'LOSS', -bet_size
    
    def process_game_results(self, 
                            games_df: pd.DataFrame, 
                            predictions_df: pd.DataFrame) -> List[Dict]:
        """
        Match games with predictions and calculate results
        
        Returns:
            List of result dictionaries
        """
        results = []
        current_bankroll = self.get_current_bankroll()
        
        # Merge games with predictions
        merged = predictions_df.merge(
            games_df[['game_id', 'gameDate', 'homeTeam_score', 'awayTeam_score', 
                     'homeTeam_abbrev', 'awayTeam_abbrev']],
            on='game_id',
            how='inner'
        )
        
        logger.info(f"Matched {len(merged)} predictions with game results")
        
        for _, row in merged.iterrows():
            # Determine actual winner
            actual_winner = self.determine_winner(
                row['homeTeam_score'], 
                row['awayTeam_score']
            )
            
            # Calculate bet result
            bet_result, pnl = self.calculate_bet_result(
                row['predicted_winner'],
                actual_winner,
                float(row['bet_size']),
                float(row['decimal_odds'])
            )
            
            bankroll_before = current_bankroll
            current_bankroll += pnl
            
            result = {
                'game_id': int(row['game_id']),
                'prediction_date': row['prediction_date'],
                'game_date': row['gameDate'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'predicted_team': row['predicted_team'],
                'predicted_winner': row['predicted_winner'],
                'bet_size': float(row['bet_size']),
                'decimal_odds': float(row['decimal_odds']),
                'american_odds': int(row['american_odds']),
                'model_probability': float(row['model_probability']),
                'edge': float(row['edge']),
                'home_score': int(row['homeTeam_score']),
                'away_score': int(row['awayTeam_score']),
                'actual_winner': actual_winner,
                'bet_result': bet_result,
                'pnl': pnl,
                'bankroll_before': bankroll_before,
                'bankroll_after': current_bankroll
            }
            
            results.append(result)
            
            # Log result
            emoji = "✅" if bet_result == 'WIN' else "❌"
            logger.info(
                f"{emoji} {row['away_team']} @ {row['home_team']}: "
                f"Bet {row['predicted_team']} ({row['predicted_winner']}) | "
                f"Actual: {actual_winner} | "
                f"P&L: ${pnl:+.2f} | Bankroll: ${current_bankroll:.2f}"
            )
        
        return results
    
    def save_results(self, results: List[Dict]):
        """Save bet results to PostgreSQL"""
        if not results:
            return
        
        cursor = self.pg_conn.cursor()
        
        insert_query = """
            INSERT INTO bet_results (
                game_id, prediction_date, game_date, home_team, away_team,
                predicted_team, predicted_winner, bet_size, decimal_odds,
                american_odds, model_probability, edge, home_score, away_score,
                actual_winner, bet_result, pnl, bankroll_before, bankroll_after
            ) VALUES (
                %(game_id)s, %(prediction_date)s, %(game_date)s, %(home_team)s,
                %(away_team)s, %(predicted_team)s, %(predicted_winner)s,
                %(bet_size)s, %(decimal_odds)s, %(american_odds)s,
                %(model_probability)s, %(edge)s, %(home_score)s, %(away_score)s,
                %(actual_winner)s, %(bet_result)s, %(pnl)s,
                %(bankroll_before)s, %(bankroll_after)s
            )
            ON CONFLICT (game_id, prediction_date) DO UPDATE SET
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                actual_winner = EXCLUDED.actual_winner,
                bet_result = EXCLUDED.bet_result,
                pnl = EXCLUDED.pnl,
                bankroll_before = EXCLUDED.bankroll_before,
                bankroll_after = EXCLUDED.bankroll_after,
                processed_at = NOW()
        """
        
        execute_batch(cursor, insert_query, results)
        
        # Update current bankroll
        if results:
            final_bankroll = results[-1]['bankroll_after']
            self.update_bankroll(final_bankroll)
        
        self.pg_conn.commit()
        logger.info(f"✅ Saved {len(results)} bet results to database")
    
    def save_daily_snapshot(self, snapshot_date: date, results: List[Dict]):
        """Save daily bankroll snapshot"""
        if not results:
            return
        
        # Filter results for this date
        date_results = [r for r in results 
                       if r['game_date'] == snapshot_date or 
                       (hasattr(r['game_date'], 'date') and r['game_date'].date() == snapshot_date)]
        
        if not date_results:
            return
        
        wins = sum(1 for r in date_results if r['bet_result'] == 'WIN')
        losses = sum(1 for r in date_results if r['bet_result'] == 'LOSS')
        pushes = sum(1 for r in date_results if r['bet_result'] == 'PUSH')
        
        total_pnl = sum(r['pnl'] for r in date_results)
        total_staked = sum(r['bet_size'] for r in date_results)
        
        starting = date_results[0]['bankroll_before']
        ending = date_results[-1]['bankroll_after']
        
        win_rate = wins / len(date_results) if date_results else 0
        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
        
        cursor = self.pg_conn.cursor()
        cursor.execute("""
            INSERT INTO bankroll_history (
                snapshot_date, starting_bankroll, ending_bankroll, daily_pnl,
                bets_placed, bets_won, bets_lost, bets_pushed,
                win_rate, total_staked, roi_pct
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (snapshot_date) DO UPDATE SET
                ending_bankroll = EXCLUDED.ending_bankroll,
                daily_pnl = EXCLUDED.daily_pnl,
                bets_placed = EXCLUDED.bets_placed,
                bets_won = EXCLUDED.bets_won,
                bets_lost = EXCLUDED.bets_lost,
                bets_pushed = EXCLUDED.bets_pushed,
                win_rate = EXCLUDED.win_rate,
                total_staked = EXCLUDED.total_staked,
                roi_pct = EXCLUDED.roi_pct
        """, (snapshot_date, starting, ending, total_pnl,
              len(date_results), wins, losses, pushes,
              win_rate, total_staked, roi))
        
        self.pg_conn.commit()
        logger.info(f"📊 Daily snapshot saved for {snapshot_date}")
    
    def run_backfill(self, 
                     start_date: date = None, 
                     end_date: date = None,
                     reset_bankroll: bool = False):
        """
        Run backfill to process historical predictions
        
        Args:
            start_date: Start date for backfill
            end_date: End date for backfill
            reset_bankroll: If True, reset bankroll to starting amount
        """
        logger.info("="*80)
        logger.info("HISTORICAL GAME TRACKER - BACKFILL")
        logger.info("="*80)
        
        if reset_bankroll:
            self.update_bankroll(self.starting_bankroll)
            logger.info(f"🔄 Bankroll reset to ${self.starting_bankroll:.2f}")
        
        # Fetch completed games
        games_df = self.fetch_completed_games(start_date, end_date)
        
        if len(games_df) == 0:
            logger.info("No completed games found in date range")
            return
        
        # Get game IDs
        game_ids = games_df['game_id'].tolist()
        
        # Fetch predictions
        predictions_df = self.fetch_predictions_for_games(game_ids)
        
        if len(predictions_df) == 0:
            logger.info("No predictions found for these games")
            return
        
        # Process results
        results = self.process_game_results(games_df, predictions_df)
        
        # Save results
        self.save_results(results)
        
        # Save daily snapshots
        if results:
            unique_dates = set()
            for r in results:
                gd = r['game_date']
                if hasattr(gd, 'date'):
                    unique_dates.add(gd.date())
                else:
                    unique_dates.add(gd)
            
            for snapshot_date in sorted(unique_dates):
                self.save_daily_snapshot(snapshot_date, results)
        
        # Print summary
        self.print_summary()
    
    def run_daily_update(self):
        """Run daily update to process yesterday's games"""
        logger.info("="*80)
        logger.info("HISTORICAL GAME TRACKER - DAILY UPDATE")
        logger.info("="*80)
        
        yesterday = date.today() - timedelta(days=1)
        
        # Fetch unprocessed predictions
        predictions_df = self.fetch_unprocessed_predictions()
        
        if len(predictions_df) == 0:
            logger.info("No unprocessed predictions found")
            return
        
        # Get game IDs from predictions
        game_ids = predictions_df['game_id'].tolist()
        
        # Fetch game results
        games_df = self.fetch_completed_games(
            start_date=yesterday - timedelta(days=7),
            end_date=yesterday
        )
        
        # Filter to only games we have predictions for
        games_df = games_df[games_df['game_id'].isin(game_ids)]
        
        if len(games_df) == 0:
            logger.info("No completed games found for pending predictions")
            return
        
        # Process results
        results = self.process_game_results(games_df, predictions_df)
        
        # Save results
        self.save_results(results)
        
        # Save daily snapshot
        if results:
            self.save_daily_snapshot(yesterday, results)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary statistics"""
        cursor = self.pg_conn.cursor()
        
        # Get overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_bets,
                SUM(CASE WHEN bet_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN bet_result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                SUM(bet_size) as total_staked,
                AVG(edge) as avg_edge,
                MIN(game_date) as first_bet,
                MAX(game_date) as last_bet
            FROM bet_results
        """)
        
        stats = cursor.fetchone()
        
        if stats and stats[0] > 0:
            total_bets, wins, losses, total_pnl, total_staked, avg_edge, first_bet, last_bet = stats
            win_rate = wins / total_bets if total_bets > 0 else 0
            roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
            
            current_bankroll = self.get_current_bankroll()
            total_return = ((current_bankroll - self.starting_bankroll) / self.starting_bankroll) * 100
            
            logger.info("\n" + "="*80)
            logger.info("📊 TRACKING SUMMARY")
            logger.info("="*80)
            logger.info(f"Period: {first_bet} to {last_bet}")
            logger.info(f"Total Bets: {total_bets}")
            logger.info(f"Record: {wins}W - {losses}L ({win_rate:.1%} win rate)")
            logger.info(f"Total Staked: ${total_staked:.2f}")
            logger.info(f"Total P&L: ${total_pnl:+.2f}")
            logger.info(f"ROI: {roi:+.2f}%")
            logger.info(f"Average Edge: {avg_edge:.2%}")
            logger.info(f"Starting Bankroll: ${self.starting_bankroll:.2f}")
            logger.info(f"Current Bankroll: ${current_bankroll:.2f}")
            logger.info(f"Total Return: {total_return:+.2f}%")
            logger.info("="*80)
    
    def get_performance_data(self, days: int = 30) -> Dict:
        """Get performance data for API endpoint"""
        cursor = self.pg_conn.cursor()
        
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_bets,
                SUM(CASE WHEN bet_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN bet_result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                SUM(bet_size) as total_staked,
                AVG(edge) as avg_edge
            FROM bet_results
            WHERE game_date >= CURRENT_DATE - INTERVAL '%s days'
        """, (days,))
        
        stats = cursor.fetchone()
        
        # Daily history
        cursor.execute("""
            SELECT 
                snapshot_date,
                ending_bankroll,
                daily_pnl,
                bets_placed,
                bets_won,
                bets_lost,
                win_rate,
                roi_pct
            FROM bankroll_history
            WHERE snapshot_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY snapshot_date ASC
        """, (days,))
        
        daily_history = cursor.fetchall()
        
        current_bankroll = self.get_current_bankroll()
        
        if stats[0] and stats[0] > 0:
            return {
                'current_bankroll': float(current_bankroll),
                'starting_bankroll': self.starting_bankroll,
                'total_return_pct': ((current_bankroll - self.starting_bankroll) / self.starting_bankroll) * 100,
                'period_days': days,
                'total_bets': int(stats[0]),
                'wins': int(stats[1]),
                'losses': int(stats[2]),
                'win_rate': float(stats[1] / stats[0]) if stats[0] > 0 else 0,
                'total_pnl': float(stats[3]) if stats[3] else 0,
                'total_staked': float(stats[4]) if stats[4] else 0,
                'roi_pct': float((stats[3] / stats[4] * 100)) if stats[4] and stats[4] > 0 else 0,
                'avg_edge': float(stats[5]) if stats[5] else 0,
                'daily_history': [
                    {
                        'date': str(row[0]),
                        'bankroll': float(row[1]),
                        'pnl': float(row[2]),
                        'bets': int(row[3]),
                        'wins': int(row[4]),
                        'losses': int(row[5]),
                        'win_rate': float(row[6]) if row[6] else 0,
                        'roi': float(row[7]) if row[7] else 0
                    }
                    for row in daily_history
                ]
            }
        
        return {
            'current_bankroll': float(current_bankroll),
            'starting_bankroll': self.starting_bankroll,
            'total_bets': 0,
            'message': 'No betting data available'
        }


def main():
    """Main execution"""
    
    mssql_conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-J9IV3OH;"
        "DATABASE=nhlDB;"
        "Trusted_Connection=yes;"
    )
    
    tracker = HistoricalGameTracker(
        mssql_connection_string=mssql_conn_str,
        starting_bankroll=1000.0
    )
    
    try:
        tracker.connect()
        tracker.ensure_tracking_tables()
        
        # Run backfill for last 30 days with bankroll reset
        tracker.run_backfill(
            start_date=date.today() - timedelta(days=30),
            end_date=date.today() - timedelta(days=1),
            reset_bankroll=True
        )
        
    finally:
        tracker.disconnect()


if __name__ == "__main__":
    main()