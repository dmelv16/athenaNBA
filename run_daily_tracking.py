"""
Daily Tracking Script
Run this after games complete to update results and bankroll

Usage:
    python run_daily_tracking.py                    # Daily update
    python run_daily_tracking.py --backfill 30     # Backfill last 30 days
    python run_daily_tracking.py --reset 1000      # Reset bankroll to $1000
"""

import argparse
from datetime import date, timedelta
from src.tracking.historical_tracker import HistoricalGameTracker
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='NHL Betting Tracker')
    parser.add_argument('--backfill', type=int, default=None,
                       help='Number of days to backfill')
    parser.add_argument('--reset', type=float, default=None,
                       help='Reset bankroll to this amount')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for backfill (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for backfill (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    mssql_conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-J9IV3OH;"
        "DATABASE=nhlDB;"
        "Trusted_Connection=yes;"
    )
    
    # Determine starting bankroll
    starting_bankroll = args.reset if args.reset else 1000.0
    
    tracker = HistoricalGameTracker(
        mssql_connection_string=mssql_conn_str,
        starting_bankroll=starting_bankroll
    )
    
    try:
        tracker.connect()
        tracker.ensure_tracking_tables()
        
        if args.backfill:
            # Run backfill
            end_date = date.today() - timedelta(days=1)
            start_date = end_date - timedelta(days=args.backfill)
            
            if args.start_date:
                start_date = date.fromisoformat(args.start_date)
            if args.end_date:
                end_date = date.fromisoformat(args.end_date)
            
            logger.info(f"Running backfill from {start_date} to {end_date}")
            
            tracker.run_backfill(
                start_date=start_date,
                end_date=end_date,
                reset_bankroll=(args.reset is not None)
            )
        else:
            # Run daily update
            logger.info("Running daily update")
            tracker.run_daily_update()
        
        # Print final summary
        tracker.print_summary()
        
    finally:
        tracker.disconnect()


if __name__ == "__main__":
    main()