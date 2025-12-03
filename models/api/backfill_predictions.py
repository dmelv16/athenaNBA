"""
Backfill NBA Predictions
Generate predictions for historical games (for backtesting/analysis)

Usage:
    python -m models.api.backfill_predictions --start 2025-10-22 --end 2025-12-01
    python -m models.api.backfill_predictions --days 30
"""

import argparse
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.api.run_daily_predictions import DailyPredictionRunner


def backfill_predictions(
    start_date: date,
    end_date: date,
    skip_existing: bool = True
):
    """
    Generate predictions for a date range
    
    Args:
        start_date: First date to predict
        end_date: Last date to predict (exclusive - won't predict this day)
        skip_existing: Skip dates that already have predictions
    """
    print("\n" + "=" * 60)
    print("🏀 NBA PREDICTION BACKFILL")
    print("=" * 60)
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    
    # Calculate total days
    total_days = (end_date - start_date).days
    print(f"Total Days: {total_days}")
    
    # Initialize runner once
    runner = DailyPredictionRunner()
    runner.initialize()
    
    # Track stats
    processed = 0
    skipped = 0
    errors = 0
    total_predictions = 0
    
    current_date = start_date
    
    while current_date < end_date:
        print(f"\n{'='*60}")
        print(f"📅 Processing: {current_date} ({processed + 1}/{total_days})")
        print(f"{'='*60}")
        
        try:
            # Check if predictions already exist for this date
            if skip_existing:
                existing = runner.uploader.get_predictions_by_date(current_date)
                if existing.get('player_predictions'):
                    print(f"  ⏭️  Skipping - {len(existing['player_predictions'])} predictions already exist")
                    skipped += 1
                    current_date += timedelta(days=1)
                    continue
            
            # Generate predictions for this date
            predictions = runner.generate_predictions(current_date)
            
            if predictions['player_predictions'] or predictions['team_predictions']:
                # Upload to database
                runner.upload_predictions(predictions, current_date)
                
                day_total = len(predictions['player_predictions']) + len(predictions['team_predictions'])
                total_predictions += day_total
                processed += 1
                
                print(f"  ✓ Generated {day_total} predictions")
            else:
                print(f"  ⚠️  No games on this date")
                skipped += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            errors += 1
        
        # Move to next day
        current_date += timedelta(days=1)
        
        # Small delay to avoid overwhelming APIs
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"Days Processed: {processed}")
    print(f"Days Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Total Predictions: {total_predictions}")
    print("=" * 60)
    
    # Cleanup
    if runner.uploader:
        runner.uploader.close()
    if runner.data_loader:
        runner.data_loader.close()


def main():
    parser = argparse.ArgumentParser(
        description='Backfill NBA Predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Backfill from season start to yesterday
    python -m models.api.backfill_predictions --start 2025-10-22 --end 2025-12-02
    
    # Backfill last 30 days
    python -m models.api.backfill_predictions --days 30
    
    # Backfill specific range, overwrite existing
    python -m models.api.backfill_predictions --start 2025-11-01 --end 2025-11-15 --overwrite
        """
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD). Defaults to today.'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        help='Number of days to backfill (alternative to --start/--end)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing predictions (default: skip)'
    )
    
    args = parser.parse_args()
    
    # Determine date range
    if args.days:
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days)
    elif args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        if args.end:
            end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
        else:
            end_date = date.today()
    else:
        print("Error: Must specify either --start or --days")
        print("Run with --help for usage information")
        sys.exit(1)
    
    # Validate
    if start_date >= end_date:
        print(f"Error: Start date ({start_date}) must be before end date ({end_date})")
        sys.exit(1)
    
    if end_date > date.today():
        print(f"Warning: End date adjusted to today ({date.today()})")
        end_date = date.today()
    
    # Run backfill
    backfill_predictions(
        start_date=start_date,
        end_date=end_date,
        skip_existing=not args.overwrite
    )


if __name__ == "__main__":
    main()