"""
NBA ETL Pipeline - Main Entry Point

Usage examples:
    # Full pipeline
    python main.py --mode full
    
    # Static data only
    python main.py --mode static
    
    # Player data only
    python main.py --mode players
    
    # Team data only
    python main.py --mode teams
    
    # Incremental update for current season
    python main.py --mode incremental --season 2024-25
    
    # Test with 5 players
    python main.py --mode full --test
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestration.pipeline import ETLPipeline
from config.settings import ETLConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='NBA Data ETL Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full                    # Run full pipeline
  python main.py --mode static                  # Load static data only
  python main.py --mode incremental --season 2024-25  # Update current season
  python main.py --mode full --test             # Test run with 5 players
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'static', 'players', 'teams', 'incremental'],
        required=True,
        help='ETL mode to run'
    )
    
    parser.add_argument(
        '--season',
        type=str,
        help='Season for incremental mode (e.g., 2024-25)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only 5 players'
    )
    
    parser.add_argument(
        '--players',
        type=str,
        help='Comma-separated list of player IDs to process'
    )
    
    parser.add_argument(
        '--seasons',
        type=str,
        help='Comma-separated list of seasons to process (e.g., 2023-24,2024-25)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Initialize pipeline
    pipeline = ETLPipeline()
    
    # Parse filters
    player_filter = None
    season_filter = None
    
    if args.test:
        # Get first 5 player IDs for testing
        from extractors.static_extractor import StaticDataExtractor
        extractor = StaticDataExtractor()
        all_players = extractor.get_all_players()
        player_filter = [p['id'] for p in all_players[:5]]
        logger.info(f" TEST MODE: Processing only 5 players")
    
    if args.players:
        player_filter = [int(pid.strip()) for pid in args.players.split(',')]
        logger.info(f" Player filter: {len(player_filter)} players")
    
    if args.seasons:
        season_filter = [s.strip() for s in args.seasons.split(',')]
        logger.info(f" Season filter: {season_filter}")
    
    # Run appropriate pipeline mode
    try:
        if args.mode == 'full':
            pipeline.run_full_pipeline(
                player_filter=player_filter,
                season_filter=season_filter
            )
        
        elif args.mode == 'static':
            pipeline.run_static_data_only()
        
        elif args.mode == 'players':
            pipeline.run_player_data_only(
                player_filter=player_filter,
                season_filter=season_filter
            )
        
        elif args.mode == 'teams':
            pipeline.run_team_data_only(
                season_filter=season_filter
            )
        
        elif args.mode == 'incremental':
            if not args.season:
                logger.error(" --season required for incremental mode")
                sys.exit(1)
            pipeline.run_incremental_update(args.season)
        
        logger.info("✅ Pipeline completed successfully!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("\n  Pipeline interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"\n Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()