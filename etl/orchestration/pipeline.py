"""
Main ETL pipeline orchestrator
"""

import time
from datetime import datetime
from typing import List, Optional

from database.connection import get_db_connection
from database.schema import SchemaManager
from orchestration.tasks import ETLTasks
from utils.logger import get_logger

logger = get_logger(__name__)


class ETLPipeline:
    """Main ETL pipeline orchestrator"""
    
    def __init__(self):
        self.db = get_db_connection()
        self.schema_manager = SchemaManager()
        self.tasks = ETLTasks()
        self.start_time = None
    
    def initialize_database(self):
        """Initialize database schema"""
        logger.info("Initializing database schema...")
        self.db.connect()
        self.schema_manager.initialize_schema()
    
    def run_full_pipeline(
        self,
        player_filter: Optional[List[int]] = None,
        season_filter: Optional[List[str]] = None
    ):
        """
        Run the complete ETL pipeline
        
        Args:
            player_filter: Optional list of player IDs to process
            season_filter: Optional list of seasons to process
        """
        self.start_time = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info("NBA DATA ETL PIPELINE - FULL RUN")
        logger.info("=" * 70)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70 + "\n")
        
        try:
            # Initialize
            self.initialize_database()
            
            # Run tasks
            self.tasks.load_static_data()
            self.tasks.load_player_game_logs(player_filter, season_filter)
            self.tasks.load_player_opponent_stats(player_filter, season_filter)
            self.tasks.load_player_splits(player_filter, season_filter)
            self.tasks.load_team_game_logs(season_filter=season_filter)
            
            # Summary
            self._print_summary()
            
        except KeyboardInterrupt:
            logger.warning("\n\n⚠️  Pipeline interrupted by user")
            self._print_summary()
            raise
            
        except Exception as e:
            logger.error(f"\n\n❌ Pipeline failed: {e}")
            self._print_summary()
            raise
            
        finally:
            self.db.close()
    
    def run_static_data_only(self):
        """Run only static data loading"""
        self.start_time = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info("NBA DATA ETL PIPELINE - STATIC DATA ONLY")
        logger.info("=" * 70 + "\n")
        
        try:
            self.initialize_database()
            self.tasks.load_static_data()
            self._print_summary()
        except Exception as e:
            logger.error(f"Static data loading failed: {e}")
            raise
        finally:
            self.db.close()
    
    def run_player_data_only(
        self,
        player_filter: Optional[List[int]] = None,
        season_filter: Optional[List[str]] = None
    ):
        """Run only player data loading"""
        self.start_time = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info("NBA DATA ETL PIPELINE - PLAYER DATA ONLY")
        logger.info("=" * 70 + "\n")
        
        try:
            self.db.connect()
            self.tasks.load_player_game_logs(player_filter, season_filter)
            self.tasks.load_player_opponent_stats(player_filter, season_filter)
            self.tasks.load_player_splits(player_filter, season_filter)
            self._print_summary()
        except Exception as e:
            logger.error(f"Player data loading failed: {e}")
            raise
        finally:
            self.db.close()
    
    def run_team_data_only(
        self,
        team_filter: Optional[List[int]] = None,
        season_filter: Optional[List[str]] = None
    ):
        """Run only team data loading"""
        self.start_time = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info("NBA DATA ETL PIPELINE - TEAM DATA ONLY")
        logger.info("=" * 70 + "\n")
        
        try:
            self.db.connect()
            self.tasks.load_team_game_logs(team_filter, season_filter)
            self._print_summary()
        except Exception as e:
            logger.error(f"Team data loading failed: {e}")
            raise
        finally:
            self.db.close()
    
    def run_incremental_update(self, season: str):
        """
        Run incremental update for a specific season
        
        Args:
            season: Season to update (e.g., '2024-25')
        """
        self.start_time = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"NBA DATA ETL PIPELINE - INCREMENTAL UPDATE ({season})")
        logger.info("=" * 70 + "\n")
        
        try:
            self.db.connect()
            
            # Update static data
            self.tasks.load_static_data()
            
            # Update only specified season
            self.tasks.load_player_game_logs(season_filter=[season])
            self.tasks.load_player_opponent_stats(season_filter=[season])
            self.tasks.load_player_splits(season_filter=[season])
            self.tasks.load_team_game_logs(season_filter=[season])
            
            self._print_summary()
            
        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            raise
        finally:
            self.db.close()
    
    def _print_summary(self):
        """Print pipeline execution summary"""
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total time: {hours}h {minutes}m {seconds}s")
        
        # Get extraction stats
        player_stats = self.tasks.player_extractor.get_stats()
        team_stats = self.tasks.team_extractor.get_stats()
        
        logger.info(f"\nPlayer API Requests: {player_stats['total_requests']}")
        logger.info(f"Player API Errors: {player_stats['total_errors']}")
        logger.info(f"Player Success Rate: {player_stats['success_rate']:.2f}%")
        
        logger.info(f"\nTeam API Requests: {team_stats['total_requests']}")
        logger.info(f"Team API Errors: {team_stats['total_errors']}")
        logger.info(f"Team Success Rate: {team_stats['success_rate']:.2f}%")
        
        logger.info("\n" + "=" * 70)
        logger.info("Pipeline execution completed")
        logger.info("=" * 70 + "\n")