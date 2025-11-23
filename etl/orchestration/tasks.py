"""
Individual ETL tasks
"""

from typing import List, Dict, Any

from config.settings import ETLConfig
from database.operations import DataLoader
from extractors.static_extractor import StaticDataExtractor
from extractors.player_extractor import PlayerDataExtractor
from extractors.team_extractor import TeamDataExtractor
from transformers.game_log_transformer import GameLogTransformer
from transformers.opponent_transformer import OpponentStatsTransformer
from transformers.splits_transformer import SplitsTransformer
from utils.logger import get_logger

logger = get_logger(__name__)


class ETLTasks:
    """Collection of individual ETL tasks"""
    
    def __init__(self):
        self.static_extractor = StaticDataExtractor()
        self.player_extractor = PlayerDataExtractor()
        self.team_extractor = TeamDataExtractor()
        self.data_loader = DataLoader()
    
    def load_static_data(self):
        """Load players and teams static data"""
        logger.info("=" * 70)
        logger.info("Task: Loading Static Data (Players & Teams)")
        logger.info("=" * 70)
        
        # Load players
        players = self.static_extractor.get_all_players()
        self.data_loader.insert_players(players)
        
        # Load teams
        teams = self.static_extractor.get_all_teams()
        self.data_loader.insert_teams(teams)
        
        logger.info("✓ Static data loaded successfully\n")
    
    def load_player_game_logs(
        self,
        player_filter: List[int] = None,
        season_filter: List[str] = None
    ):
        """
        Load player game logs
        
        Args:
            player_filter: List of player IDs to process (None = all)
            season_filter: List of seasons to process (None = all from config)
        """
        logger.info("=" * 70)
        logger.info("Task: Loading Player Game Logs")
        logger.info("=" * 70)
        
        # Get players and seasons
        all_players = self.static_extractor.get_all_players()
        
        if player_filter:
            all_players = [p for p in all_players if p['id'] in player_filter]
        
        seasons = season_filter or self.static_extractor.get_seasons_list(
            ETLConfig.START_SEASON,
            ETLConfig.END_SEASON
        )
        
        total_players = len(all_players)
        logger.info(f"Processing {total_players} players across {len(seasons)} seasons\n")
        
        for idx, player in enumerate(all_players, 1):
            player_id = player['id']
            player_name = player['full_name']
            
            logger.info(f"[{idx}/{total_players}] {player_name}")
            
            for season in seasons:
                # Extract
                df = self.player_extractor.get_player_game_logs(
                    player_id, season, player_name
                )
                
                # Transform
                df_clean = GameLogTransformer.transform_player_game_log(df)
                
                # Load
                if df_clean is not None and not df_clean.empty:
                    self.data_loader.insert_dataframe(
                        df_clean,
                        'player_game_logs',
                        ['game_id', 'player_id']
                    )
        
        logger.info(f"\n✓ Player game logs loaded successfully\n")
    
    def load_player_opponent_stats(
        self,
        player_filter: List[int] = None,
        season_filter: List[str] = None
    ):
        """
        Load player vs opponent statistics
        
        Args:
            player_filter: List of player IDs to process
            season_filter: List of seasons to process
        """
        logger.info("=" * 70)
        logger.info("Task: Loading Player Opponent Stats")
        logger.info("=" * 70)
        
        all_players = self.static_extractor.get_all_players()
        
        if player_filter:
            all_players = [p for p in all_players if p['id'] in player_filter]
        
        seasons = season_filter or self.static_extractor.get_seasons_list(
            ETLConfig.START_SEASON,
            ETLConfig.END_SEASON
        )
        
        total_players = len(all_players)
        logger.info(f"Processing {total_players} players across {len(seasons)} seasons\n")
        
        for idx, player in enumerate(all_players, 1):
            player_id = player['id']
            player_name = player['full_name']
            
            logger.info(f"[{idx}/{total_players}] {player_name}")
            
            for season in seasons:
                # Extract
                df = self.player_extractor.get_player_opponent_stats(
                    player_id, season, player_name
                )
                
                # Transform
                df_clean = OpponentStatsTransformer.transform_opponent_stats(df, player_id)
                
                # Load
                if df_clean is not None and not df_clean.empty:
                    self.data_loader.insert_dataframe(
                        df_clean,
                        'player_opponent_stats',
                        ['player_id', 'season', 'opponent_team_id']
                    )
        
        logger.info(f"\n✓ Player opponent stats loaded successfully\n")
    
    def load_player_splits(
        self,
        player_filter: List[int] = None,
        season_filter: List[str] = None
    ):
        """
        Load player general splits
        
        Args:
            player_filter: List of player IDs to process
            season_filter: List of seasons to process
        """
        logger.info("=" * 70)
        logger.info("Task: Loading Player Splits")
        logger.info("=" * 70)
        
        all_players = self.static_extractor.get_all_players()
        
        if player_filter:
            all_players = [p for p in all_players if p['id'] in player_filter]
        
        seasons = season_filter or self.static_extractor.get_seasons_list(
            ETLConfig.START_SEASON,
            ETLConfig.END_SEASON
        )
        
        total_players = len(all_players)
        logger.info(f"Processing {total_players} players across {len(seasons)} seasons\n")
        
        for idx, player in enumerate(all_players, 1):
            player_id = player['id']
            player_name = player['full_name']
            
            logger.info(f"[{idx}/{total_players}] {player_name}")
            
            for season in seasons:
                # Extract
                df = self.player_extractor.get_player_general_splits(
                    player_id, season, player_name
                )
                
                # Transform
                df_clean = SplitsTransformer.transform_splits(df, player_id)
                
                # Load
                if df_clean is not None and not df_clean.empty:
                    self.data_loader.insert_dataframe(
                        df_clean,
                        'player_general_splits',
                        ['player_id', 'season', 'split_type', 'split_value']
                    )
        
        logger.info(f"\n✓ Player splits loaded successfully\n")
    
    def load_team_game_logs(
        self,
        team_filter: List[int] = None,
        season_filter: List[str] = None
    ):
        """
        Load team game logs
        
        Args:
            team_filter: List of team IDs to process
            season_filter: List of seasons to process
        """
        logger.info("=" * 70)
        logger.info("Task: Loading Team Game Logs")
        logger.info("=" * 70)
        
        all_teams = self.static_extractor.get_all_teams()
        
        if team_filter:
            all_teams = [t for t in all_teams if t['id'] in team_filter]
        
        seasons = season_filter or self.static_extractor.get_seasons_list(
            ETLConfig.START_SEASON,
            ETLConfig.END_SEASON
        )
        
        total_teams = len(all_teams)
        logger.info(f"Processing {total_teams} teams across {len(seasons)} seasons\n")
        
        for idx, team in enumerate(all_teams, 1):
            team_id = team['id']
            team_name = team['full_name']
            
            logger.info(f"[{idx}/{total_teams}] {team_name}")
            
            for season in seasons:
                # Extract
                df = self.team_extractor.get_team_game_logs(
                    team_id, season, team_name
                )
                
                # Transform
                df_clean = GameLogTransformer.transform_team_game_log(df)
                
                # Load
                if df_clean is not None and not df_clean.empty:
                    self.data_loader.insert_dataframe(
                        df_clean,
                        'team_game_logs',
                        ['game_id', 'team_id']
                    )
        
        logger.info(f"\n✓ Team game logs loaded successfully\n")