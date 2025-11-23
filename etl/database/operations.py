"""
Database operations for data loading
"""

import pandas as pd
from typing import List, Dict, Any
from psycopg2.extras import execute_values

from database.connection import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Handles loading data into database"""
    
    def __init__(self):
        self.db = get_db_connection()
    
    def insert_players(self, players_data: List[Dict[str, Any]]):
        """
        Insert or update players data
        
        Args:
            players_data: List of player dictionaries
        """
        if not players_data:
            logger.warning("No players data to insert")
            return
        
        logger.info(f"Inserting {len(players_data)} players...")
        
        query = """
            INSERT INTO players (player_id, full_name, first_name, last_name, is_active)
            VALUES %s
            ON CONFLICT (player_id) DO UPDATE SET
                full_name = EXCLUDED.full_name,
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                is_active = EXCLUDED.is_active,
                updated_at = CURRENT_TIMESTAMP
        """
        
        values = [
            (p['id'], p['full_name'], p['first_name'], p['last_name'], p['is_active'])
            for p in players_data
        ]
        
        with self.db.get_cursor() as cur:
            execute_values(cur, query, values)
        
        logger.info(f"{len(players_data)} players inserted/updated")
    
    def insert_teams(self, teams_data: List[Dict[str, Any]]):
        """
        Insert or update teams data
        
        Args:
            teams_data: List of team dictionaries
        """
        if not teams_data:
            logger.warning("No teams data to insert")
            return
        
        logger.info(f"Inserting {len(teams_data)} teams...")
        
        query = """
            INSERT INTO teams (team_id, team_name, abbreviation, city, state, year_founded)
            VALUES %s
            ON CONFLICT (team_id) DO UPDATE SET
                team_name = EXCLUDED.team_name,
                abbreviation = EXCLUDED.abbreviation,
                city = EXCLUDED.city,
                state = EXCLUDED.state,
                year_founded = EXCLUDED.year_founded,
                updated_at = CURRENT_TIMESTAMP
        """
        
        values = [
            (t['id'], t['full_name'], t['abbreviation'], t['city'], t['state'], t['year_founded'])
            for t in teams_data
        ]
        
        with self.db.get_cursor() as cur:
            execute_values(cur, query, values)
        
        logger.info(f"{len(teams_data)} teams inserted/updated")
    
    def insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        conflict_columns: List[str]
    ):
        """
        Insert pandas DataFrame into database with upsert logic
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            conflict_columns: Columns to use for conflict resolution
        """
        if df.empty:
            logger.warning(f"Empty dataframe for table {table_name}, skipping...")
            return
        
        # Replace NaN with None for proper NULL handling
        df = df.where(pd.notnull(df), None)
        
        columns = list(df.columns)
        values = [tuple(row) for row in df.values]
        
        # Build the INSERT query
        cols_str = ', '.join(columns)
        conflict_str = ', '.join(conflict_columns)
        update_cols = [col for col in columns if col not in conflict_columns]
        update_str = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_cols])
        
        query = f"""
            INSERT INTO {table_name} ({cols_str})
            VALUES %s
            ON CONFLICT ({conflict_str}) DO UPDATE SET
                {update_str}
        """
        
        try:
            with self.db.get_cursor() as cur:
                execute_values(cur, query, values)
            logger.info(f"Inserted {len(df)} rows into {table_name}")
        except Exception as e:
            logger.error(f"✗ Failed to insert into {table_name}: {e}")
            raise
    
    def bulk_insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        conflict_columns: List[str],
        batch_size: int = 1000
    ):
        """
        Insert large DataFrame in batches
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            conflict_columns: Columns for conflict resolution
            batch_size: Number of rows per batch
        """
        if df.empty:
            logger.warning(f"Empty dataframe for table {table_name}, skipping...")
            return
        
        total_rows = len(df)
        logger.info(f"Bulk inserting {total_rows} rows into {table_name} (batch size: {batch_size})")
        
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i + batch_size]
            self.insert_dataframe(batch, table_name, conflict_columns)
            logger.info(f"  Progress: {min(i + batch_size, total_rows)}/{total_rows} rows")
        
        logger.info(f"Bulk insert completed: {total_rows} rows into {table_name}")