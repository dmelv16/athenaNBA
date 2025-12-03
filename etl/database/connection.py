"""
Database connection management
"""

import psycopg2
from psycopg2.extensions import connection as PGConnection
from typing import Optional
from contextlib import contextmanager

from etl.config.settings import DatabaseConfig
from etl.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseConnection:
    """Manages PostgreSQL database connections"""
    
    def __init__(self):
        self._conn: Optional[PGConnection] = None
    
    def connect(self) -> PGConnection:
        """
        Establish database connection
        
        Returns:
            PostgreSQL connection object
            
        Raises:
            psycopg2.OperationalError: If connection fails
        """
        if self._conn is not None and not self._conn.closed:
            return self._conn
        
        try:
            connection_params = {
                'host': DatabaseConfig.HOST,
                'port': DatabaseConfig.PORT,
                'dbname': DatabaseConfig.NAME,
                'user': DatabaseConfig.USER,
                'password': DatabaseConfig.PASSWORD,
                'sslmode': DatabaseConfig.SSLMODE,
                'connect_timeout': DatabaseConfig.TIMEOUT
            }
            
            self._conn = psycopg2.connect(**connection_params)
            logger.info(
                f"Database connection established: "
                f"{DatabaseConfig.HOST}:{DatabaseConfig.PORT}/{DatabaseConfig.NAME}"
            )
            return self._conn
            
        except psycopg2.OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            logger.info("Database connection closed")
            self._conn = None
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
                result = cur.fetchone()
                return result[0] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_connection(self) -> PGConnection:
        """
        Get current connection (creates if doesn't exist)
        
        Returns:
            PostgreSQL connection object
        """
        if self._conn is None or self._conn.closed:
            return self.connect()
        return self._conn
    
    @contextmanager
    def get_cursor(self):
        """
        Context manager for database cursor
        
        Yields:
            Database cursor
            
        Example:
            with db.get_cursor() as cur:
                cur.execute("SELECT * FROM players")
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Singleton instance
_db_instance: Optional[DatabaseConnection] = None


def get_db_connection() -> DatabaseConnection:
    """
    Get singleton database connection instance
    
    Returns:
        DatabaseConnection instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance