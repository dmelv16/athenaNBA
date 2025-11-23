"""
Configuration settings for NBA ETL Pipeline
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConfig:
    """Database configuration settings"""
    HOST = os.getenv('DB_HOST', 'localhost')
    PORT = os.getenv('DB_PORT', '5432')
    NAME = os.getenv('DB_NAME', 'nba_props')
    USER = os.getenv('DB_USER', 'postgres')
    PASSWORD = os.getenv('DB_PASSWORD', 'password')
    SSLMODE = os.getenv('DB_SSLMODE', 'prefer')
    TIMEOUT = int(os.getenv('DB_TIMEOUT', '10'))


class ETLConfig:
    """ETL pipeline configuration"""
    # Seasons to collect
    START_SEASON = os.getenv('START_SEASON', '2014-15')
    END_SEASON = os.getenv('END_SEASON', '2024-25')
    
    # Rate limiting for NBA API
    REQUEST_DELAY = float(os.getenv('REQUEST_DELAY', '0.6'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))
    BATCH_DELAY = int(os.getenv('BATCH_DELAY', '30'))
    
    # Retry settings
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))
    
    # Processing settings
    ENABLE_PARALLEL = os.getenv('ENABLE_PARALLEL', 'False').lower() == 'true'
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))


class LogConfig:
    """Logging configuration"""
    LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    FILE = os.getenv('LOG_FILE', 'logs/nba_etl.log')
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class AppConfig:
    """Application-wide configuration"""
    ENV = os.getenv('ENVIRONMENT', 'development')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    @staticmethod
    def is_production():
        return AppConfig.ENV == 'production'
    
    @staticmethod
    def is_development():
        return AppConfig.ENV == 'development'