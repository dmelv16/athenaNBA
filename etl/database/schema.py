"""
Database schema definitions
"""

from database.connection import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)


# SQL statements for table creation
CREATE_PLAYERS_TABLE = """
    CREATE TABLE IF NOT EXISTS players (
        player_id INTEGER PRIMARY KEY,
        full_name VARCHAR(100),
        first_name VARCHAR(50),
        last_name VARCHAR(50),
        is_active BOOLEAN,
        position VARCHAR(20),
        height VARCHAR(10),
        weight VARCHAR(10),
        birthdate DATE,
        country VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""

CREATE_TEAMS_TABLE = """
    CREATE TABLE IF NOT EXISTS teams (
        team_id INTEGER PRIMARY KEY,
        team_name VARCHAR(100),
        abbreviation VARCHAR(10),
        city VARCHAR(50),
        state VARCHAR(50),
        year_founded INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""

CREATE_PLAYER_GAME_LOGS_TABLE = """
    CREATE TABLE IF NOT EXISTS player_game_logs (
        id SERIAL PRIMARY KEY,
        game_id VARCHAR(20),
        player_id INTEGER REFERENCES players(player_id),
        season VARCHAR(10),
        game_date DATE,
        matchup VARCHAR(20),
        wl VARCHAR(1),
        min DECIMAL,
        fgm INTEGER,
        fga INTEGER,
        fg_pct DECIMAL,
        fg3m INTEGER,
        fg3a INTEGER,
        fg3_pct DECIMAL,
        ftm INTEGER,
        fta INTEGER,
        ft_pct DECIMAL,
        oreb INTEGER,
        dreb INTEGER,
        reb INTEGER,
        ast INTEGER,
        stl INTEGER,
        blk INTEGER,
        tov INTEGER,
        pf INTEGER,
        pts INTEGER,
        plus_minus INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(game_id, player_id)
    )
"""

CREATE_TEAM_GAME_LOGS_TABLE = """
    CREATE TABLE IF NOT EXISTS team_game_logs (
        id SERIAL PRIMARY KEY,
        game_id VARCHAR(20),
        team_id INTEGER REFERENCES teams(team_id),
        season VARCHAR(10),
        game_date DATE,
        matchup VARCHAR(20),
        wl VARCHAR(1),
        min INTEGER,
        fgm INTEGER,
        fga INTEGER,
        fg_pct DECIMAL,
        fg3m INTEGER,
        fg3a INTEGER,
        fg3_pct DECIMAL,
        ftm INTEGER,
        fta INTEGER,
        ft_pct DECIMAL,
        oreb INTEGER,
        dreb INTEGER,
        reb INTEGER,
        ast INTEGER,
        stl INTEGER,
        blk INTEGER,
        tov INTEGER,
        pf INTEGER,
        pts INTEGER,
        plus_minus INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(game_id, team_id)
    )
"""

CREATE_PLAYER_OPPONENT_STATS_TABLE = """
    CREATE TABLE IF NOT EXISTS player_opponent_stats (
        id SERIAL PRIMARY KEY,
        player_id INTEGER REFERENCES players(player_id),
        season VARCHAR(10),
        opponent_team_id INTEGER REFERENCES teams(team_id),
        gp INTEGER,
        w INTEGER,
        l INTEGER,
        min DECIMAL,
        fgm DECIMAL,
        fga DECIMAL,
        fg_pct DECIMAL,
        fg3m DECIMAL,
        fg3a DECIMAL,
        fg3_pct DECIMAL,
        ftm DECIMAL,
        fta DECIMAL,
        ft_pct DECIMAL,
        oreb DECIMAL,
        dreb DECIMAL,
        reb DECIMAL,
        ast DECIMAL,
        tov DECIMAL,
        stl DECIMAL,
        blk DECIMAL,
        blka DECIMAL,
        pf DECIMAL,
        pfd DECIMAL,
        pts DECIMAL,
        plus_minus DECIMAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(player_id, season, opponent_team_id)
    )
"""

CREATE_PLAYER_GENERAL_SPLITS_TABLE = """
    CREATE TABLE IF NOT EXISTS player_general_splits (
        id SERIAL PRIMARY KEY,
        player_id INTEGER REFERENCES players(player_id),
        season VARCHAR(10),
        split_type VARCHAR(50),
        split_value VARCHAR(50),
        gp INTEGER,
        w INTEGER,
        l INTEGER,
        min DECIMAL,
        fgm DECIMAL,
        fga DECIMAL,
        fg_pct DECIMAL,
        fg3m DECIMAL,
        fg3a DECIMAL,
        fg3_pct DECIMAL,
        ftm DECIMAL,
        fta DECIMAL,
        ft_pct DECIMAL,
        oreb DECIMAL,
        dreb DECIMAL,
        reb DECIMAL,
        ast DECIMAL,
        tov DECIMAL,
        stl DECIMAL,
        blk DECIMAL,
        blka DECIMAL,
        pf DECIMAL,
        pfd DECIMAL,
        pts DECIMAL,
        plus_minus DECIMAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(player_id, season, split_type, split_value)
    )
"""

# Index creation statements
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_player_game_logs_player_season ON player_game_logs(player_id, season)",
    "CREATE INDEX IF NOT EXISTS idx_player_game_logs_date ON player_game_logs(game_date)",
    "CREATE INDEX IF NOT EXISTS idx_player_game_logs_game_id ON player_game_logs(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_team_game_logs_team_season ON team_game_logs(team_id, season)",
    "CREATE INDEX IF NOT EXISTS idx_team_game_logs_date ON team_game_logs(game_date)",
    "CREATE INDEX IF NOT EXISTS idx_player_opponent_stats_player ON player_opponent_stats(player_id, season)",
    "CREATE INDEX IF NOT EXISTS idx_player_splits_player ON player_general_splits(player_id, season)"
]


class SchemaManager:
    """Manages database schema creation and updates"""
    
    def __init__(self):
        self.db = get_db_connection()
    
    def create_all_tables(self):
        """Create all tables in the database"""
        logger.info("Creating database tables...")
        
        tables = [
            ('players', CREATE_PLAYERS_TABLE),
            ('teams', CREATE_TEAMS_TABLE),
            ('player_game_logs', CREATE_PLAYER_GAME_LOGS_TABLE),
            ('team_game_logs', CREATE_TEAM_GAME_LOGS_TABLE),
            ('player_opponent_stats', CREATE_PLAYER_OPPONENT_STATS_TABLE),
            ('player_general_splits', CREATE_PLAYER_GENERAL_SPLITS_TABLE)
        ]
        
        for table_name, create_sql in tables:
            try:
                with self.db.get_cursor() as cur:
                    cur.execute(create_sql)
                logger.info(f"✓ Table '{table_name}' created/verified")
            except Exception as e:
                logger.error(f"✗ Failed to create table '{table_name}': {e}")
                raise
        
        logger.info("All tables created successfully")
    
    def create_indexes(self):
        """Create all indexes"""
        logger.info("Creating database indexes...")
        
        for idx, index_sql in enumerate(CREATE_INDEXES, 1):
            try:
                with self.db.get_cursor() as cur:
                    cur.execute(index_sql)
                logger.info(f"✓ Index {idx}/{len(CREATE_INDEXES)} created")
            except Exception as e:
                logger.error(f"✗ Failed to create index: {e}")
                # Don't raise - indexes are not critical
        
        logger.info("All indexes created successfully")
    
    def initialize_schema(self):
        """Initialize complete database schema"""
        logger.info("Initializing database schema...")
        self.create_all_tables()
        self.create_indexes()
        logger.info("Database schema initialized successfully")
    
    def drop_all_tables(self, confirm: bool = False):
        """
        Drop all tables (use with caution!)
        
        Args:
            confirm: Must be True to actually drop tables
        """
        if not confirm:
            logger.warning("drop_all_tables called without confirmation - skipping")
            return
        
        logger.warning("Dropping all tables...")
        
        tables = [
            'player_general_splits',
            'player_opponent_stats',
            'team_game_logs',
            'player_game_logs',
            'teams',
            'players'
        ]
        
        for table in tables:
            try:
                with self.db.get_cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                logger.info(f"✓ Table '{table}' dropped")
            except Exception as e:
                logger.error(f"✗ Failed to drop table '{table}': {e}")