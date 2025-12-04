-- NBA Odds Database Schema
-- Tables for storing historical odds data from OddsPAPI

-- ==========================================
-- Game Lines (Spread, Total, Moneyline)
-- ==========================================
CREATE TABLE IF NOT EXISTS nba_game_odds (
    id SERIAL PRIMARY KEY,
    
    -- Game identification
    game_id VARCHAR(50) NOT NULL,
    game_date DATE NOT NULL,
    snapshot_time TIMESTAMP NOT NULL,  -- When odds were captured
    
    -- Teams
    home_team VARCHAR(50) NOT NULL,
    away_team VARCHAR(50) NOT NULL,
    home_team_abbrev VARCHAR(10),
    away_team_abbrev VARCHAR(10),
    home_team_id INTEGER,
    away_team_id INTEGER,
    
    -- Spread
    spread_line DECIMAL(5,2),          -- Home team spread (e.g., -3.5)
    spread_home_odds INTEGER,          -- American odds
    spread_away_odds INTEGER,
    
    -- Total (Over/Under)
    total_line DECIMAL(5,2),           -- O/U line (e.g., 224.5)
    over_odds INTEGER,
    under_odds INTEGER,
    
    -- Moneyline
    home_ml INTEGER,                   -- Home moneyline
    away_ml INTEGER,                   -- Away moneyline
    
    -- Metadata
    bookmaker VARCHAR(50) NOT NULL,
    is_pregame BOOLEAN DEFAULT TRUE,   -- False if captured during game
    odds_source VARCHAR(50) DEFAULT 'oddspapi',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint: one record per game/bookmaker/snapshot
    UNIQUE(game_id, bookmaker, snapshot_time)
);

-- ==========================================
-- Player Prop Lines
-- ==========================================
CREATE TABLE IF NOT EXISTS nba_player_prop_odds (
    id SERIAL PRIMARY KEY,
    
    -- Game identification
    game_id VARCHAR(50) NOT NULL,
    game_date DATE NOT NULL,
    snapshot_time TIMESTAMP NOT NULL,
    
    -- Player identification
    player_name VARCHAR(100) NOT NULL,
    player_id INTEGER,                  -- NBA player ID if matched
    team_abbrev VARCHAR(10),
    opponent_abbrev VARCHAR(10),
    
    -- Prop details
    prop_type VARCHAR(20) NOT NULL,    -- pts, reb, ast, pra, etc.
    line DECIMAL(6,2) NOT NULL,        -- The line (e.g., 24.5)
    over_odds INTEGER NOT NULL,        -- American odds for over
    under_odds INTEGER NOT NULL,       -- American odds for under
    
    -- Metadata
    bookmaker VARCHAR(50) NOT NULL,
    is_pregame BOOLEAN DEFAULT TRUE,
    odds_source VARCHAR(50) DEFAULT 'oddspapi',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint
    UNIQUE(game_id, player_name, prop_type, bookmaker, snapshot_time)
);

-- ==========================================
-- Odds Snapshots Log (tracking when we captured data)
-- ==========================================
CREATE TABLE IF NOT EXISTS nba_odds_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_time TIMESTAMP NOT NULL,
    game_date DATE NOT NULL,
    games_captured INTEGER DEFAULT 0,
    player_props_captured INTEGER DEFAULT 0,
    game_odds_captured INTEGER DEFAULT 0,
    bookmaker VARCHAR(50),
    status VARCHAR(20) DEFAULT 'success',  -- success, partial, failed
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- Indexes for performance
-- ==========================================
CREATE INDEX IF NOT EXISTS idx_game_odds_date ON nba_game_odds(game_date);
CREATE INDEX IF NOT EXISTS idx_game_odds_game ON nba_game_odds(game_id);
CREATE INDEX IF NOT EXISTS idx_game_odds_snapshot ON nba_game_odds(snapshot_time);
CREATE INDEX IF NOT EXISTS idx_game_odds_teams ON nba_game_odds(home_team_abbrev, away_team_abbrev);

CREATE INDEX IF NOT EXISTS idx_player_prop_date ON nba_player_prop_odds(game_date);
CREATE INDEX IF NOT EXISTS idx_player_prop_game ON nba_player_prop_odds(game_id);
CREATE INDEX IF NOT EXISTS idx_player_prop_player ON nba_player_prop_odds(player_id);
CREATE INDEX IF NOT EXISTS idx_player_prop_player_name ON nba_player_prop_odds(player_name);
CREATE INDEX IF NOT EXISTS idx_player_prop_type ON nba_player_prop_odds(prop_type);
CREATE INDEX IF NOT EXISTS idx_player_prop_snapshot ON nba_player_prop_odds(snapshot_time);

CREATE INDEX IF NOT EXISTS idx_snapshots_date ON nba_odds_snapshots(game_date);

-- ==========================================
-- Views for easy querying
-- ==========================================

-- Latest game odds per game/bookmaker
CREATE OR REPLACE VIEW v_latest_game_odds AS
SELECT DISTINCT ON (game_id, bookmaker)
    *
FROM nba_game_odds
WHERE is_pregame = TRUE
ORDER BY game_id, bookmaker, snapshot_time DESC;

-- Latest player props per player/game/prop
CREATE OR REPLACE VIEW v_latest_player_props AS
SELECT DISTINCT ON (game_id, player_name, prop_type, bookmaker)
    *
FROM nba_player_prop_odds
WHERE is_pregame = TRUE
ORDER BY game_id, player_name, prop_type, bookmaker, snapshot_time DESC;

-- Odds movement tracking (compare first vs last snapshot)
CREATE OR REPLACE VIEW v_odds_movement AS
WITH first_odds AS (
    SELECT DISTINCT ON (game_id, player_name, prop_type)
        game_id, player_name, prop_type, 
        line as opening_line,
        over_odds as opening_over_odds,
        snapshot_time as first_snapshot
    FROM nba_player_prop_odds
    WHERE is_pregame = TRUE
    ORDER BY game_id, player_name, prop_type, snapshot_time ASC
),
last_odds AS (
    SELECT DISTINCT ON (game_id, player_name, prop_type)
        game_id, player_name, prop_type,
        line as closing_line,
        over_odds as closing_over_odds,
        snapshot_time as last_snapshot
    FROM nba_player_prop_odds
    WHERE is_pregame = TRUE
    ORDER BY game_id, player_name, prop_type, snapshot_time DESC
)
SELECT 
    f.game_id,
    f.player_name,
    f.prop_type,
    f.opening_line,
    l.closing_line,
    l.closing_line - f.opening_line as line_movement,
    f.opening_over_odds,
    l.closing_over_odds,
    f.first_snapshot,
    l.last_snapshot
FROM first_odds f
JOIN last_odds l ON f.game_id = l.game_id 
    AND f.player_name = l.player_name 
    AND f.prop_type = l.prop_type;