-- NBA Predictions Database Schema
-- Run this to create the predictions tables

-- Player prop predictions
CREATE TABLE IF NOT EXISTS nba_player_predictions (
    id SERIAL PRIMARY KEY,
    prediction_date DATE NOT NULL,
    game_id VARCHAR(20) NOT NULL,
    game_date DATE NOT NULL,
    
    -- Player info
    player_id INTEGER NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    team_abbrev VARCHAR(10),
    opponent_abbrev VARCHAR(10),
    is_home BOOLEAN,
    
    -- Prop info
    prop_type VARCHAR(20) NOT NULL,  -- pts, reb, ast, pra, etc.
    
    -- Prediction
    predicted_value DECIMAL(10,2) NOT NULL,
    confidence DECIMAL(5,4),
    lower_bound DECIMAL(10,2),
    upper_bound DECIMAL(10,2),
    
    -- Line comparison (if available)
    line DECIMAL(10,2),
    edge DECIMAL(10,2),
    recommended_bet VARCHAR(10),  -- over/under
    
    -- Metadata
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(prediction_date, game_id, player_id, prop_type)
);

-- Team prop predictions (spread, totals)
CREATE TABLE IF NOT EXISTS nba_team_predictions (
    id SERIAL PRIMARY KEY,
    prediction_date DATE NOT NULL,
    game_id VARCHAR(20) NOT NULL,
    game_date DATE NOT NULL,
    
    -- Teams
    home_team_id INTEGER NOT NULL,
    home_team_abbrev VARCHAR(10),
    away_team_id INTEGER NOT NULL,
    away_team_abbrev VARCHAR(10),
    
    -- Prop info
    prop_type VARCHAR(20) NOT NULL,  -- spread, total
    
    -- Prediction
    predicted_value DECIMAL(10,2) NOT NULL,
    confidence DECIMAL(5,4),
    lower_bound DECIMAL(10,2),
    upper_bound DECIMAL(10,2),
    
    -- Line comparison
    line DECIMAL(10,2),
    edge DECIMAL(10,2),
    recommended_bet VARCHAR(10),  -- home/away for spread, over/under for total
    
    -- Metadata
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(prediction_date, game_id, prop_type)
);

-- Parlay recommendations
CREATE TABLE IF NOT EXISTS nba_parlay_predictions (
    id SERIAL PRIMARY KEY,
    prediction_date DATE NOT NULL,
    parlay_type VARCHAR(20),  -- standard, same_game
    
    -- Parlay details
    num_legs INTEGER,
    legs JSONB NOT NULL,  -- Array of leg details
    
    -- Metrics
    combined_odds DECIMAL(10,2),
    american_odds INTEGER,
    avg_confidence DECIMAL(5,4),
    min_confidence DECIMAL(5,4),
    avg_edge DECIMAL(10,2),
    expected_value DECIMAL(5,4),
    score DECIMAL(10,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prediction results (for tracking accuracy)
CREATE TABLE IF NOT EXISTS nba_prediction_results (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER,
    prediction_type VARCHAR(20),  -- player, team
    
    -- Result
    actual_value DECIMAL(10,2),
    hit BOOLEAN,  -- Did prediction beat the line?
    
    -- Metadata
    result_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_player_pred_date ON nba_player_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_player_pred_game ON nba_player_predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_player_pred_player ON nba_player_predictions(player_id);
CREATE INDEX IF NOT EXISTS idx_team_pred_date ON nba_team_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_team_pred_game ON nba_team_predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_parlay_date ON nba_parlay_predictions(prediction_date);