"""
Configuration for prediction models
"""

from dataclasses import dataclass, field
from typing import List, Dict
import os


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Rolling window sizes for averages
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    
    # Minimum games required for prediction
    min_games_required: int = 5
    
    # Stats to calculate rolling features for
    player_stats: List[str] = field(default_factory=lambda: [
        'pts', 'reb', 'ast', 'stl', 'blk', 'min', 
        'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
        'oreb', 'dreb', 'tov', 'pf', 'plus_minus'
    ])
    
    team_stats: List[str] = field(default_factory=lambda: [
        'pts', 'reb', 'ast', 'stl', 'blk',
        'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
        'oreb', 'dreb', 'tov', 'pf'
    ])


@dataclass
class ModelConfig:
    """Model training configuration"""
    # XGBoost parameters
    xgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # Test split ratio
    test_ratio: float = 0.15
    
    # Cross-validation folds
    cv_folds: int = 5
    
    # Early stopping rounds
    early_stopping: int = 50


@dataclass
class ParlayConfig:
    """Parlay building configuration"""
    # Confidence thresholds
    min_confidence: float = 0.6  # Minimum confidence for single bet
    high_confidence: float = 0.7  # High confidence threshold
    
    # Edge thresholds (predicted - line)
    min_edge: float = 1.5  # Minimum edge to consider
    strong_edge: float = 3.0  # Strong edge
    
    # Parlay constraints
    max_legs: int = 6
    min_legs: int = 2
    max_same_game_legs: int = 3
    
    # Risk levels
    conservative_confidence: float = 0.65
    moderate_confidence: float = 0.60
    aggressive_confidence: float = 0.55
    
    # Props to USE (high accuracy)
    preferred_props: List[str] = field(default_factory=lambda: [
        'pts', 'reb', 'ast', 'pra', 'pr', 'pa', 'ra', 'spread'
    ])
    
    # Props to AVOID (low accuracy - essentially random)
    excluded_props: List[str] = field(default_factory=lambda: [
        'stl', 'blk'  # Below 50% accuracy - worse than coin flip
    ])


@dataclass
class PathConfig:
    """File paths configuration"""
    models_dir: str = "saved_models"
    predictions_dir: str = "predictions"
    logs_dir: str = "logs"
    
    def __post_init__(self):
        # Create directories if they don't exist
        for path in [self.models_dir, self.predictions_dir, self.logs_dir]:
            os.makedirs(path, exist_ok=True)


# Global config instances
FEATURE_CONFIG = FeatureConfig()
MODEL_CONFIG = ModelConfig()
PARLAY_CONFIG = ParlayConfig()
PATH_CONFIG = PathConfig()