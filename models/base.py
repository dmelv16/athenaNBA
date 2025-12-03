"""
Base classes and data structures for prediction models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class Prediction:
    """Single prediction result"""
    pred_value: float
    confidence: float
    lower_bound: float  # 25th percentile
    upper_bound: float  # 75th percentile
    features_used: int
    model_name: str


@dataclass
class PropPrediction:
    """Prediction for a specific prop bet"""
    player_id: Optional[int]
    player_name: Optional[str]
    team_id: Optional[int]
    team_name: Optional[str]
    game_id: str
    game_date: datetime
    opponent: str
    
    prop_type: str  # 'pts', 'reb', 'ast', 'spread', 'total', etc.
    prediction: Prediction
    
    # Line comparison (if available)
    line: Optional[float] = None
    edge: Optional[float] = None  # prediction - line
    recommended_bet: Optional[str] = None  # 'over', 'under', 'home', 'away'
    
    def __post_init__(self):
        if self.line is not None:
            self.edge = self.prediction.pred_value - self.line
            if self.prop_type in ['spread']:
                self.recommended_bet = 'home' if self.edge > 0 else 'away'
            else:
                self.recommended_bet = 'over' if self.edge > 0 else 'under'


@dataclass 
class ParlayLeg:
    """Single leg of a parlay"""
    prop: PropPrediction
    bet_type: str  # 'over', 'under', 'home', 'away'
    odds: float = -110  # Default American odds
    
    @property
    def implied_probability(self) -> float:
        """Convert American odds to implied probability"""
        if self.odds < 0:
            return abs(self.odds) / (abs(self.odds) + 100)
        return 100 / (self.odds + 100)
    
    @property
    def decimal_odds(self) -> float:
        """Convert to decimal odds"""
        if self.odds < 0:
            return 1 + (100 / abs(self.odds))
        return 1 + (self.odds / 100)


@dataclass
class Parlay:
    """Multi-leg parlay bet"""
    legs: List[ParlayLeg]
    created_at: datetime = field(default_factory=datetime.now)
    parlay_type: str = "standard"  # 'standard', 'same_game', 'round_robin'
    
    @property
    def num_legs(self) -> int:
        return len(self.legs)
    
    @property
    def combined_odds(self) -> float:
        """Calculate combined decimal odds"""
        odds = 1.0
        for leg in self.legs:
            odds *= leg.decimal_odds
        return odds
    
    @property
    def american_odds(self) -> float:
        """Convert combined odds to American"""
        decimal = self.combined_odds
        if decimal >= 2.0:
            return (decimal - 1) * 100
        return -100 / (decimal - 1)
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence across legs"""
        return np.mean([leg.prop.prediction.confidence for leg in self.legs])
    
    @property
    def min_confidence(self) -> float:
        """Minimum confidence across legs"""
        return min(leg.prop.prediction.confidence for leg in self.legs)
    
    @property
    def avg_edge(self) -> float:
        """Average edge across legs"""
        edges = [leg.prop.edge for leg in self.legs if leg.prop.edge is not None]
        return np.mean(edges) if edges else 0
    
    @property
    def expected_value(self) -> float:
        """Rough EV calculation (simplified)"""
        # Product of individual win probabilities * payout - 1
        win_prob = 1.0
        for leg in self.legs:
            # Use confidence as rough win probability
            win_prob *= leg.prop.prediction.confidence
        return (win_prob * self.combined_odds) - 1
    
    @property
    def games_involved(self) -> List[str]:
        """List of unique game IDs"""
        return list(set(leg.prop.game_id for leg in self.legs))
    
    @property
    def is_same_game(self) -> bool:
        """Check if all legs are from same game"""
        return len(self.games_involved) == 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'parlay_type': self.parlay_type,
            'num_legs': self.num_legs,
            'combined_odds': round(self.combined_odds, 2),
            'american_odds': round(self.american_odds, 0),
            'avg_confidence': round(self.avg_confidence, 3),
            'min_confidence': round(self.min_confidence, 3),
            'avg_edge': round(self.avg_edge, 2),
            'expected_value': round(self.expected_value, 3),
            'is_same_game': self.is_same_game,
            'legs': [
                {
                    'player': leg.prop.player_name,
                    'team': leg.prop.team_name,
                    'prop': leg.prop.prop_type,
                    'bet': leg.bet_type,
                    'prediction': round(leg.prop.prediction.pred_value, 1),
                    'line': leg.prop.line,
                    'edge': round(leg.prop.edge, 1) if leg.prop.edge else None,
                    'confidence': round(leg.prop.prediction.confidence, 3)
                }
                for leg in self.legs
            ]
        }


@dataclass
class ModelMetrics:
    """Performance metrics for a trained model"""
    mae: float
    rmse: float
    mape: float  # Mean absolute percentage error
    r2: float
    directional_accuracy: float  # % of over/under correct
    
    # Binned accuracy
    accuracy_by_confidence: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self):
        return (
            f"MAE: {self.mae:.2f} | RMSE: {self.rmse:.2f} | "
            f"MAPE: {self.mape:.1f}% | R²: {self.r2:.3f} | "
            f"Direction: {self.directional_accuracy:.1%}"
        )