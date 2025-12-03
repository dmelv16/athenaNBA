"""
Game prediction orchestrator - predicts all props for a game
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass

from models.base import PropPrediction, Prediction
from models.features.player_features import PlayerFeatureEngineer
from models.features.team_features import TeamFeatureEngineer
from models.trainers.player_props_trainer import PlayerPropsTrainer
from models.trainers.team_props_trainer import TeamPropsTrainer


@dataclass
class GameInfo:
    """Information about an upcoming game"""
    game_id: str
    game_date: datetime
    home_team_id: int
    home_team_name: str
    home_team_abbrev: str
    away_team_id: int
    away_team_name: str
    away_team_abbrev: str
    
    # Optional betting lines
    spread_line: Optional[float] = None  # Home team spread
    total_line: Optional[float] = None   # Over/under line


@dataclass
class PlayerInfo:
    """Information about a player in a game"""
    player_id: int
    player_name: str
    team_id: int
    team_abbrev: str
    opponent_abbrev: str
    is_home: bool
    rest_days: int = 2
    
    # Optional prop lines
    lines: Dict[str, float] = None  # {'pts': 24.5, 'reb': 8.5, etc.}


class GamePredictor:
    """Orchestrates predictions for all props in a game"""
    
    PLAYER_PROPS = ['pts', 'reb', 'ast', 'stl', 'blk', 'pra', 'pr', 'pa', 'ra']
    TEAM_PROPS = ['spread', 'total_pts']
    
    def __init__(
        self,
        player_feature_engineer: PlayerFeatureEngineer,
        team_feature_engineer: TeamFeatureEngineer,
        player_trainer: PlayerPropsTrainer,
        team_trainer: TeamPropsTrainer
    ):
        self.player_fe = player_feature_engineer
        self.team_fe = team_feature_engineer
        self.player_trainer = player_trainer
        self.team_trainer = team_trainer
        
        # Cache for player features
        self._player_feature_cache: Dict[int, pd.DataFrame] = {}
    
    def predict_player_props(
        self,
        player: PlayerInfo,
        game: GameInfo,
        props: List[str] = None
    ) -> List[PropPrediction]:
        """
        Predict all props for a player in a game
        
        Args:
            player: Player information
            game: Game information
            props: List of props to predict (defaults to all)
            
        Returns:
            List of PropPrediction objects
        """
        props = props or self.PLAYER_PROPS
        predictions = []
        
        # Build features for this player
        features = self.player_fe.prepare_prediction_features(
            player_id=player.player_id,
            opponent_abbrev=player.opponent_abbrev,
            is_home=player.is_home,
            rest_days=player.rest_days,
            target_stat='pts'  # Features are same for all stats
        )
        
        if features is None:
            return predictions
        
        for prop in props:
            if prop not in self.player_trainer.models:
                continue
            
            try:
                pred_result = self.player_trainer.predict(prop, features, with_confidence=True)
                if pred_result:
                    pred = pred_result[0]
                    
                    # Get line if available
                    line = player.lines.get(prop) if player.lines else None
                    
                    prop_pred = PropPrediction(
                        player_id=player.player_id,
                        player_name=player.player_name,
                        team_id=player.team_id,
                        team_name=player.team_abbrev,
                        game_id=game.game_id,
                        game_date=game.game_date,
                        opponent=player.opponent_abbrev,
                        prop_type=prop,
                        prediction=pred,
                        line=line
                    )
                    predictions.append(prop_pred)
                    
            except Exception as e:
                print(f"Error predicting {prop} for {player.player_name}: {e}")
        
        return predictions
    
    def predict_team_props(self, game: GameInfo) -> List[PropPrediction]:
        """
        Predict spread and total for a game
        
        Args:
            game: Game information
            
        Returns:
            List of PropPrediction objects
        """
        predictions = []
        
        # Build features for this matchup
        features = self.team_fe.prepare_game_prediction(
            home_team_id=game.home_team_id,
            away_team_id=game.away_team_id,
            game_date=pd.Timestamp(game.game_date)
        )
        
        if features is None:
            return predictions
        
        # Spread prediction
        if 'spread' in self.team_trainer.models:
            try:
                pred_result = self.team_trainer.predict('spread', features, with_confidence=True)
                if pred_result:
                    pred = pred_result[0]
                    
                    prop_pred = PropPrediction(
                        player_id=None,
                        player_name=None,
                        team_id=game.home_team_id,
                        team_name=f"{game.home_team_abbrev} vs {game.away_team_abbrev}",
                        game_id=game.game_id,
                        game_date=game.game_date,
                        opponent=game.away_team_abbrev,
                        prop_type='spread',
                        prediction=pred,
                        line=game.spread_line
                    )
                    predictions.append(prop_pred)
            except Exception as e:
                print(f"Error predicting spread: {e}")
        
        # Total prediction
        if 'total_pts' in self.team_trainer.models:
            try:
                pred_result = self.team_trainer.predict('total_pts', features, with_confidence=True)
                if pred_result:
                    pred = pred_result[0]
                    
                    prop_pred = PropPrediction(
                        player_id=None,
                        player_name=None,
                        team_id=game.home_team_id,
                        team_name=f"{game.home_team_abbrev} vs {game.away_team_abbrev}",
                        game_id=game.game_id,
                        game_date=game.game_date,
                        opponent=game.away_team_abbrev,
                        prop_type='total',
                        prediction=pred,
                        line=game.total_line
                    )
                    predictions.append(prop_pred)
            except Exception as e:
                print(f"Error predicting total: {e}")
        
        return predictions
    
    def predict_full_game(
        self,
        game: GameInfo,
        players: List[PlayerInfo]
    ) -> Dict[str, List[PropPrediction]]:
        """
        Predict all props for a complete game
        
        Args:
            game: Game information
            players: List of players to predict
            
        Returns:
            Dictionary with 'player_props' and 'team_props' keys
        """
        results = {
            'player_props': [],
            'team_props': []
        }
        
        # Team props
        team_preds = self.predict_team_props(game)
        results['team_props'] = team_preds
        
        # Player props for each player
        for player in players:
            player_preds = self.predict_player_props(player, game)
            results['player_props'].extend(player_preds)
        
        return results
    
    def predict_daily_slate(
        self,
        games: List[GameInfo],
        players_by_game: Dict[str, List[PlayerInfo]]
    ) -> Dict[str, Dict]:
        """
        Predict all props for a full day's slate
        
        Args:
            games: List of games
            players_by_game: Dictionary mapping game_id to list of players
            
        Returns:
            Dictionary with predictions for each game
        """
        daily_predictions = {}
        
        for game in games:
            players = players_by_game.get(game.game_id, [])
            game_preds = self.predict_full_game(game, players)
            daily_predictions[game.game_id] = {
                'game': game,
                'predictions': game_preds
            }
        
        return daily_predictions
    
    def get_high_confidence_predictions(
        self,
        predictions: List[PropPrediction],
        min_confidence: float = 0.6,
        min_edge: float = 1.5
    ) -> List[PropPrediction]:
        """
        Filter predictions to high confidence only
        
        Args:
            predictions: List of predictions
            min_confidence: Minimum confidence threshold
            min_edge: Minimum edge (if lines available)
            
        Returns:
            Filtered list of predictions
        """
        filtered = []
        
        for pred in predictions:
            if pred.prediction.confidence < min_confidence:
                continue
            
            if pred.line is not None and pred.edge is not None:
                if abs(pred.edge) < min_edge:
                    continue
            
            filtered.append(pred)
        
        # Sort by confidence
        filtered.sort(key=lambda x: x.prediction.confidence, reverse=True)
        
        return filtered
    
    def format_predictions_table(
        self,
        predictions: List[PropPrediction]
    ) -> pd.DataFrame:
        """Convert predictions to DataFrame for display"""
        rows = []
        
        for pred in predictions:
            row = {
                'Player': pred.player_name or 'Team',
                'Team': pred.team_name,
                'Opponent': pred.opponent,
                'Prop': pred.prop_type,
                'Prediction': round(pred.prediction.pred_value, 1),
                'Confidence': f"{pred.prediction.confidence:.1%}",
                'Line': pred.line,
                'Edge': round(pred.edge, 1) if pred.edge else None,
                'Recommendation': pred.recommended_bet
            }
            rows.append(row)
        
        return pd.DataFrame(rows)