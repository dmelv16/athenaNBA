"""
Parlay builder - creates optimal parlays from predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from itertools import combinations
from dataclasses import dataclass

from models.base import PropPrediction, ParlayLeg, Parlay
from models.config import PARLAY_CONFIG


@dataclass
class ParlayCandidate:
    """A potential parlay with scoring"""
    parlay: Parlay
    score: float
    reasoning: str


class ParlayBuilder:
    """Build optimal parlays from predictions"""
    
    def __init__(self, config: PARLAY_CONFIG = None):
        self.config = config or PARLAY_CONFIG
    
    def filter_bettable_predictions(
        self,
        predictions: List[PropPrediction],
        min_confidence: float = None,
        min_edge: float = None,
        require_line: bool = False
    ) -> List[PropPrediction]:
        """
        Filter predictions to those worth betting
        
        Args:
            predictions: All predictions
            min_confidence: Minimum confidence (default from config)
            min_edge: Minimum edge (default from config)
            require_line: Whether line must be present
            
        Returns:
            Filtered predictions
        """
        min_confidence = min_confidence or self.config.min_confidence
        min_edge = min_edge or self.config.min_edge
        
        # Get excluded props from config
        excluded_props = getattr(self.config, 'excluded_props', ['stl', 'blk'])
        
        filtered = []
        
        for pred in predictions:
            # Skip excluded prop types (low accuracy)
            if pred.prop_type in excluded_props:
                continue
            
            # Confidence check
            if pred.prediction.confidence < min_confidence:
                continue
            
            # Edge check (only if line exists)
            if pred.line is not None:
                if pred.edge is None or abs(pred.edge) < min_edge:
                    continue
            elif require_line:
                continue
            
            filtered.append(pred)
        
        return filtered
    
    def score_prediction(self, pred: PropPrediction) -> float:
        """
        Score a prediction for parlay inclusion
        
        Higher score = better candidate
        """
        score = 0.0
        
        # Base confidence score (0-40 points)
        score += pred.prediction.confidence * 40
        
        # Edge score (0-30 points)
        if pred.edge is not None:
            edge_score = min(abs(pred.edge) / 5, 1) * 30
            score += edge_score
        
        # Prop type bonuses (some props are more predictable)
        prop_bonuses = {
            'pts': 5,     # Points are relatively predictable
            'pra': 5,     # Combos smooth variance
            'reb': 3,
            'ast': 3,
            'pr': 4,
            'pa': 4,
            'total': 5,   # Game totals
            'spread': 3,
        }
        score += prop_bonuses.get(pred.prop_type, 0)
        
        # Consistency bonus (lower variance in prediction)
        pred_range = pred.prediction.upper_bound - pred.prediction.lower_bound
        if pred_range > 0:
            consistency = max(0, 10 - pred_range)
            score += consistency
        
        return score
    
    def create_parlay_leg(self, pred: PropPrediction) -> ParlayLeg:
        """Create a parlay leg from a prediction"""
        bet_type = pred.recommended_bet or ('over' if pred.edge and pred.edge > 0 else 'under')
        
        return ParlayLeg(
            prop=pred,
            bet_type=bet_type,
            odds=-110  # Default odds
        )
    
    def check_correlation(
        self,
        leg1: ParlayLeg,
        leg2: ParlayLeg
    ) -> Tuple[bool, float]:
        """
        Check if two legs are correlated
        
        Returns:
            (is_correlated, correlation_strength)
        """
        p1, p2 = leg1.prop, leg2.prop
        
        # Same player - highly correlated
        if p1.player_id and p1.player_id == p2.player_id:
            # Same player different props
            corr_map = {
                ('pts', 'pra'): 0.9,
                ('pts', 'pr'): 0.8,
                ('pts', 'pa'): 0.8,
                ('reb', 'pra'): 0.7,
                ('reb', 'pr'): 0.8,
                ('ast', 'pra'): 0.7,
                ('ast', 'pa'): 0.8,
            }
            
            key = tuple(sorted([p1.prop_type, p2.prop_type]))
            corr = corr_map.get(key, 0.3)
            return True, corr
        
        # Same game - some correlation
        if p1.game_id == p2.game_id:
            # Team total and player points
            if p1.prop_type == 'total' and p2.prop_type == 'pts':
                return True, 0.3
            if p1.prop_type == 'spread' and p2.prop_type in ['pts', 'pra']:
                return True, 0.2
            return True, 0.1
        
        # Different games - independent
        return False, 0.0
    
    def calculate_parlay_score(self, parlay: Parlay) -> float:
        """Calculate overall parlay quality score"""
        score = 0.0
        
        # Average confidence (0-30)
        score += parlay.avg_confidence * 30
        
        # Minimum confidence bonus (0-20)
        score += parlay.min_confidence * 20
        
        # Expected value (0-25)
        ev = max(-0.5, min(parlay.expected_value, 0.5))
        score += (ev + 0.5) * 25
        
        # Leg count penalty (fewer is better for probability)
        leg_penalty = max(0, parlay.num_legs - 3) * 5
        score -= leg_penalty
        
        # Correlation penalty
        corr_penalty = 0
        for i, leg1 in enumerate(parlay.legs):
            for leg2 in parlay.legs[i+1:]:
                is_corr, strength = self.check_correlation(leg1, leg2)
                if is_corr:
                    corr_penalty += strength * 10
        score -= corr_penalty
        
        # Diversification bonus (different games)
        num_games = len(parlay.games_involved)
        if num_games > 1:
            score += min(num_games, 4) * 3
        
        return max(0, score)
    
    def build_best_parlays(
        self,
        predictions: List[PropPrediction],
        num_parlays: int = 5,
        min_legs: int = None,
        max_legs: int = None,
        risk_level: str = 'moderate'
    ) -> List[ParlayCandidate]:
        """
        Build the best parlays from predictions
        
        Args:
            predictions: All predictions
            num_parlays: Number of parlays to return
            min_legs: Minimum legs per parlay
            max_legs: Maximum legs per parlay
            risk_level: 'conservative', 'moderate', or 'aggressive'
            
        Returns:
            List of ParlayCandidate objects
        """
        min_legs = min_legs or self.config.min_legs
        max_legs = max_legs or self.config.max_legs
        
        # Set confidence threshold based on risk level
        conf_thresholds = {
            'conservative': self.config.conservative_confidence,
            'moderate': self.config.moderate_confidence,
            'aggressive': self.config.aggressive_confidence
        }
        min_conf = conf_thresholds.get(risk_level, self.config.moderate_confidence)
        
        # Filter predictions
        filtered = self.filter_bettable_predictions(predictions, min_confidence=min_conf)
        
        if len(filtered) < min_legs:
            return []
        
        # Score all predictions
        scored = [(pred, self.score_prediction(pred)) for pred in filtered]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Take top candidates
        top_preds = [pred for pred, _ in scored[:20]]
        
        # Generate parlay combinations
        all_parlays = []
        
        for num_legs in range(min_legs, min(max_legs + 1, len(top_preds) + 1)):
            for combo in combinations(top_preds, num_legs):
                legs = [self.create_parlay_leg(pred) for pred in combo]
                parlay = Parlay(legs=legs)
                score = self.calculate_parlay_score(parlay)
                
                # Create reasoning
                reasons = []
                if parlay.avg_confidence > 0.65:
                    reasons.append("High average confidence")
                if parlay.avg_edge > 2:
                    reasons.append(f"Strong avg edge ({parlay.avg_edge:.1f})")
                if not parlay.is_same_game:
                    reasons.append("Diversified across games")
                
                candidate = ParlayCandidate(
                    parlay=parlay,
                    score=score,
                    reasoning="; ".join(reasons) if reasons else "Standard parlay"
                )
                all_parlays.append(candidate)
        
        # Sort by score and return top
        all_parlays.sort(key=lambda x: x.score, reverse=True)
        
        return all_parlays[:num_parlays]
    
    def build_same_game_parlays(
        self,
        predictions: List[PropPrediction],
        game_id: str,
        num_parlays: int = 3
    ) -> List[ParlayCandidate]:
        """
        Build same-game parlays for a specific game
        
        Args:
            predictions: All predictions
            game_id: Game to build parlays for
            num_parlays: Number of parlays to return
            
        Returns:
            List of ParlayCandidate objects
        """
        # Filter to this game
        game_preds = [p for p in predictions if p.game_id == game_id]
        
        # Be more lenient with SGPs
        filtered = self.filter_bettable_predictions(
            game_preds,
            min_confidence=self.config.aggressive_confidence,
            min_edge=1.0
        )
        
        if len(filtered) < 2:
            return []
        
        # Score and sort
        scored = [(pred, self.score_prediction(pred)) for pred in filtered]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_preds = [pred for pred, _ in scored[:10]]
        
        # Generate combinations
        all_parlays = []
        max_sgp_legs = min(self.config.max_same_game_legs, len(top_preds))
        
        for num_legs in range(2, max_sgp_legs + 1):
            for combo in combinations(top_preds, num_legs):
                # Check for duplicate players (want variety)
                player_ids = [p.player_id for p in combo if p.player_id]
                if len(player_ids) != len(set(player_ids)):
                    continue  # Skip duplicate players
                
                legs = [self.create_parlay_leg(pred) for pred in combo]
                parlay = Parlay(legs=legs, parlay_type='same_game')
                score = self.calculate_parlay_score(parlay)
                
                candidate = ParlayCandidate(
                    parlay=parlay,
                    score=score,
                    reasoning="Same game parlay"
                )
                all_parlays.append(candidate)
        
        all_parlays.sort(key=lambda x: x.score, reverse=True)
        return all_parlays[:num_parlays]
    
    def build_daily_parlays(
        self,
        predictions: List[PropPrediction],
        num_standard: int = 5,
        num_sgp_per_game: int = 2,
        risk_level: str = 'moderate'
    ) -> Dict[str, List[ParlayCandidate]]:
        """
        Build complete set of parlays for a day
        
        Args:
            predictions: All predictions for the day
            num_standard: Number of standard parlays
            num_sgp_per_game: Number of SGPs per game
            risk_level: Risk tolerance
            
        Returns:
            Dictionary with 'standard' and 'same_game' parlays
        """
        result = {
            'standard': [],
            'same_game': {},
            'best_bets': []
        }
        
        # Standard parlays
        result['standard'] = self.build_best_parlays(
            predictions,
            num_parlays=num_standard,
            risk_level=risk_level
        )
        
        # Same game parlays for each game
        game_ids = set(p.game_id for p in predictions)
        for game_id in game_ids:
            sgps = self.build_same_game_parlays(
                predictions,
                game_id,
                num_parlays=num_sgp_per_game
            )
            if sgps:
                result['same_game'][game_id] = sgps
        
        # Best single bets (high confidence)
        best_singles = self.filter_bettable_predictions(
            predictions,
            min_confidence=self.config.high_confidence,
            min_edge=self.config.strong_edge
        )
        best_singles.sort(key=lambda x: x.prediction.confidence, reverse=True)
        result['best_bets'] = best_singles[:10]
        
        return result
    
    def format_parlay_output(self, candidate: ParlayCandidate) -> str:
        """Format a parlay for display"""
        p = candidate.parlay
        
        lines = [
            f"{'='*50}",
            f"PARLAY ({p.num_legs} legs) - Score: {candidate.score:.1f}",
            f"{'='*50}",
            f"Combined Odds: {p.american_odds:+.0f} ({p.combined_odds:.2f}x)",
            f"Avg Confidence: {p.avg_confidence:.1%}",
            f"Expected Value: {p.expected_value:+.1%}",
            f"Reasoning: {candidate.reasoning}",
            f"{'-'*50}",
        ]
        
        for i, leg in enumerate(p.legs, 1):
            prop = leg.prop
            if prop.player_name:
                desc = f"{prop.player_name} ({prop.team_name})"
            else:
                desc = prop.team_name
            
            line_str = f"@ {prop.line}" if prop.line else ""
            edge_str = f"Edge: {prop.edge:+.1f}" if prop.edge else ""
            
            lines.append(
                f"  {i}. {desc}: {prop.prop_type.upper()} {leg.bet_type.upper()} "
                f"{prop.prediction.pred_value:.1f} {line_str} "
                f"[{prop.prediction.confidence:.0%}] {edge_str}"
            )
        
        lines.append(f"{'='*50}")
        return "\n".join(lines)
    
    def generate_daily_report(
        self,
        daily_parlays: Dict[str, List[ParlayCandidate]],
        output_path: str = None
    ) -> str:
        """
        Generate a complete daily betting report
        
        Args:
            daily_parlays: Output from build_daily_parlays
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report_lines = [
            "=" * 60,
            f"NBA BETTING REPORT - {datetime.now().strftime('%Y-%m-%d')}",
            "=" * 60,
            "",
        ]
        
        # Best single bets
        report_lines.append("🎯 TOP SINGLE BETS")
        report_lines.append("-" * 40)
        
        for pred in daily_parlays.get('best_bets', [])[:5]:
            name = pred.player_name or pred.team_name
            report_lines.append(
                f"  • {name}: {pred.prop_type.upper()} {pred.recommended_bet} "
                f"{pred.prediction.pred_value:.1f} [{pred.prediction.confidence:.0%}]"
            )
        
        report_lines.append("")
        
        # Standard parlays
        report_lines.append("📊 RECOMMENDED PARLAYS")
        for candidate in daily_parlays.get('standard', [])[:3]:
            report_lines.append(self.format_parlay_output(candidate))
            report_lines.append("")
        
        # Same game parlays
        report_lines.append("🎮 SAME GAME PARLAYS")
        for game_id, sgps in daily_parlays.get('same_game', {}).items():
            if sgps:
                report_lines.append(f"\nGame: {game_id}")
                report_lines.append(self.format_parlay_output(sgps[0]))
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report