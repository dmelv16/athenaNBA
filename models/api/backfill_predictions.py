"""
Backfill NBA Predictions
Generate predictions for historical games using database game logs

Usage:
    python -m models.api.backfill_predictions --start 2024-10-22 --end 2025-12-01
    python -m models.api.backfill_predictions --days 30
"""

import argparse
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Set, Tuple
import time
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from etl.database.connection import get_db_connection
from models.data.data_loader import NBADataLoader
from models.features.player_features import PlayerFeatureEngineer
from models.features.team_features import TeamFeatureEngineer
from models.trainers.player_props_trainer import PlayerPropsTrainer
from models.trainers.team_props_trainer import TeamPropsTrainer
from models.predictors.game_predictor import GamePredictor, GameInfo, PlayerInfo
from models.predictors.parlay_builder import ParlayBuilder
from models.api.db_uploader import NBAPredictionUploader
from models.config import FEATURE_CONFIG


class HistoricalPredictionRunner:
    """Generate predictions using historical data from database"""
    
    def __init__(self):
        self.db = get_db_connection()
        self.data_loader = None
        self.player_fe = None
        self.team_fe = None
        self.player_trainer = None
        self.team_trainer = None
        self.predictor = None
        self.parlay_builder = None
        self.uploader = None
        
        # Caches
        self.team_id_to_abbrev: Dict[int, str] = {}
        self.abbrev_to_team_id: Dict[str, int] = {}
        self.players_with_data: Set[int] = set()
    
    def initialize(self):
        """Load data and models"""
        print("\n" + "=" * 60)
        print("INITIALIZING HISTORICAL PREDICTION SYSTEM")
        print("=" * 60)
        
        # Load data
        print("\n📊 Loading data from database...")
        self.data_loader = NBADataLoader(min_date='2023-01-01')
        player_logs, team_logs, players, teams_df = self.data_loader.load_all_data()
        print(f"  Player logs: {len(player_logs):,}")
        print(f"  Team logs: {len(team_logs):,}")
        
        # Build caches
        self.players_with_data = set(player_logs['player_id'].unique())
        print(f"  Players with data: {len(self.players_with_data)}")
        
        # Team mappings
        for _, row in teams_df.iterrows():
            self.team_id_to_abbrev[row['team_id']] = row['abbreviation']
            self.abbrev_to_team_id[row['abbreviation']] = row['team_id']
        
        # Initialize feature engineers
        print("\n🔧 Initializing feature engineers...")
        self.player_fe = PlayerFeatureEngineer(player_logs, team_logs, players, teams_df)
        self.team_fe = TeamFeatureEngineer(team_logs, teams_df)
        
        # Load trained models
        print("\n🤖 Loading trained models...")
        self.player_trainer = PlayerPropsTrainer(self.player_fe)
        self.team_trainer = TeamPropsTrainer(self.team_fe)
        
        try:
            self.player_trainer.load_all_models()
            self.team_trainer.load_all_models()
            print("  ✓ Models loaded successfully")
            
            # Show what models we have
            print(f"\n  Player prop models: {list(self.player_trainer.models.keys())}")
            print(f"  Team prop models: {list(self.team_trainer.models.keys())}")
            
        except Exception as e:
            print(f"  ✗ Error loading models: {e}")
            print("  → Training new models...")
            self.player_trainer.train_all_models(verbose=False)
            self.team_trainer.train_all_models(verbose=False)
            self.player_trainer.save_all_models()
            self.team_trainer.save_all_models()
        
        # Initialize predictor
        self.predictor = GamePredictor(
            self.player_fe, self.team_fe,
            self.player_trainer, self.team_trainer
        )
        
        # Initialize parlay builder
        self.parlay_builder = ParlayBuilder()
        
        # Initialize database uploader
        print("\n💾 Connecting to prediction database...")
        self.uploader = NBAPredictionUploader()
        self.uploader.init_schema()
        
        print("\n✓ Initialization complete!")
    
    def get_games_for_date(self, target_date: date) -> List[Dict]:
        """
        Get games from team_game_logs for a specific date
        Returns unique games with home/away teams identified
        """
        query = """
            SELECT DISTINCT 
                game_id,
                game_date,
                team_id,
                matchup,
                pts,
                wl
            FROM team_game_logs
            WHERE game_date = %s
            ORDER BY game_id, team_id
        """
        
        with self.db.get_cursor() as cur:
            cur.execute(query, (target_date,))
            columns = [desc[0] for desc in cur.description]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        # Group by game_id to pair home/away teams
        games_dict: Dict[str, Dict] = {}
        
        for row in results:
            game_id = row['game_id']
            matchup = row['matchup'] or ''
            
            # Determine if home or away from matchup string
            is_home = ' vs. ' in matchup
            
            if game_id not in games_dict:
                games_dict[game_id] = {
                    'game_id': game_id,
                    'game_date': row['game_date'],
                    'home_team_id': None,
                    'away_team_id': None,
                    'home_team_abbrev': None,
                    'away_team_abbrev': None,
                    'home_pts': None,
                    'away_pts': None
                }
            
            team_abbrev = self.team_id_to_abbrev.get(row['team_id'], 'UNK')
            
            if is_home:
                games_dict[game_id]['home_team_id'] = row['team_id']
                games_dict[game_id]['home_team_abbrev'] = team_abbrev
                games_dict[game_id]['home_pts'] = row['pts']
            else:
                games_dict[game_id]['away_team_id'] = row['team_id']
                games_dict[game_id]['away_team_abbrev'] = team_abbrev
                games_dict[game_id]['away_pts'] = row['pts']
        
        # Filter to complete games only
        games = [g for g in games_dict.values() 
                 if g['home_team_id'] and g['away_team_id']]
        
        return games
    
    def get_players_for_game(self, game: Dict) -> List[PlayerInfo]:
        """
        Get players who played in this game from player_game_logs
        """
        query = """
            SELECT DISTINCT player_id, matchup
            FROM player_game_logs
            WHERE game_id = %s
        """
        
        with self.db.get_cursor() as cur:
            cur.execute(query, (game['game_id'],))
            results = cur.fetchall()
        
        players = []
        
        for player_id, matchup in results:
            if player_id not in self.players_with_data:
                continue
            
            # Get player name
            player_query = "SELECT full_name FROM players WHERE player_id = %s"
            with self.db.get_cursor() as cur:
                cur.execute(player_query, (player_id,))
                result = cur.fetchone()
                player_name = result[0] if result else f"Player {player_id}"
            
            # Determine team from matchup
            matchup = matchup or ''
            is_home = ' vs. ' in matchup
            
            if is_home:
                team_id = game['home_team_id']
                team_abbrev = game['home_team_abbrev']
                opponent_abbrev = game['away_team_abbrev']
            else:
                team_id = game['away_team_id']
                team_abbrev = game['away_team_abbrev']
                opponent_abbrev = game['home_team_abbrev']
            
            players.append(PlayerInfo(
                player_id=player_id,
                player_name=player_name,
                team_id=team_id,
                team_abbrev=team_abbrev,
                opponent_abbrev=opponent_abbrev,
                is_home=is_home,
                rest_days=2,
                lines=None
            ))
        
        return players
    
    def generate_predictions(self, target_date: date) -> Dict:
        """Generate all predictions for a date using DB data"""
        
        # Get games from database
        games = self.get_games_for_date(target_date)
        
        if not games:
            return {'games': [], 'player_predictions': [], 'team_predictions': [], 'parlays': []}
        
        all_player_predictions = []
        all_team_predictions = []
        all_prop_predictions = []
        
        for game in games:
            print(f"\n  🏀 {game['away_team_abbrev']} @ {game['home_team_abbrev']}")
            
            # Create GameInfo
            game_info = GameInfo(
                game_id=game['game_id'],
                game_date=datetime.combine(game['game_date'], datetime.min.time()),
                home_team_id=game['home_team_id'],
                home_team_name=game['home_team_abbrev'],
                home_team_abbrev=game['home_team_abbrev'],
                away_team_id=game['away_team_id'],
                away_team_name=game['away_team_abbrev'],
                away_team_abbrev=game['away_team_abbrev']
            )
            
            # Team predictions (spread + total)
            try:
                team_preds = self.predictor.predict_team_props(game_info)
                
                for pred in team_preds:
                    pred_dict = {
                        'game_id': game['game_id'],
                        'game_date': target_date,
                        'home_team_id': game['home_team_id'],
                        'home_team_abbrev': game['home_team_abbrev'],
                        'away_team_id': game['away_team_id'],
                        'away_team_abbrev': game['away_team_abbrev'],
                        'prop_type': pred.prop_type,
                        'predicted_value': pred.prediction.pred_value,
                        'confidence': pred.prediction.confidence,
                        'lower_bound': pred.prediction.lower_bound,
                        'upper_bound': pred.prediction.upper_bound,
                        'line': pred.line,
                        'edge': pred.edge,
                        'recommended_bet': pred.recommended_bet
                    }
                    all_team_predictions.append(pred_dict)
                    all_prop_predictions.append(pred)
                
                print(f"    Team props: {len(team_preds)} predictions")
            except Exception as e:
                print(f"    ⚠️ Team prediction error: {e}")
            
            # Player predictions
            players = self.get_players_for_game(game)
            player_count = 0
            
            for player in players:
                try:
                    player_preds = self.predictor.predict_player_props(player, game_info)
                    
                    for pred in player_preds:
                        # Skip low-accuracy props
                        if pred.prop_type in ['stl', 'blk']:
                            continue
                        
                        pred_dict = {
                            'game_id': game['game_id'],
                            'game_date': target_date,
                            'player_id': player.player_id,
                            'player_name': player.player_name,
                            'team_abbrev': player.team_abbrev,
                            'opponent_abbrev': player.opponent_abbrev,
                            'is_home': player.is_home,
                            'prop_type': pred.prop_type,
                            'predicted_value': pred.prediction.pred_value,
                            'confidence': pred.prediction.confidence,
                            'lower_bound': pred.prediction.lower_bound,
                            'upper_bound': pred.prediction.upper_bound,
                            'line': pred.line,
                            'edge': pred.edge,
                            'recommended_bet': pred.recommended_bet
                        }
                        all_player_predictions.append(pred_dict)
                        all_prop_predictions.append(pred)
                    
                    if player_preds:
                        player_count += 1
                        
                except Exception:
                    continue
            
            print(f"    Player props: {player_count} players predicted")
        
        # Build parlays
        parlay_dicts = []
        if all_prop_predictions:
            try:
                daily_parlays = self.parlay_builder.build_daily_parlays(
                    all_prop_predictions,
                    num_standard=5,
                    num_sgp_per_game=2,
                    risk_level='moderate'
                )
                
                for candidate in daily_parlays.get('standard', []):
                    parlay_dicts.append(candidate.parlay.to_dict())
                    parlay_dicts[-1]['score'] = candidate.score
                    parlay_dicts[-1]['parlay_type'] = 'standard'
                
                for game_id, sgps in daily_parlays.get('same_game', {}).items():
                    for candidate in sgps:
                        parlay_dict = candidate.parlay.to_dict()
                        parlay_dict['score'] = candidate.score
                        parlay_dict['parlay_type'] = 'same_game'
                        parlay_dict['game_id'] = game_id
                        parlay_dicts.append(parlay_dict)
            except Exception as e:
                print(f"    ⚠️ Parlay building error: {e}")
        
        return {
            'games': games,
            'player_predictions': all_player_predictions,
            'team_predictions': all_team_predictions,
            'parlays': parlay_dicts
        }
    
    def upload_predictions(self, predictions: Dict, target_date: date):
        """Upload predictions to database"""
        player_count = self.uploader.upload_player_predictions(
            predictions['player_predictions'],
            prediction_date=target_date
        )
        
        team_count = self.uploader.upload_team_predictions(
            predictions['team_predictions'],
            prediction_date=target_date
        )
        
        parlay_count = self.uploader.upload_parlays(
            predictions['parlays'],
            prediction_date=target_date
        )
        
        return player_count, team_count, parlay_count
    
    def close(self):
        """Cleanup connections"""
        if self.uploader:
            self.uploader.close()
        if self.data_loader:
            self.data_loader.close()
        if self.db:
            self.db.close()


def backfill_predictions(
    start_date: date,
    end_date: date,
    skip_existing: bool = True
):
    """
    Generate predictions for a date range using database game logs
    """
    print("\n" + "=" * 60)
    print("🏀 NBA PREDICTION BACKFILL (DATABASE-BASED)")
    print("=" * 60)
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    
    total_days = (end_date - start_date).days
    print(f"Total Days: {total_days}")
    
    # Initialize runner once
    runner = HistoricalPredictionRunner()
    runner.initialize()
    
    # Track stats
    processed = 0
    skipped = 0
    errors = 0
    total_player_preds = 0
    total_team_preds = 0
    total_parlays = 0
    
    current_date = start_date
    
    while current_date < end_date:
        print(f"\n{'='*60}")
        print(f"📅 Processing: {current_date} ({processed + skipped + errors + 1}/{total_days})")
        print(f"{'='*60}")
        
        try:
            # Check if predictions already exist
            if skip_existing:
                existing = runner.uploader.get_predictions_by_date(current_date)
                if existing.get('player_predictions'):
                    print(f"  ⏭️ Skipping - {len(existing['player_predictions'])} predictions exist")
                    skipped += 1
                    current_date += timedelta(days=1)
                    continue
            
            # Generate predictions
            predictions = runner.generate_predictions(current_date)
            
            if predictions['player_predictions'] or predictions['team_predictions']:
                # Upload to database
                p_count, t_count, par_count = runner.upload_predictions(predictions, current_date)
                
                total_player_preds += p_count
                total_team_preds += t_count
                total_parlays += par_count
                processed += 1
                
                print(f"\n  ✓ Uploaded: {p_count} player, {t_count} team, {par_count} parlays")
            else:
                print(f"  ⚠️ No games found in database for this date")
                skipped += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
        
        current_date += timedelta(days=1)
    
    # Summary
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"Days Processed: {processed}")
    print(f"Days Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"\nTotal Predictions:")
    print(f"  Player props: {total_player_preds}")
    print(f"  Team props: {total_team_preds}")
    print(f"  Parlays: {total_parlays}")
    print("=" * 60)
    
    runner.close()


def main():
    parser = argparse.ArgumentParser(
        description='Backfill NBA Predictions (Database-Based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Backfill from season start to yesterday
    python -m models.api.backfill_predictions --start 2024-10-22 --end 2025-12-02
    
    # Backfill last 30 days
    python -m models.api.backfill_predictions --days 30
    
    # Backfill specific range, overwrite existing
    python -m models.api.backfill_predictions --start 2024-11-01 --end 2024-11-15 --overwrite
        """
    )
    
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD). Defaults to today.')
    parser.add_argument('--days', type=int, help='Number of days to backfill (alternative to --start/--end)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing predictions')
    
    args = parser.parse_args()
    
    # Determine date range
    if args.days:
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days)
    elif args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date() if args.end else date.today()
    else:
        print("Error: Must specify either --start or --days")
        sys.exit(1)
    
    if start_date >= end_date:
        print(f"Error: Start date ({start_date}) must be before end date ({end_date})")
        sys.exit(1)
    
    if end_date > date.today():
        end_date = date.today()
    
    backfill_predictions(start_date, end_date, skip_existing=not args.overwrite)


if __name__ == "__main__":
    main()