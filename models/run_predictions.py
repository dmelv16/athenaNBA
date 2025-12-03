"""
Main script to run predictions and build parlays
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.database.connection import get_db_connection
from models.config import PATH_CONFIG, PARLAY_CONFIG
from models.features.player_features import PlayerFeatureEngineer
from models.features.team_features import TeamFeatureEngineer
from models.trainers.player_props_trainer import PlayerPropsTrainer
from models.trainers.team_props_trainer import TeamPropsTrainer
from models.predictors.game_predictor import GamePredictor, GameInfo, PlayerInfo
from models.predictors.parlay_builder import ParlayBuilder


def load_data_from_db():
    """Load required data from database"""
    db = get_db_connection()
    
    with db.get_cursor() as cur:
        # Player game logs
        cur.execute("""
            SELECT * FROM player_game_logs 
            WHERE game_date >= '2022-01-01'
            ORDER BY player_id, game_date
        """)
        columns = [desc[0] for desc in cur.description]
        player_logs = pd.DataFrame(cur.fetchall(), columns=columns)
        
        # Team game logs
        cur.execute("""
            SELECT * FROM team_game_logs
            WHERE game_date >= '2022-01-01'
            ORDER BY team_id, game_date
        """)
        columns = [desc[0] for desc in cur.description]
        team_logs = pd.DataFrame(cur.fetchall(), columns=columns)
        
        # Players reference
        cur.execute("SELECT * FROM players WHERE is_active = true")
        columns = [desc[0] for desc in cur.description]
        players = pd.DataFrame(cur.fetchall(), columns=columns)
        
        # Teams reference
        cur.execute("SELECT * FROM teams")
        columns = [desc[0] for desc in cur.description]
        teams = pd.DataFrame(cur.fetchall(), columns=columns)
    
    db.close()
    
    return player_logs, team_logs, players, teams


def train_all_models(player_logs, team_logs, save_models=True):
    """Train all prediction models"""
    print("\n" + "="*60)
    print("INITIALIZING FEATURE ENGINEERS")
    print("="*60)
    
    player_fe = PlayerFeatureEngineer(player_logs, team_logs)
    team_fe = TeamFeatureEngineer(team_logs)
    
    print("\n" + "="*60)
    print("TRAINING PLAYER PROP MODELS")
    print("="*60)
    
    player_trainer = PlayerPropsTrainer(player_fe)
    player_models = player_trainer.train_all_models(verbose=True)
    
    print("\n" + "="*60)
    print("TRAINING TEAM PROP MODELS")
    print("="*60)
    
    team_trainer = TeamPropsTrainer(team_fe)
    team_models = team_trainer.train_all_models(verbose=True)
    
    if save_models:
        print("\nSaving models...")
        player_trainer.save_all_models()
        team_trainer.save_all_models()
    
    return player_fe, team_fe, player_trainer, team_trainer


def create_sample_game_slate():
    """Create sample game slate for demonstration"""
    # In production, this would come from a schedule API or database
    
    games = [
        GameInfo(
            game_id="0022400001",
            game_date=datetime.now(),
            home_team_id=1610612747,  # Lakers
            home_team_name="Los Angeles Lakers",
            home_team_abbrev="LAL",
            away_team_id=1610612744,  # Warriors
            away_team_name="Golden State Warriors",
            away_team_abbrev="GSW",
            spread_line=-3.5,
            total_line=224.5
        ),
        GameInfo(
            game_id="0022400002",
            game_date=datetime.now(),
            home_team_id=1610612738,  # Celtics
            home_team_name="Boston Celtics",
            home_team_abbrev="BOS",
            away_team_id=1610612755,  # 76ers
            away_team_name="Philadelphia 76ers",
            away_team_abbrev="PHI",
            spread_line=-5.5,
            total_line=218.0
        ),
    ]
    
    # Sample players for each game
    players_by_game = {
        "0022400001": [
            PlayerInfo(
                player_id=2544,  # LeBron
                player_name="LeBron James",
                team_id=1610612747,
                team_abbrev="LAL",
                opponent_abbrev="GSW",
                is_home=True,
                rest_days=2,
                lines={'pts': 24.5, 'reb': 7.5, 'ast': 7.5, 'pra': 39.5}
            ),
            PlayerInfo(
                player_id=201142,  # Durant (example)
                player_name="Kevin Durant",
                team_id=1610612744,
                team_abbrev="GSW",
                opponent_abbrev="LAL",
                is_home=False,
                rest_days=1,
                lines={'pts': 28.5, 'reb': 6.5, 'ast': 5.5, 'pra': 40.5}
            ),
        ],
        "0022400002": [
            PlayerInfo(
                player_id=1628369,  # Tatum
                player_name="Jayson Tatum",
                team_id=1610612738,
                team_abbrev="BOS",
                opponent_abbrev="PHI",
                is_home=True,
                rest_days=3,
                lines={'pts': 27.5, 'reb': 8.5, 'ast': 4.5, 'pra': 40.5}
            ),
        ],
    }
    
    return games, players_by_game


def run_daily_predictions(
    player_fe: PlayerFeatureEngineer,
    team_fe: TeamFeatureEngineer,
    player_trainer: PlayerPropsTrainer,
    team_trainer: TeamPropsTrainer,
    games: list,
    players_by_game: dict
):
    """Run predictions for a day's slate"""
    print("\n" + "="*60)
    print("GENERATING DAILY PREDICTIONS")
    print("="*60)
    
    # Initialize predictor
    predictor = GamePredictor(
        player_feature_engineer=player_fe,
        team_feature_engineer=team_fe,
        player_trainer=player_trainer,
        team_trainer=team_trainer
    )
    
    # Get all predictions
    all_predictions = []
    
    for game in games:
        print(f"\nProcessing: {game.away_team_abbrev} @ {game.home_team_abbrev}")
        
        # Team predictions
        team_preds = predictor.predict_team_props(game)
        all_predictions.extend(team_preds)
        
        for pred in team_preds:
            print(f"  {pred.prop_type}: {pred.prediction.pred_value:.1f} "
                  f"(line: {pred.line}, edge: {pred.edge:.1f if pred.edge else 'N/A'})")
        
        # Player predictions
        players = players_by_game.get(game.game_id, [])
        for player in players:
            player_preds = predictor.predict_player_props(player, game)
            all_predictions.extend(player_preds)
            
            print(f"\n  {player.player_name}:")
            for pred in player_preds:
                edge_str = f"edge: {pred.edge:+.1f}" if pred.edge else ""
                print(f"    {pred.prop_type}: {pred.prediction.pred_value:.1f} "
                      f"[{pred.prediction.confidence:.0%}] {edge_str}")
    
    return all_predictions


def build_daily_parlays(predictions):
    """Build parlays from predictions"""
    print("\n" + "="*60)
    print("BUILDING PARLAYS")
    print("="*60)
    
    builder = ParlayBuilder()
    
    # Build all parlay types
    daily_parlays = builder.build_daily_parlays(
        predictions,
        num_standard=5,
        num_sgp_per_game=2,
        risk_level='moderate'
    )
    
    # Generate report
    report = builder.generate_daily_report(daily_parlays)
    print(report)
    
    # Save report
    report_path = Path(PATH_CONFIG.predictions_dir) / f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    return daily_parlays


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("NBA PROPS PREDICTION SYSTEM")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Check for command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='NBA Props Prediction System')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--predict', action='store_true', help='Run predictions')
    parser.add_argument('--load-models', action='store_true', help='Load saved models')
    parser.add_argument('--sample', action='store_true', help='Use sample data')
    args = parser.parse_args()
    
    # Load data
    print("\nLoading data from database...")
    try:
        player_logs, team_logs, players_df, teams_df = load_data_from_db()
        print(f"  Player logs: {len(player_logs):,} rows")
        print(f"  Team logs: {len(team_logs):,} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using sample data for demonstration...")
        args.sample = True
    
    if args.sample:
        # Create sample data for demonstration
        print("\nCreating sample data...")
        # This would need actual sample data generation
        return
    
    # Initialize feature engineers
    player_fe = PlayerFeatureEngineer(player_logs, team_logs, players_df, teams_df)
    team_fe = TeamFeatureEngineer(team_logs, teams_df)
    
    # Train or load models
    if args.train or not args.load_models:
        player_fe, team_fe, player_trainer, team_trainer = train_all_models(
            player_logs, team_logs, save_models=True
        )
    else:
        print("\nLoading saved models...")
        player_trainer = PlayerPropsTrainer(player_fe)
        team_trainer = TeamPropsTrainer(team_fe)
        player_trainer.load_all_models()
        team_trainer.load_all_models()
    
    # Run predictions
    if args.predict:
        games, players_by_game = create_sample_game_slate()
        predictions = run_daily_predictions(
            player_fe, team_fe, player_trainer, team_trainer,
            games, players_by_game
        )
        
        # Build parlays
        daily_parlays = build_daily_parlays(predictions)
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()