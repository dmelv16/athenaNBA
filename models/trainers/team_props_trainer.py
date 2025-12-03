"""
Model training for team prop predictions (spread, totals)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime

from models.config import MODEL_CONFIG, PATH_CONFIG
from models.base import ModelMetrics, Prediction
from models.features.team_features import TeamFeatureEngineer


class TrainedTeamModel:
    """Container for a trained team prop model"""
    
    def __init__(
        self,
        target: str,
        model: xgb.XGBRegressor,
        feature_cols: List[str],
        metrics: ModelMetrics,
        trained_at: datetime = None
    ):
        self.target = target
        self.model = model
        self.feature_cols = feature_cols
        self.metrics = metrics
        self.trained_at = trained_at or datetime.now()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X_clean = self._prepare_features(X)
        return self.model.predict(X_clean)
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features exist"""
        X_clean = X.copy()
        for col in self.feature_cols:
            if col not in X_clean.columns:
                X_clean[col] = 0
        return X_clean[self.feature_cols].fillna(0)
    
    def predict_with_confidence(self, X: pd.DataFrame) -> List[Prediction]:
        """Make predictions with confidence estimates"""
        X_clean = self._prepare_features(X)
        preds = self.model.predict(X_clean)
        
        results = []
        for i, pred in enumerate(preds):
            # Base confidence on model's historical accuracy
            base_confidence = min(0.7, self.metrics.directional_accuracy)
            
            # Adjust based on prediction magnitude (more extreme = less confident)
            if self.target == 'spread':
                # Spreads near 0 are harder to predict
                confidence = base_confidence * (1 - 0.1 * min(abs(pred), 10) / 10)
            else:
                confidence = base_confidence
            
            results.append(Prediction(
                pred_value=pred,
                confidence=np.clip(confidence, 0.4, 0.75),
                lower_bound=pred - self.metrics.mae,
                upper_bound=pred + self.metrics.mae,
                features_used=len(self.feature_cols),
                model_name=f"team_{self.target}"
            ))
        
        return results
    
    def save(self, path: str = None):
        """Save model to disk"""
        if path is None:
            path = Path(PATH_CONFIG.models_dir) / f"team_{self.target}_model.joblib"
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'TrainedTeamModel':
        """Load model from disk"""
        return joblib.load(path)


class TeamPropsTrainer:
    """Train and manage team prop models"""
    
    TARGETS = ['spread', 'total_pts']
    
    def __init__(self, feature_engineer: TeamFeatureEngineer):
        self.fe = feature_engineer
        self.models: Dict[str, TrainedTeamModel] = {}
    
    def train_spread_model(
        self,
        test_ratio: float = None,
        verbose: bool = True
    ) -> TrainedTeamModel:
        """Train model for spread prediction"""
        return self._train_model('spread', test_ratio, verbose)
    
    def train_total_model(
        self,
        test_ratio: float = None,
        verbose: bool = True
    ) -> TrainedTeamModel:
        """Train model for total points prediction"""
        return self._train_model('total_pts', test_ratio, verbose)
    
    def _train_model(
        self,
        target: str,
        test_ratio: float = None,
        verbose: bool = True
    ) -> TrainedTeamModel:
        """Train XGBoost model for a team stat"""
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training model for: {target.upper()}")
            print('='*50)
        
        # Build features
        if target == 'spread':
            df, _ = self.fe.build_spread_features()
        else:
            df, _ = self.fe.build_total_features()
        
        # Drop rows with missing target
        df = df.dropna(subset=[target])
        
        if verbose:
            print(f"Total games: {len(df):,}")
        
        # Get feature columns
        feature_cols = self.fe.get_feature_columns(df)
        
        if verbose:
            print(f"Features: {len(feature_cols)}")
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        y = df[target]
        
        # Time-based split
        test_ratio = test_ratio or MODEL_CONFIG.test_ratio
        split_idx = int(len(df) * (1 - test_ratio))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if verbose:
            print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Train
        params = MODEL_CONFIG.xgb_params.copy()
        model = xgb.XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_train, train_preds, y_test, test_preds, target)
        
        if verbose:
            print(f"\nTest Metrics:")
            print(f"  MAE:  {metrics.mae:.2f}")
            print(f"  RMSE: {metrics.rmse:.2f}")
            if target == 'spread':
                print(f"  ATS Accuracy: {metrics.directional_accuracy:.1%}")
            else:
                print(f"  O/U Accuracy: {metrics.directional_accuracy:.1%}")
        
        trained_model = TrainedTeamModel(
            target=target,
            model=model,
            feature_cols=feature_cols,
            metrics=metrics
        )
        
        self.models[target] = trained_model
        return trained_model
    
    def _calculate_metrics(
        self,
        y_train: pd.Series,
        train_preds: np.ndarray,
        y_test: pd.Series,
        test_preds: np.ndarray,
        target: str
    ) -> ModelMetrics:
        """Calculate model metrics"""
        mae = mean_absolute_error(y_test, test_preds)
        rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        r2 = r2_score(y_test, test_preds)
        
        non_zero = y_test != 0
        if non_zero.sum() > 0:
            mape = np.mean(np.abs((y_test[non_zero] - test_preds[non_zero]) / y_test[non_zero])) * 100
        else:
            mape = 0
        
        # Directional accuracy
        if target == 'spread':
            # For spread: predict if home wins (spread > 0) or loses
            actual_home_wins = y_test > 0
            pred_home_wins = test_preds > 0
            directional = (actual_home_wins == pred_home_wins).mean()
        else:
            # For totals: use median as benchmark
            median_total = y_train.median()
            actual_over = y_test > median_total
            pred_over = test_preds > median_total
            directional = (actual_over == pred_over).mean()
        
        return ModelMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            r2=r2,
            directional_accuracy=directional
        )
    
    def train_all_models(self, verbose: bool = True) -> Dict[str, TrainedTeamModel]:
        """Train both spread and total models"""
        if verbose:
            print("\n" + "="*60)
            print("TRAINING TEAM PROP MODELS")
            print("="*60)
        
        self.train_spread_model(verbose=verbose)
        self.train_total_model(verbose=verbose)
        
        return self.models
    
    def get_feature_importance(self, target: str, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance"""
        if target not in self.models:
            raise ValueError(f"No model trained for {target}")
        
        model = self.models[target]
        importance = pd.DataFrame({
            'feature': model.feature_cols,
            'importance': model.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance.head(top_n)
    
    def save_all_models(self, directory: str = None):
        """Save all models"""
        directory = directory or PATH_CONFIG.models_dir
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for target, model in self.models.items():
            path = Path(directory) / f"team_{target}_model.joblib"
            model.save(path)
            print(f"Saved: {path}")
    
    def load_all_models(self, directory: str = None):
        """Load all saved models"""
        directory = directory or PATH_CONFIG.models_dir
        
        for target in self.TARGETS:
            path = Path(directory) / f"team_{target}_model.joblib"
            if path.exists():
                self.models[target] = TrainedTeamModel.load(path)
                print(f"Loaded: {path}")
    
    def predict(
        self,
        target: str,
        features_df: pd.DataFrame,
        with_confidence: bool = True
    ):
        """Make predictions"""
        if target not in self.models:
            raise ValueError(f"No model trained for {target}")
        
        model = self.models[target]
        
        if with_confidence:
            return model.predict_with_confidence(features_df)
        return model.predict(features_df)