"""
Model training for player prop predictions - with FAST prediction mode
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime

from models.config import MODEL_CONFIG, PATH_CONFIG
from models.base import ModelMetrics, Prediction
from models.features.player_features import PlayerFeatureEngineer


class TrainedPlayerModel:
    """Container for a trained player prop model"""
    
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
        
        # Precompute base confidence from metrics
        self._base_confidence = min(0.75, max(0.5, self.metrics.directional_accuracy))
    
    def __setstate__(self, state):
        """Handle loading older model versions that lack newer attributes"""
        self.__dict__.update(state)
        # Add _base_confidence if missing (for backwards compatibility with old saved models)
        if '_base_confidence' not in state:
            self._base_confidence = min(0.75, max(0.5, self.metrics.directional_accuracy))
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X_clean = X[self.feature_cols].fillna(0)
        return self.model.predict(X_clean)
    
    def predict_fast(self, X: pd.DataFrame) -> List[Prediction]:
        """
        FAST prediction without tree variance calculation.
        Uses precomputed base confidence from training metrics.
        ~10x faster than predict_with_confidence
        """
        X_clean = X[self.feature_cols].fillna(0)
        preds = self.model.predict(X_clean)
        
        results = []
        for pred in preds:
            results.append(Prediction(
                pred_value=float(pred),
                confidence=self._base_confidence,
                lower_bound=float(pred - 1.5 * self.metrics.mae),
                upper_bound=float(pred + 1.5 * self.metrics.mae),
                features_used=len(self.feature_cols),
                model_name=f"player_{self.target}"
            ))
        
        return results
    
    def predict_with_confidence(self, X: pd.DataFrame, fast_mode: bool = True) -> List[Prediction]:
        """
        Make predictions with confidence intervals.
        Set fast_mode=False for full tree variance calculation (slower but more accurate confidence)
        """
        if fast_mode:
            return self.predict_fast(X)
        
        X_clean = X[self.feature_cols].fillna(0)
        
        # Get point prediction
        preds = self.model.predict(X_clean)
        
        # Estimate confidence using tree variance (SLOW)
        all_tree_preds = []
        n_trees = min(self.model.n_estimators, 50)  # Cap at 50 trees for speed
        step = max(1, self.model.n_estimators // n_trees)
        
        for i in range(step, self.model.n_estimators + 1, step):
            tree_pred = self.model.predict(X_clean, iteration_range=(0, i))
            all_tree_preds.append(tree_pred)
        
        tree_preds = np.array(all_tree_preds)
        pred_std = tree_preds.std(axis=0)
        
        results = []
        for i in range(len(preds)):
            std = pred_std[i] if i < len(pred_std) else self.metrics.mae
            confidence = 1 / (1 + std / (abs(preds[i]) + 1))
            confidence = np.clip(confidence, 0.3, 0.9)
            
            results.append(Prediction(
                pred_value=float(preds[i]),
                confidence=float(confidence),
                lower_bound=float(preds[i] - 1.5 * self.metrics.mae),
                upper_bound=float(preds[i] + 1.5 * self.metrics.mae),
                features_used=len(self.feature_cols),
                model_name=f"player_{self.target}"
            ))
        
        return results
    
    def predict_batch(self, X: pd.DataFrame) -> Tuple[np.ndarray, float, float, float]:
        """
        Ultra-fast batch prediction returning raw arrays.
        Returns: (predictions, confidence, lower_mult, upper_mult)
        Caller handles conversion to Prediction objects
        """
        X_clean = X[self.feature_cols].fillna(0)
        preds = self.model.predict(X_clean)
        return preds, self._base_confidence, self.metrics.mae * 1.5, self.metrics.mae * 1.5
    
    def save(self, path: str = None):
        """Save model to disk"""
        if path is None:
            path = Path(PATH_CONFIG.models_dir) / f"player_{self.target}_model.joblib"
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'TrainedPlayerModel':
        """Load model from disk"""
        return joblib.load(path)


class PlayerPropsTrainer:
    """Train and manage player prop models"""
    
    TARGETS = ['pts', 'reb', 'ast', 'stl', 'blk', 'pra', 'pr', 'pa', 'ra', 'stocks']
    
    def __init__(self, feature_engineer: PlayerFeatureEngineer):
        self.fe = feature_engineer
        self.models: Dict[str, TrainedPlayerModel] = {}
    
    def train_model(
        self,
        target: str,
        test_ratio: float = None,
        params: Dict = None,
        verbose: bool = True
    ) -> TrainedPlayerModel:
        """Train XGBoost model for a specific stat"""
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training model for: {target.upper()}")
            print('='*50)
        
        # Build features
        df = self.fe.build_feature_set(target)
        
        if verbose:
            print(f"Total samples: {len(df):,}")
        
        # Get feature columns
        feature_cols = self.fe.get_feature_columns(df, target)
        
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
        
        # Train model
        params = params or MODEL_CONFIG.xgb_params
        model = xgb.XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_train, train_preds, y_test, test_preds)
        
        if verbose:
            print(f"\nTest Metrics:")
            print(f"  MAE:  {metrics.mae:.2f}")
            print(f"  RMSE: {metrics.rmse:.2f}")
            print(f"  MAPE: {metrics.mape:.1f}%")
            print(f"  R²:   {metrics.r2:.3f}")
            print(f"  Direction Accuracy: {metrics.directional_accuracy:.1%}")
        
        # Create trained model object
        trained_model = TrainedPlayerModel(
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
        test_preds: np.ndarray
    ) -> ModelMetrics:
        """Calculate comprehensive model metrics"""
        
        mae = mean_absolute_error(y_test, test_preds)
        rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        r2 = r2_score(y_test, test_preds)
        
        # MAPE (handle zeros)
        non_zero = y_test != 0
        if non_zero.sum() > 0:
            mape = np.mean(np.abs((y_test[non_zero] - test_preds[non_zero]) / y_test[non_zero])) * 100
        else:
            mape = 0
        
        # Directional accuracy (if we had lines, this would be over/under accuracy)
        median_val = y_train.median()
        actual_over = y_test > median_val
        pred_over = test_preds > median_val
        directional_acc = (actual_over == pred_over).mean()
        
        return ModelMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            r2=r2,
            directional_accuracy=directional_acc
        )
    
    def train_all_models(self, verbose: bool = True) -> Dict[str, TrainedPlayerModel]:
        """Train models for all target stats"""
        if verbose:
            print("\n" + "="*60)
            print("TRAINING ALL PLAYER PROP MODELS")
            print("="*60)
        
        for target in self.TARGETS:
            try:
                self.train_model(target, verbose=verbose)
            except Exception as e:
                print(f"Failed to train {target}: {e}")
        
        if verbose:
            print("\n" + "="*60)
            print("TRAINING COMPLETE - SUMMARY")
            print("="*60)
            for target, model in self.models.items():
                print(f"  {target:6s} | MAE: {model.metrics.mae:5.2f} | "
                      f"Dir: {model.metrics.directional_accuracy:.1%}")
        
        return self.models
    
    def get_feature_importance(self, target: str, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for a model"""
        if target not in self.models:
            raise ValueError(f"No model trained for {target}")
        
        model = self.models[target]
        importance = pd.DataFrame({
            'feature': model.feature_cols,
            'importance': model.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance.head(top_n)
    
    def cross_validate(
        self,
        target: str,
        n_splits: int = 5,
        verbose: bool = True
    ) -> Dict:
        """Perform time-series cross-validation"""
        df = self.fe.build_feature_set(target)
        feature_cols = self.fe.get_feature_columns(df, target)
        
        X = df[feature_cols].fillna(0)
        y = df[target]
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {'mae': [], 'rmse': [], 'r2': [], 'directional': []}
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = xgb.XGBRegressor(**MODEL_CONFIG.xgb_params)
            model.fit(X_train, y_train, verbose=False)
            
            preds = model.predict(X_test)
            
            results['mae'].append(mean_absolute_error(y_test, preds))
            results['rmse'].append(np.sqrt(mean_squared_error(y_test, preds)))
            results['r2'].append(r2_score(y_test, preds))
            
            median_val = y_train.median()
            directional = ((y_test > median_val) == (preds > median_val)).mean()
            results['directional'].append(directional)
            
            if verbose:
                print(f"Fold {fold+1}: MAE={results['mae'][-1]:.2f}, "
                      f"Dir={results['directional'][-1]:.1%}")
        
        summary = {
            'mae_mean': np.mean(results['mae']),
            'mae_std': np.std(results['mae']),
            'rmse_mean': np.mean(results['rmse']),
            'r2_mean': np.mean(results['r2']),
            'directional_mean': np.mean(results['directional'])
        }
        
        if verbose:
            print(f"\nCV Summary for {target}:")
            print(f"  MAE: {summary['mae_mean']:.2f} ± {summary['mae_std']:.2f}")
            print(f"  Direction: {summary['directional_mean']:.1%}")
        
        return summary
    
    def save_all_models(self, directory: str = None):
        """Save all trained models"""
        directory = directory or PATH_CONFIG.models_dir
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for target, model in self.models.items():
            path = Path(directory) / f"player_{target}_model.joblib"
            model.save(path)
            print(f"Saved: {path}")
    
    def load_all_models(self, directory: str = None):
        """Load all saved models"""
        directory = directory or PATH_CONFIG.models_dir
        
        for target in self.TARGETS:
            path = Path(directory) / f"player_{target}_model.joblib"
            if path.exists():
                self.models[target] = TrainedPlayerModel.load(path)
                print(f"Loaded: {path}")
    
    def predict(
        self,
        target: str,
        features_df: pd.DataFrame,
        with_confidence: bool = True,
        fast_mode: bool = True
    ):
        """Make predictions for a target stat"""
        if target not in self.models:
            raise ValueError(f"No model trained for {target}")
        
        model = self.models[target]
        
        if with_confidence:
            return model.predict_with_confidence(features_df, fast_mode=fast_mode)
        return model.predict(features_df)