"""
Feature engineering for player prop predictions - OPTIMIZED VERSION
Uses precomputation, caching, and vectorized operations for speed

BACKFILL COMPATIBLE: Does not filter by games_this_season so historical predictions work
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import joblib
import hashlib
from models.config import FEATURE_CONFIG, PATH_CONFIG


class PlayerFeatureEngineer:
    """Engineer features for player prop predictions - OPTIMIZED"""
    
    def __init__(
        self, 
        player_game_logs: pd.DataFrame,
        team_game_logs: pd.DataFrame,
        players_df: Optional[pd.DataFrame] = None,
        teams_df: Optional[pd.DataFrame] = None,
        use_cache: bool = True,
        min_games_for_prediction: int = 3  # Minimum games before we can predict (lowered for backfill)
    ):
        self.players = players_df
        self.teams = teams_df
        self.use_cache = use_cache
        self.min_games_for_prediction = min_games_for_prediction
        
        # Prepare data once
        self.player_logs = self._prepare_player_logs(player_game_logs)
        self.team_logs = self._prepare_team_logs(team_game_logs)
        
        # Build mappings
        self._build_team_mappings()
        
        # Precompute ALL features once
        self._feature_cache: Optional[pd.DataFrame] = None
        self._cache_hash: Optional[str] = None
        self._player_latest_features: Dict[int, pd.Series] = {}
        
        # Precompute on init
        self._precompute_all_features()
    
    def _get_data_hash(self) -> str:
        """Generate hash of data for cache validation"""
        hash_str = f"{len(self.player_logs)}_{self.player_logs['game_date'].max()}_{self.min_games_for_prediction}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]
    
    def _get_cache_path(self) -> Path:
        """Get path for feature cache"""
        cache_dir = Path(PATH_CONFIG.models_dir) / "feature_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"player_features_{self._get_data_hash()}.parquet"
    
    def _prepare_player_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare player game logs - vectorized"""
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)
        
        # Add combined stats - vectorized
        df['pra'] = df['pts'] + df['reb'] + df['ast']
        df['pr'] = df['pts'] + df['reb']
        df['pa'] = df['pts'] + df['ast']
        df['ra'] = df['reb'] + df['ast']
        df['stocks'] = df['stl'] + df['blk']
        
        # Parse matchup - vectorized
        df['is_home'] = df['matchup'].str.contains(' vs\\. ', na=False).astype(np.int8)
        df['opponent_abbrev'] = np.where(
            df['matchup'].str.contains(' vs\\. ', na=False),
            df['matchup'].str.split(' vs\\. ').str[1],
            df['matchup'].str.split(' @ ').str[1]
        )
        
        return df
    
    def _prepare_team_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare team game logs"""
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['team_id', 'game_date']).reset_index(drop=True)
        
        # Vectorized calculations
        df['pace'] = df['fga'] + 0.44 * df['fta'] - df['oreb'] + df['tov']
        df['is_home'] = df['matchup'].str.contains(' vs\\. ', na=False).astype(np.int8)
        df['opponent_abbrev'] = np.where(
            df['matchup'].str.contains(' vs\\. ', na=False),
            df['matchup'].str.split(' vs\\. ').str[1],
            df['matchup'].str.split(' @ ').str[1]
        )
        
        return df
    
    def _build_team_mappings(self):
        """Build team abbreviation to ID mappings"""
        if self.teams is not None and 'abbreviation' in self.teams.columns:
            self.team_id_to_abbrev = dict(zip(self.teams['team_id'], self.teams['abbreviation']))
        else:
            # Extract from logs
            team_abbrevs = self.team_logs.groupby('team_id')['matchup'].first().apply(
                lambda x: x.split()[0] if pd.notna(x) else None
            )
            self.team_id_to_abbrev = team_abbrevs.to_dict()
        
        self.abbrev_to_team_id = {v: k for k, v in self.team_id_to_abbrev.items() if v}
    
    def _precompute_all_features(self):
        """Precompute all features once - the key optimization"""
        cache_path = self._get_cache_path()
        
        # Try loading from cache
        if self.use_cache and cache_path.exists():
            try:
                print("  Loading cached features...")
                self._feature_cache = pd.read_parquet(cache_path)
                self._build_player_latest_index()
                print(f"  ✓ Loaded {len(self._feature_cache):,} cached feature rows")
                return
            except Exception as e:
                print(f"  Cache load failed: {e}")
        
        print("  Building features (one-time computation)...")
        
        df = self.player_logs.copy()
        
        # Stats to compute features for - INCLUDING plus_minus
        stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'min', 'fgm', 'fga', 
                     'fg3m', 'fg3a', 'ftm', 'fta', 'oreb', 'dreb', 'tov', 'pf',
                     'pra', 'pr', 'pa', 'ra', 'stocks', 'plus_minus']
        stat_cols = [s for s in stat_cols if s in df.columns]
        
        # =====================================================
        # VECTORIZED ROLLING CALCULATIONS (FAST)
        # =====================================================
        
        windows = FEATURE_CONFIG.rolling_windows  # [3, 5, 10, 20]
        
        # Group once
        grouped = df.groupby('player_id')
        
        # Pre-allocate dictionary for new columns
        new_cols = {}
        
        for col in stat_cols:
            if col not in df.columns:
                continue
            
            # Shift once per column
            shifted = grouped[col].shift(1)
            
            for window in windows:
                # Rolling calculations on shifted data - use min_periods=1 for early games
                rolling = shifted.groupby(df['player_id']).rolling(window, min_periods=1)
                
                new_cols[f'{col}_avg_{window}'] = rolling.mean().reset_index(level=0, drop=True)
                new_cols[f'{col}_std_{window}'] = shifted.groupby(df['player_id']).rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
                new_cols[f'{col}_max_{window}'] = rolling.max().reset_index(level=0, drop=True)
                new_cols[f'{col}_min_{window}'] = rolling.min().reset_index(level=0, drop=True)
            
            # Season average
            new_cols[f'{col}_season_avg'] = grouped[col].apply(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            ).reset_index(level=0, drop=True)
        
        # Games this season (cumcount starts at 0, so game 1 = 0, game 2 = 1, etc.)
        new_cols['games_this_season'] = df.groupby(['player_id', 'season']).cumcount()
        
        # Career games (total games for this player)
        new_cols['career_games'] = grouped.cumcount()
        
        # =====================================================
        # TREND FEATURES
        # =====================================================
        
        for col in stat_cols:
            avg_3 = f'{col}_avg_3'
            avg_10 = f'{col}_avg_10'
            if avg_3 in new_cols and avg_10 in new_cols:
                new_cols[f'{col}_trend'] = new_cols[avg_3] - new_cols[avg_10]
            
            std_10 = f'{col}_std_10'
            if avg_10 in new_cols and std_10 in new_cols:
                new_cols[f'{col}_consistency'] = new_cols[avg_10] / (new_cols[std_10] + 0.1)
        
        # =====================================================
        # MOMENTUM FEATURES
        # =====================================================
        
        for col in stat_cols:
            avg_3 = f'{col}_avg_3'
            avg_10 = f'{col}_avg_10'
            if avg_3 in new_cols and avg_10 in new_cols:
                # Momentum: ratio of recent performance to baseline
                new_cols[f'{col}_momentum'] = new_cols[avg_3] / (new_cols[avg_10] + 0.1)
        
        # =====================================================
        # CONTEXT FEATURES
        # =====================================================
        
        # Rest days - vectorized
        new_cols['rest_days'] = grouped['game_date'].diff().dt.days.fillna(3).clip(0, 10)
        new_cols['is_b2b'] = (new_cols['rest_days'] <= 1).astype(np.int8)
        
        # Date features
        new_cols['day_of_week'] = df['game_date'].dt.dayofweek.astype(np.int8)
        new_cols['month'] = df['game_date'].dt.month.astype(np.int8)
        
        # Win/loss features
        new_cols['recent_wins'] = grouped['wl'].apply(
            lambda x: (x.shift(1) == 'W').rolling(5, min_periods=1).sum()
        ).reset_index(level=0, drop=True)
        
        # Team streak - consecutive wins (positive) or losses (negative)
        def calc_streak(s):
            s = s.shift(1)
            streak = pd.Series(0, index=s.index)
            current = 0
            for i, val in enumerate(s):
                if pd.isna(val):
                    current = 0
                elif val == 'W':
                    current = current + 1 if current > 0 else 1
                else:
                    current = current - 1 if current < 0 else -1
                streak.iloc[i] = current
            return streak
        
        new_cols['team_streak'] = grouped['wl'].apply(calc_streak).reset_index(level=0, drop=True)
        
        # =====================================================
        # MINUTES & USAGE FEATURES  
        # =====================================================
        
        if 'min' in df.columns:
            min_shifted = grouped['min'].shift(1)
            new_cols['min_trend'] = (
                min_shifted.groupby(df['player_id']).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True) -
                min_shifted.groupby(df['player_id']).rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
            )
            new_cols['min_stability'] = min_shifted.groupby(df['player_id']).rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
            
            # Per-minute rates for main stats
            min_safe = df['min'].clip(lower=1)
            for stat in ['pts', 'reb', 'ast', 'stl', 'blk']:
                if stat in df.columns:
                    per_min = df[stat] / min_safe
                    new_cols[f'{stat}_per_min_avg_5'] = (
                        per_min.groupby(df['player_id']).shift(1)
                        .groupby(df['player_id']).rolling(5, min_periods=1).mean()
                        .reset_index(level=0, drop=True)
                    )
        
        # Usage rate proxy
        if 'fga' in df.columns and 'min' in df.columns:
            fga_rate = df['fga'] / df['min'].clip(lower=1) * 36
            new_cols['fga_rate_avg_5'] = (
                fga_rate.groupby(df['player_id']).shift(1)
                .groupby(df['player_id']).rolling(5, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
        
        # True shooting attempts
        if all(c in df.columns for c in ['fga', 'fta']):
            tsa = df['fga'] + 0.44 * df['fta']
            new_cols['tsa_avg_5'] = (
                tsa.groupby(df['player_id']).shift(1)
                .groupby(df['player_id']).rolling(5, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
        
        # Assist to turnover ratio
        if all(c in df.columns for c in ['ast', 'tov']):
            ast_to_tov = df['ast'] / df['tov'].clip(lower=0.5)
            new_cols['ast_to_tov_avg_5'] = (
                ast_to_tov.groupby(df['player_id']).shift(1)
                .groupby(df['player_id']).rolling(5, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
        
        # =====================================================
        # OPPONENT FEATURES
        # =====================================================
        
        df['opponent_team_id'] = df['opponent_abbrev'].map(self.abbrev_to_team_id)
        
        # Precompute team defensive stats (with all windows)
        team_stats = self._compute_team_stats()
        
        # Merge opponent stats
        df = df.merge(
            team_stats.add_prefix('opp_'),
            left_on='opponent_team_id',
            right_index=True,
            how='left'
        )
        
        # Historical vs opponent - averages
        for stat in ['pts', 'reb', 'ast', 'pra']:
            if stat in df.columns:
                new_cols[f'{stat}_vs_opp_avg'] = (
                    df.groupby(['player_id', 'opponent_team_id'])[stat]
                    .apply(lambda x: x.shift(1).expanding(min_periods=1).mean())
                    .reset_index(level=[0,1], drop=True)
                )
        
        # Games vs opponent count
        for stat in ['pts', 'reb', 'ast', 'pra']:
            new_cols[f'{stat}_vs_opp_games'] = (
                df.groupby(['player_id', 'opponent_team_id']).cumcount()
            )
        
        # =====================================================
        # COMBINE ALL FEATURES
        # =====================================================
        
        # Add all new columns at once
        feature_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        # Remove duplicates and clean
        feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]
        
        # =====================================================
        # FILTER: Only require min_games_for_prediction (default 3)
        # This is MUCH more permissive than before for backfill compatibility
        # =====================================================
        # Use career_games instead of games_this_season so cross-season works
        feature_df = feature_df[feature_df['career_games'] >= self.min_games_for_prediction]
        
        # Convert numpy types for database compatibility
        feature_df = self._convert_types_for_db(feature_df)
        
        self._feature_cache = feature_df
        
        # Build index for fast player lookup
        self._build_player_latest_index()
        
        # Save cache
        if self.use_cache:
            try:
                feature_df.to_parquet(cache_path)
                print(f"  ✓ Cached features to {cache_path}")
            except Exception as e:
                print(f"  Cache save failed: {e}")
        
        print(f"  ✓ Built {len(feature_df):,} feature rows")
    
    def _compute_team_stats(self) -> pd.DataFrame:
        """Compute team defensive stats for opponent features - all windows"""
        df = self.team_logs.copy()
        
        windows = FEATURE_CONFIG.rolling_windows  # [3, 5, 10, 20]
        stats_dict = {}
        
        for stat in ['pts', 'fg_pct', 'fg3_pct', 'pace']:
            if stat not in df.columns:
                continue
            for window in windows:
                col_name = f'team_{stat}_avg_{window}'
                stats_dict[col_name] = (
                    df.groupby('team_id')[stat]
                    .apply(lambda x: x.shift(1).rolling(window, min_periods=1).mean().iloc[-1] if len(x) > 0 else np.nan)
                )
        
        return pd.DataFrame(stats_dict)
    
    def _convert_types_for_db(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame types for PostgreSQL compatibility"""
        df = df.copy()
        
        # Convert float32 to float64
        for col in df.select_dtypes(include=['float32']).columns:
            df[col] = df[col].astype('float64')
        
        # Convert int8/int32 to int64
        for col in df.select_dtypes(include=['int8', 'int32']).columns:
            df[col] = df[col].astype('int64')
        
        return df
    
    def _build_player_latest_index(self):
        """Build index of latest features per player for fast prediction"""
        if self._feature_cache is None:
            return
        
        # Get last row per player
        latest_idx = self._feature_cache.groupby('player_id').tail(1).index
        latest_df = self._feature_cache.loc[latest_idx]
        
        self._player_latest_features = {
            row['player_id']: row for _, row in latest_df.iterrows()
        }
        
        print(f"  ✓ Indexed {len(self._player_latest_features)} players for fast lookup")
    
    def build_feature_set(self, target_stat: str, include_all: bool = True) -> pd.DataFrame:
        """Return precomputed features - instant!"""
        if self._feature_cache is None:
            self._precompute_all_features()
        
        if target_stat not in self._feature_cache.columns:
            raise ValueError(f"Target stat '{target_stat}' not found")
        
        return self._feature_cache.copy()
    
    def get_feature_columns(self, df: pd.DataFrame, target: str) -> List[str]:
        """Get list of feature columns"""
        exclude_cols = {
            target, 'player_id', 'game_id', 'game_date', 'season', 
            'matchup', 'wl', 'opponent_abbrev', 'opponent_team_id',
            'id', 'player_name', 'is_active', 'created_at'
        }
        
        raw_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'min', 'fgm', 'fga',
                     'fg3m', 'fg3a', 'ftm', 'fta', 'oreb', 'dreb', 'tov', 'pf',
                     'pra', 'pr', 'pa', 'ra', 'stocks', 'plus_minus', 'fg_pct',
                     'fg3_pct', 'ft_pct']
        exclude_cols.update(raw_stats)
        
        feature_cols = []
        for c in df.columns:
            if c in exclude_cols:
                continue
            if c.startswith('opp_game_date') or c.startswith('opp_team_id'):
                continue
            if df[c].dtype in ['float64', 'int64', 'float32', 'int32', 'int8', 'float']:
                feature_cols.append(c)
        
        return feature_cols
    
    def prepare_prediction_features(
        self,
        player_id: int,
        opponent_abbrev: str,
        is_home: bool,
        rest_days: int = 2,
        target_stat: str = 'pts'
    ) -> Optional[pd.DataFrame]:
        """
        Get features for prediction - FAST via precomputed index
        """
        # Fast lookup from precomputed index
        if player_id not in self._player_latest_features:
            return None
        
        latest = self._player_latest_features[player_id].copy()
        
        # Override context features
        latest['is_home'] = int(is_home)
        latest['rest_days'] = rest_days
        latest['is_b2b'] = int(rest_days <= 1)
        latest['opponent_abbrev'] = opponent_abbrev
        
        # Update opponent team id
        opp_team_id = self.abbrev_to_team_id.get(opponent_abbrev)
        if opp_team_id:
            latest['opponent_team_id'] = opp_team_id
        
        return pd.DataFrame([latest])
    
    def refresh_cache(self):
        """Force rebuild of feature cache"""
        self._feature_cache = None
        self._player_latest_features = {}
        self.use_cache = False  # Don't load old cache
        self._precompute_all_features()
        self.use_cache = True
    
    def get_player_feature_summary(self, player_id: int) -> Optional[Dict]:
        """Quick summary of a player's features"""
        if player_id not in self._player_latest_features:
            return None
        
        features = self._player_latest_features[player_id]
        
        return {
            'player_id': player_id,
            'games_this_season': features.get('games_this_season', 0),
            'career_games': features.get('career_games', 0),
            'pts_avg_5': features.get('pts_avg_5'),
            'pts_avg_10': features.get('pts_avg_10'),
            'reb_avg_5': features.get('reb_avg_5'),
            'ast_avg_5': features.get('ast_avg_5'),
            'min_avg_5': features.get('min_avg_5'),
        }