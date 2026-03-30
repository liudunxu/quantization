"""CatBoost model for stock trading decisions."""

from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
from ..utils.config import get_config


class StockTradingModel:
    """CatBoost model for stock trading decisions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section('model').get('catboost', {})
        self.iterations = self.config.get('iterations', 500)
        self.depth = self.config.get('depth', 6)
        self.learning_rate = self.config.get('learning_rate', 0.03)
        self.l2_leaf_reg = self.config.get('l2_leaf_reg', 3)
        self.random_seed = self.config.get('random_seed', 42)
        self.model: Optional[CatBoostClassifier] = None
        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None
        self.max_features: int = 80  # Maximum features to use

    def _select_features(self, df: pd.DataFrame, labels: pd.Series) -> List[str]:
        """Select top features based on importance and diversity.

        Uses a two-stage approach:
        1. Quick preliminary model to get feature importance
        2. Remove low importance, correlated, and low variance features
        """
        # Drop non-feature columns
        drop_cols = ['date', 'stock_code', 'sector', 'industry', 'close', 'open', 'high', 'low', 'volume']
        drop_cols = [c for c in drop_cols if c in df.columns]
        feature_df = df.drop(columns=drop_cols, errors='ignore')

        # Remove object columns
        object_cols = [c for c in feature_df.columns if feature_df[c].dtype == 'object' or feature_df[c].dtype == 'str']
        feature_df = feature_df.drop(columns=object_cols, errors='ignore')

        # Remove rows where labels are NaN
        valid_idx = ~labels.isna()
        X_quick = feature_df[valid_idx]
        y_quick = labels[valid_idx]

        if len(X_quick) < 30:
            # Not enough samples for importance-based selection, fall back to variance
            variances = feature_df.var()
            low_var_cols = variances[variances < 0.0001].index.tolist()
            feature_df = feature_df.drop(columns=low_var_cols, errors='ignore')
            return feature_df.columns.tolist()[:self.max_features]

        # Quick preliminary model to get feature importance
        quick_model = CatBoostClassifier(
            iterations=50,  # Few iterations for speed
            depth=4,
            learning_rate=0.1,
            random_seed=self.random_seed,
            verbose=False
        )
        quick_model.fit(X_quick, y_quick, verbose=False)

        # Get feature importance
        importance = quick_model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': X_quick.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Keep top features by importance (at least top 60%)
        importance_threshold = importance_df['importance'].quantile(0.40)
        top_features = importance_df[importance_df['importance'] >= importance_threshold]['feature'].tolist()

        if len(top_features) < 20:
            # If too few, take top 40 by importance
            top_features = importance_df.head(40)['feature'].tolist()

        feature_df = feature_df[top_features]

        # Remove features with low variance (near-constant)
        variances = feature_df.var()
        low_var_cols = variances[variances < 0.0001].index.tolist()
        feature_df = feature_df.drop(columns=low_var_cols, errors='ignore')

        # Remove features highly correlated with each other
        if len(feature_df.columns) > 1:
            corr_matrix = feature_df.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            feature_df = feature_df.drop(columns=to_drop, errors='ignore')

        # Final selection: limit to max_features
        if len(feature_df.columns) > self.max_features:
            # Use importance from preliminary model to select final features
            final_importance = importance_df[importance_df['feature'].isin(feature_df.columns)]
            final_cols = final_importance.head(self.max_features)['feature'].tolist()
            feature_df = feature_df[final_cols]

        return feature_df.columns.tolist()

    def _prepare_features(self, df: pd.DataFrame, use_selection: bool = True) -> pd.DataFrame:
        """Prepare features for modeling."""
        # Drop non-feature columns
        drop_cols = ['date', 'stock_code', 'sector', 'industry']
        drop_cols = [c for c in drop_cols if c in df.columns]

        feature_df = df.drop(columns=drop_cols, errors='ignore')

        # Remove any remaining object or string columns
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object' or feature_df[col].dtype == 'str':
                feature_df = feature_df.drop(columns=[col])

        # Apply feature selection if trained
        if use_selection and self.selected_features is not None:
            available_features = [f for f in self.selected_features if f in feature_df.columns]
            if len(available_features) > 0:
                feature_df = feature_df[available_features]

        # Fill NaN with median
        for col in feature_df.columns:
            if feature_df[col].isna().any():
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())

        return feature_df

    def _create_labels(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        threshold: float = 0.02,
        use_composite: bool = True,
        trend_weight: float = 0.3,
        momentum_weight: float = 0.3,
        market_weight: float = 0.2
    ) -> pd.Series:
        """Create labels for training with multi-dimensional composite scoring.

        Label meanings:
        - 1: Buy (strong upward signal)
        - 0: Hold (neutral/inconclusive)
        - -1: Sell (strong downward signal)

        Composite scoring considers:
        - Future returns: Whether price actually went up/down
        - Trend alignment: MA arrangement (bullish/bearish)
        - Momentum confirmation: RSI, MACD direction
        - Market environment: Market index performance
        """
        future_returns = df['close'].shift(-forward_days) / df['close'] - 1

        if not use_composite:
            # Original simple threshold method
            labels = pd.Series(0, index=df.index)
            labels[future_returns > threshold] = 1
            labels[future_returns < -threshold] = -1
            return labels

        # ========== Calculate component scores ==========

        # 1. Future return score (required condition) - use float for mixing
        return_score = pd.Series(0.0, index=df.index)
        return_score[future_returns > threshold] = 1.0
        return_score[future_returns < -threshold] = -1.0

        # 2. Trend score (MA arrangement) - use float
        trend_score = pd.Series(0.0, index=df.index)
        if 'ma_bullish_arrange' in df.columns and 'ma_bearish_arrange' in df.columns:
            trend_score[df['ma_bullish_arrange'] == 1] = 1.0
            trend_score[df['ma_bearish_arrange'] == 1] = -1.0

        # 3. Momentum score (RSI, MACD)
        momentum_score = pd.Series(0.0, index=df.index)
        if 'rsi' in df.columns:
            # RSI > 60 is bullish, < 40 is bearish
            momentum_score[df['rsi'] > 60] += 0.5
            momentum_score[df['rsi'] < 40] -= 0.5
        if 'macd_hist' in df.columns:
            # MACD histogram positive is bullish
            momentum_score[df['macd_hist'] > 0] += 0.5
            momentum_score[df['macd_hist'] < 0] -= 0.5

        # Normalize momentum to -1, 0, 1
        momentum_score = momentum_score.clip(-1.0, 1.0)

        # 4. Market score (if available)
        market_score = pd.Series(0.0, index=df.index)
        if 'index_returns' in df.columns:
            market_score[df['index_returns'] > 0.01] = 0.5
            market_score[df['index_returns'] < -0.01] = -0.5

        # ========== Combine into composite signal ==========
        # Weighted combination
        return_weight = 1.0 - trend_weight - momentum_weight - market_weight
        composite = (
            return_score * return_weight +
            trend_score * trend_weight +
            momentum_score * momentum_weight +
            market_score * market_weight
        )

        # ========== Create final labels ==========
        # Buy: composite score >= 0.5 AND future return is positive
        # Sell: composite score <= -0.5 AND future return is negative
        # Hold: everything else

        labels = pd.Series(0, index=df.index)
        buy_condition = (composite >= 0.5) & (future_returns > threshold * 0.5)
        sell_condition = (composite <= -0.5) & (future_returns < -threshold * 0.5)

        labels[buy_condition] = 1
        labels[sell_condition] = -1

        return labels

    def train(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        threshold: float = 0.02,
        eval_df: Optional[pd.DataFrame] = None,
        use_composite_labels: bool = True,
        trend_weight: float = 0.3,
        momentum_weight: float = 0.3,
        market_weight: float = 0.2
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            df: Training data with features
            forward_days: Days ahead to predict
            threshold: Return threshold for labeling
            eval_df: Optional evaluation data
            use_composite_labels: Use multi-dimensional composite labels
            trend_weight: Weight for trend alignment in composite labels
            momentum_weight: Weight for momentum in composite labels
            market_weight: Weight for market environment in composite labels
        """
        if df.empty or len(df) < 50:
            raise ValueError("Insufficient data for training")

        # Create labels with composite scoring
        labels = self._create_labels(
            df, forward_days, threshold,
            use_composite=use_composite_labels,
            trend_weight=trend_weight,
            momentum_weight=momentum_weight,
            market_weight=market_weight
        )

        # Select features before preparing
        self.selected_features = self._select_features(df, labels)
        print(f"  Selected {len(self.selected_features)} features out of {len(df.columns)}")

        # Prepare features with selection
        X = self._prepare_features(df, use_selection=True)
        self.feature_names = list(X.columns)

        # Remove rows where forward returns couldn't be calculated (last forward_days rows)
        valid_idx = ~labels.isna()
        X = X[valid_idx]
        labels = labels[valid_idx]

        if len(X) < 30:
            raise ValueError("Insufficient valid samples for training")

        # Convert labels: -1->0, 0->1, 1->2 for CatBoost
        labels = (labels + 1).astype(int)

        # Calculate class weights based on inverse frequency (with square root damping)
        # This is more aggressive than balanced but less than pure inverse
        class_counts = labels.value_counts()
        total = len(labels)
        n_classes = len(class_counts)

        # Custom class weights: sqrt(n/ck) gives moderate boost to minority classes
        class_weights = {}
        for cls in range(n_classes):
            if cls in class_counts.index:
                # Inverse frequency with sqrt damping for less aggressive weighting
                weight = np.sqrt(total / (n_classes * class_counts[cls]))
                class_weights[cls] = weight

        # Cap extreme weights to avoid overfitting to minority class
        max_weight = 5.0
        for cls in class_weights:
            class_weights[cls] = min(class_weights[cls], max_weight)

        print(f"  Class weights: {class_weights}")

        # Train model with custom class weights for balance
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=False,
            loss_function='MultiClass',
            class_weights=class_weights  # Custom weights instead of auto
        )

        train_data = X
        train_labels = labels  # Already converted to 0,1,2

        if eval_df is not None and not eval_df.empty:
            eval_labels = self._create_labels(
                eval_df, forward_days, threshold,
                use_composite=use_composite_labels,
                trend_weight=trend_weight,
                momentum_weight=momentum_weight,
                market_weight=market_weight
            )
            eval_X = self._prepare_features(eval_df, use_selection=True)
            eval_valid_idx = ~eval_labels.isna()
            eval_X_valid = eval_X[eval_valid_idx]
            eval_labels_valid = (eval_labels[eval_valid_idx] + 1).astype(int)

            # Only use eval_set if we have valid samples
            if len(eval_X_valid) > 0:
                self.model.fit(
                    train_data, train_labels,
                    eval_set=(eval_X_valid, eval_labels_valid),
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                self.model.fit(train_data, train_labels, verbose=False)
        else:
            self.model.fit(train_data, train_labels, verbose=False)

        # Calculate training metrics
        train_pred = self.model.predict(train_data)
        train_accuracy = (train_pred.flatten() == train_labels.values).mean()

        return {
            'train_accuracy': train_accuracy,
            'train_samples': len(train_data),
            'feature_count': len(self.feature_names),
            'label_distribution': {
                'buy': int((train_labels == 2).sum()),
                'hold': int((train_labels == 1).sum()),
                'sell': int((train_labels == 0).sum())
            }
        }

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Predict trading action for latest data.

        Returns:
            Tuple of (action, confidence)
            action: 1 (buy), 0 (hold), -1 (sell)
            confidence: probability of the predicted class
        """
        if self.model is None:
            raise ValueError("Model not trained")

        X = self._prepare_features(df)

        # Use only the latest row
        X_latest = X.iloc[[-1]]

        # Predict
        prediction = self.model.predict(X_latest)
        pred_class = int(prediction[0][0]) - 1  # Convert 0,1,2 back to -1,0,1

        # Get probability
        probabilities = self.model.predict_proba(X_latest)[0]
        confidence = float(probabilities[pred_class])

        return pred_class, confidence

    def predict_proba(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get prediction probabilities for all classes."""
        if self.model is None:
            raise ValueError("Model not trained")

        X = self._prepare_features(df)
        X_latest = X.iloc[[-1]]

        probabilities = self.model.predict_proba(X_latest)[0]

        # Handle case where only 2 classes were learned
        if len(probabilities) == 2:
            return {
                'sell_probability': float(probabilities[0]),
                'hold_probability': 0.0,
                'buy_probability': float(probabilities[1])
            }
        else:
            return {
                'sell_probability': float(probabilities[0]),
                'hold_probability': float(probabilities[1]),
                'buy_probability': float(probabilities[2])
            }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained")

        importance = self.model.get_feature_importance()
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save(self, path: str) -> None:
        """Save model to file."""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(model_data, path)

    def load(self, path: str) -> None:
        """Load model from file."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', self.config)


# Global instance
_model: Optional[StockTradingModel] = None


def get_model() -> StockTradingModel:
    """Get global model instance."""
    global _model
    if _model is None:
        _model = StockTradingModel()
    return _model
