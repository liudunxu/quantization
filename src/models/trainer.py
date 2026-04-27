"""CatBoost model for stock trading decisions."""

from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ..utils.config import get_config

# Feature categories for market-aware selection
FEATURE_CATEGORIES = {
    "universal": [
        # Core technical indicators that work across all markets
        "returns",
        "momentum_",
        "rsi",
        "macd",
        "volume_ratio",
        "ma_",
        "volatility",
        "atr_",
        "bb_",
        "stoch_",
        "cci",
        "adx",
        "mfi",
        "turnover",
        "drawdown",
    ],
    "a_share": [
        # A-share specific features
        "net_flow",
        "institutional_ratio",
        "main_net_flow",
        "super_large",
        "large_net_flow",
        "medium_net_flow",
        "small_net_flow",
        "huanshou",
        "换手",  # turnover in Chinese
    ],
    "hk": [
        # HK-specific features (mostly index-related)
        "hsi",
        "hang_seng",
        "hk_",
    ],
    "us": [
        # US-specific features
        "sp_",
        "nasdaq",
        "dow_",
        "^gspc",
        "qqq",
    ],
    "market": [
        # Market-wide features (index, correlation, beta)
        "index_",
        "market_",
        "alpha",
        "beta",
        "corr",
        "sector_",
    ],
    "fundamental": [
        # Fundamental features
        "pe_",
        "pb_",
        "roe_",
        "revenue_",
        "debt_",
        "profit_",
        "growth_",
        "dividend",
        "book_",
        "asset_",
    ],
}


class StockTradingModel:
    """CatBoost model for stock trading decisions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ModelTrainer with configuration."""
        self.config = config or get_config().get_section("model").get("catboost", {})
        self.iterations = self.config.get("iterations", 500)
        self.depth = self.config.get("depth", 6)
        self.learning_rate = self.config.get("learning_rate", 0.03)
        self.l2_leaf_reg = self.config.get("l2_leaf_reg", 3)
        self.random_seed = self.config.get("random_seed", 42)
        self.n_estimators = self.config.get(
            "n_estimators", 3
        )  # Number of models in ensemble
        self.model: Optional[CatBoostClassifier] = None
        self.models: List[CatBoostClassifier] = []  # Ensemble of models
        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None
        self.max_features: int = 60  # Reduced to prevent overfitting
        self.market_type: Optional[str] = None  # Track market type

    def _detect_market_type(self, df: pd.DataFrame) -> str:
        """Detect market type from available features."""
        columns = df.columns.tolist()

        # Check for A-share money flow features
        if any("net_flow" in c.lower() for c in columns):
            return "a_share"

        # Check for HK features
        if any("hsi" in c.lower() or "hang_seng" in c.lower() for c in columns):
            return "hk"

        # Check for US features
        if any(
            c.lower().startswith("sp_") or "nasdaq" in c.lower() or "^gspc" in c.lower()
            for c in columns
        ):
            return "us"

        return "a_share"  # Default

    def _get_feature_categories(self, feature: str, columns: List[str]) -> List[str]:
        """Determine which categories a feature belongs to."""
        categories = []
        feature_lower = feature.lower()

        for category, patterns in FEATURE_CATEGORIES.items():
            for pattern in patterns:
                if pattern.lower() in feature_lower:
                    categories.append(category)
                    break

        return categories if categories else ["other"]

    def _select_features(
        self, df: pd.DataFrame, labels: pd.Series, market_type: str = None
    ) -> List[str]:
        """Select top features based on importance, diversity, and market applicability.

        Uses a market-aware multi-stage approach:
        1. Detect market type if not provided
        2. Score features by category relevance for market
        3. Quick preliminary model to get feature importance
        4. Use IC-based selection for model-agnostic feature evaluation
        5. Select diverse features across categories
        """
        # Drop non-feature columns
        drop_cols = [
            "date",
            "stock_code",
            "sector",
            "industry",
            "close",
            "open",
            "high",
            "low",
            "volume",
        ]
        drop_cols = [c for c in drop_cols if c in df.columns]
        feature_df = df.drop(columns=drop_cols, errors="ignore")

        # Remove object columns
        object_cols = [
            c
            for c in feature_df.columns
            if feature_df[c].dtype == "object" or feature_df[c].dtype == "str"
        ]
        feature_df = feature_df.drop(columns=object_cols, errors="ignore")

        # Detect market type
        if market_type is None:
            market_type = self._detect_market_type(feature_df)
        self.market_type = market_type

        # Define category priority by market
        category_priority = {
            "a_share": ["universal", "a_share", "market", "fundamental"],
            "hk": ["universal", "market", "fundamental"],
            "us": ["universal", "market", "fundamental"],
        }
        priority = category_priority.get(
            market_type, ["universal", "market", "fundamental"]
        )

        # Remove rows where labels are NaN
        valid_idx = ~labels.isna()
        X_quick = feature_df[valid_idx]
        y_quick = labels[valid_idx]

        if len(X_quick) < 30:
            # Not enough samples - use variance-based selection
            variances = feature_df.var()
            low_var_cols = variances[variances < 0.0001].index.tolist()
            feature_df = feature_df.drop(columns=low_var_cols, errors="ignore")
            return feature_df.columns.tolist()[: self.max_features]

        # Stage 1: Quick preliminary model to get feature importance
        quick_model = CatBoostClassifier(
            iterations=50,
            depth=4,
            learning_rate=0.1,
            random_seed=self.random_seed,
            verbose=False,
        )
        quick_model.fit(X_quick, y_quick, verbose=False)

        # Get feature importance from CatBoost
        importance = quick_model.get_feature_importance()
        importance_df = pd.DataFrame(
            {"feature": X_quick.columns, "importance": importance}
        ).sort_values("importance", ascending=False)

        # Stage 2: IC-based feature selection (Information Coefficient)
        # Calculate correlation between each feature and labels
        ic_scores = {}
        for col in X_quick.columns:
            try:
                # Use rank correlation (Spearman) for robustness
                ic = X_quick[col].corr(y_quick, method='spearman')
                if not np.isnan(ic):
                    ic_scores[col] = abs(ic)
            except Exception:
                ic_scores[col] = 0.0

        ic_df = pd.DataFrame(
            {"feature": list(ic_scores.keys()), "ic_score": list(ic_scores.values())}
        )

        # Merge CatBoost importance with IC scores
        importance_df = importance_df.merge(ic_df, on="feature", how="left")
        importance_df["ic_score"] = importance_df["ic_score"].fillna(0.0)

        # Normalize scores to [0, 1] range
        if importance_df["importance"].max() > 0:
            importance_df["importance_norm"] = importance_df["importance"] / importance_df["importance"].max()
        else:
            importance_df["importance_norm"] = 0
            
        if importance_df["ic_score"].max() > 0:
            importance_df["ic_norm"] = importance_df["ic_score"] / importance_df["ic_score"].max()
        else:
            importance_df["ic_norm"] = 0

        # Combined score: 60% CatBoost importance + 40% IC score
        importance_df["raw_score"] = (
            0.6 * importance_df["importance_norm"] + 
            0.4 * importance_df["ic_norm"]
        )

        # Assign categories to each feature
        importance_df["categories"] = importance_df["feature"].apply(
            lambda f: self._get_feature_categories(f, feature_df.columns)
        )

        # Assign market relevance score
        def get_market_score(categories):
            """Calculate market relevance score for feature categories."""
            if "other" in categories:
                return 0.5  # Unknown features get medium score
            # Higher score for more relevant categories
            scores = []
            for cat in categories:
                if cat in priority:
                    scores.append((len(priority) - priority.index(cat)) / len(priority))
                else:
                    scores.append(0.3)
            return max(scores) if scores else 0.5

        importance_df["market_score"] = importance_df["categories"].apply(
            get_market_score
        )
        importance_df["combined_score"] = (
            importance_df["raw_score"] * importance_df["market_score"]
        )

        # Sort by combined score
        importance_df = importance_df.sort_values("combined_score", ascending=False)

        # Select features with diversity across categories
        selected = []
        category_count = {cat: 0 for cat in priority}
        max_per_category = (
            self.max_features // len(priority) + 5
        )  # Allow some imbalance

        for _, row in importance_df.iterrows():
            feature = row["feature"]
            cats = row["categories"]

            # Check if we should include this feature
            # Prefer features from under-represented categories
            can_add = False
            for cat in cats:
                if cat in priority and category_count.get(cat, 0) < max_per_category:
                    can_add = True
                    break

            if can_add or len(selected) < 20:  # Always allow first 20
                selected.append(feature)
                for cat in cats:
                    category_count[cat] = category_count.get(cat, 0) + 1

            if len(selected) >= self.max_features:
                break

        feature_df = feature_df[selected]

        # Remove features with low variance (near-constant)
        variances = feature_df.var()
        low_var_cols = variances[variances < 0.0001].index.tolist()
        feature_df = feature_df.drop(columns=low_var_cols, errors="ignore")

        # Remove features highly correlated with each other
        if len(feature_df.columns) > 1:
            corr_matrix = feature_df.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [
                column for column in upper_tri.columns if any(upper_tri[column] > 0.95)
            ]
            feature_df = feature_df.drop(columns=to_drop, errors="ignore")

        return feature_df.columns.tolist()

    def _prepare_features(
        self, df: pd.DataFrame, use_selection: bool = True
    ) -> pd.DataFrame:
        """Prepare features for modeling."""
        # Drop non-feature columns
        drop_cols = ["date", "stock_code", "sector", "industry"]
        drop_cols = [c for c in drop_cols if c in df.columns]

        feature_df = df.drop(columns=drop_cols, errors="ignore")

        # Remove any remaining object or string columns
        for col in feature_df.columns:
            if feature_df[col].dtype == "object" or feature_df[col].dtype == "str":
                feature_df = feature_df.drop(columns=[col])

        # Apply feature selection if trained
        if use_selection and self.selected_features is not None:
            available_features = [
                f for f in self.selected_features if f in feature_df.columns
            ]
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
        market_weight: float = 0.2,
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
        future_returns = df["close"].shift(-forward_days) / df["close"] - 1

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
        if "ma_bullish_arrange" in df.columns and "ma_bearish_arrange" in df.columns:
            trend_score[df["ma_bullish_arrange"] == 1] = 1.0
            trend_score[df["ma_bearish_arrange"] == 1] = -1.0

        # 3. Momentum score (RSI, MACD)
        momentum_score = pd.Series(0.0, index=df.index)
        if "rsi" in df.columns:
            # RSI > 60 is bullish, < 40 is bearish
            momentum_score[df["rsi"] > 60] += 0.5
            momentum_score[df["rsi"] < 40] -= 0.5
        if "macd_hist" in df.columns:
            # MACD histogram positive is bullish
            momentum_score[df["macd_hist"] > 0] += 0.5
            momentum_score[df["macd_hist"] < 0] -= 0.5

        # Normalize momentum to -1, 0, 1
        momentum_score = momentum_score.clip(-1.0, 1.0)

        # 4. Market score (if available)
        market_score = pd.Series(0.0, index=df.index)
        if "index_returns" in df.columns:
            market_score[df["index_returns"] > 0.01] = 0.5
            market_score[df["index_returns"] < -0.01] = -0.5

        # ========== Combine into composite signal ==========
        # Weighted combination
        return_weight = 1.0 - trend_weight - momentum_weight - market_weight
        composite = (
            return_score * return_weight
            + trend_score * trend_weight
            + momentum_score * momentum_weight
            + market_score * market_weight
        )

        # ========== Create final labels ==========
        # Buy: composite score >= buy_threshold AND future return is positive
        # Sell: composite score <= sell_threshold AND future return is negative
        # Hold: everything else
        #
        # Thresholds adjusted for more balanced label distribution

        labels = pd.Series(0, index=df.index)
        buy_threshold = 0.05  # Lowered for more buy signals
        sell_threshold = -0.05  # Lowered for more sell signals

        buy_condition = (composite >= buy_threshold) & (
            future_returns > threshold * 0.2
        )
        sell_condition = (composite <= sell_threshold) & (
            future_returns < -threshold * 0.2
        )

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
        market_weight: float = 0.2,
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
            df,
            forward_days,
            threshold,
            use_composite=use_composite_labels,
            trend_weight=trend_weight,
            momentum_weight=momentum_weight,
            market_weight=market_weight,
        )

        # Select features before preparing
        self.selected_features = self._select_features(df, labels)

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

        # Ensure all 3 classes are present (add synthetic samples if needed)
        unique_classes = labels.unique()
        if len(unique_classes) < 3:
            # Add one synthetic sample for each missing class
            missing_classes = [c for c in [0, 1, 2] if c not in unique_classes]
            for missing_class in missing_classes:
                # Create a synthetic sample by duplicating first row
                synthetic_X = X.iloc[[0]].copy()
                synthetic_label = pd.Series(
                    [missing_class], index=[f"synthetic_{missing_class}"]
                )
                X = pd.concat([X, synthetic_X])
                labels = pd.concat([labels, synthetic_label])

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

        train_data = X
        train_labels = labels  # Already converted to 0,1,2

        # Prepare eval data if available
        eval_data = None
        eval_labels_converted = None
        if eval_df is not None and not eval_df.empty:
            eval_labels = self._create_labels(
                eval_df,
                forward_days,
                threshold,
                use_composite=use_composite_labels,
                trend_weight=trend_weight,
                momentum_weight=momentum_weight,
                market_weight=market_weight,
            )
            eval_X = self._prepare_features(eval_df, use_selection=True)
            eval_valid_idx = ~eval_labels.isna()
            eval_X_valid = eval_X[eval_valid_idx]
            eval_labels_valid = (eval_labels[eval_valid_idx] + 1).astype(int)

            # Ensure eval data also has all 3 classes
            eval_unique = eval_labels_valid.unique()
            if len(eval_unique) < 3:
                eval_missing = [c for c in [0, 1, 2] if c not in eval_unique]
                for missing_class in eval_missing:
                    synthetic_eval_X = eval_X_valid.iloc[[0]].copy()
                    synthetic_eval_label = pd.Series(
                        [missing_class], index=[f"synthetic_eval_{missing_class}"]
                    )
                    eval_X_valid = pd.concat([eval_X_valid, synthetic_eval_X])
                    eval_labels_valid = pd.concat(
                        [eval_labels_valid, synthetic_eval_label]
                    )

            if len(eval_X_valid) > 0:
                eval_data = eval_X_valid
                eval_labels_converted = eval_labels_valid

        # Train ensemble of models with hyperparameter diversity
        self.models = []
        n_samples = len(train_data)
        
        # Define diverse hyperparameter configurations for ensemble
        hyperparam_configs = [
            {"depth": self.depth, "learning_rate": self.learning_rate, "l2_leaf_reg": self.l2_leaf_reg},
            {"depth": max(3, self.depth - 1), "learning_rate": self.learning_rate * 1.2, "l2_leaf_reg": self.l2_leaf_reg * 0.8},
            {"depth": min(8, self.depth + 1), "learning_rate": self.learning_rate * 0.8, "l2_leaf_reg": self.l2_leaf_reg * 1.2},
            {"depth": self.depth, "learning_rate": self.learning_rate * 0.7, "l2_leaf_reg": self.l2_leaf_reg * 1.5},
            {"depth": max(3, self.depth - 1), "learning_rate": self.learning_rate * 1.3, "l2_leaf_reg": self.l2_leaf_reg * 0.6},
        ]

        for i in range(self.n_estimators):
            # Use different random seed for each model
            model_seed = self.random_seed + i * 111  # Spread out seeds
            
            # Get hyperparameter configuration for this model
            config_idx = i % len(hyperparam_configs)
            hp_config = hyperparam_configs[config_idx]

            # Bootstrap sampling for diversity (sample with replacement)
            np.random.seed(model_seed)
            bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_data = train_data.iloc[bootstrap_idx]
            bootstrap_labels = train_labels.iloc[bootstrap_idx]

            # Ensure bootstrap sample has all 3 classes
            bootstrap_unique = bootstrap_labels.unique()
            if len(bootstrap_unique) < 3:
                # Add one sample for each missing class from original data
                for missing_class in [0, 1, 2]:
                    if missing_class not in bootstrap_unique:
                        # Find a sample with this class in original data
                        class_idx = train_labels[train_labels == missing_class].index[0]
                        sample_idx = train_data.index.get_loc(class_idx)
                        # Add it to bootstrap data
                        bootstrap_data = pd.concat(
                            [bootstrap_data, train_data.iloc[[sample_idx]]]
                        )
                        bootstrap_labels = pd.concat(
                            [bootstrap_labels, pd.Series([missing_class])]
                        )

            model = CatBoostClassifier(
                iterations=self.iterations,
                depth=hp_config["depth"],
                learning_rate=hp_config["learning_rate"],
                l2_leaf_reg=hp_config["l2_leaf_reg"],
                random_seed=model_seed,
                verbose=False,
                loss_function="MultiClass",
                class_weights=class_weights,
            )

            if eval_data is not None:
                model.fit(
                    bootstrap_data,
                    bootstrap_labels,
                    eval_set=(eval_data, eval_labels_converted),
                    early_stopping_rounds=50,
                    verbose=False,
                )
            else:
                model.fit(bootstrap_data, bootstrap_labels, verbose=False)

            self.models.append(model)

        # Use first model as primary for backwards compatibility
        self.model = self.models[0]

        # Calculate training metrics (from first model for consistency)
        train_pred = self.model.predict(train_data)
        train_accuracy = (train_pred.flatten() == train_labels.values).mean()

        return {
            "train_accuracy": train_accuracy,
            "train_samples": len(train_data),
            "feature_count": len(self.feature_names),
            "label_distribution": {
                "buy": int((train_labels == 2).sum()),
                "hold": int((train_labels == 1).sum()),
                "sell": int((train_labels == 0).sum()),
            },
        }

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Predict trading action for latest data using ensemble voting.

        Returns:
            Tuple of (action, confidence)
            action: 1 (buy), 0 (hold), -1 (sell)
            confidence: probability of the predicted class (averaged across ensemble)
        """
        if len(self.models) == 0:
            raise ValueError("Model not trained")

        X = self._prepare_features(df)

        # Use only the latest row
        X_latest = X.iloc[[-1]]

        # Ensemble prediction: average probabilities across all models
        all_probabilities = []
        for model in self.models:
            probs = model.predict_proba(X_latest)[0]
            all_probabilities.append(probs)

        # Average probabilities (soft voting)
        avg_probabilities = np.mean(all_probabilities, axis=0)
        pred_class = int(np.argmax(avg_probabilities))  # Class with highest avg prob
        confidence = float(avg_probabilities[pred_class])

        # Convert 0,1,2 back to -1,0,1
        pred_class = pred_class - 1

        return pred_class, confidence

    def predict_proba(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get prediction probabilities for all classes (averaged across ensemble)."""
        if len(self.models) == 0:
            raise ValueError("Model not trained")

        X = self._prepare_features(df)
        X_latest = X.iloc[[-1]]

        # Ensemble: average probabilities
        all_probabilities = []
        for model in self.models:
            probs = model.predict_proba(X_latest)[0]
            all_probabilities.append(probs)

        avg_probabilities = np.mean(all_probabilities, axis=0)

        # Handle case where only 2 classes were learned
        if len(avg_probabilities) == 2:
            return {
                "sell_probability": float(avg_probabilities[0]),
                "hold_probability": 0.0,
                "buy_probability": float(avg_probabilities[1]),
            }
        else:
            return {
                "sell_probability": float(avg_probabilities[0]),
                "hold_probability": float(avg_probabilities[1]),
                "buy_probability": float(avg_probabilities[2]),
            }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (averaged across ensemble)."""
        if len(self.models) == 0:
            raise ValueError("Model not trained")

        # Average importance across all models in ensemble
        all_importance = []
        for model in self.models:
            imp = model.get_feature_importance()
            all_importance.append(imp)

        avg_importance = np.mean(all_importance, axis=0)

        return pd.DataFrame(
            {"feature": self.feature_names, "importance": avg_importance}
        ).sort_values("importance", ascending=False)

    def save(self, path: str) -> None:
        """Save ensemble model to file."""
        model_data = {
            "models": self.models,
            "feature_names": self.feature_names,
            "config": self.config,
        }
        joblib.dump(model_data, path)

    def load(self, path: str) -> None:
        """Load ensemble model from file."""
        model_data = joblib.load(path)
        self.models = model_data["models"]
        self.model = self.models[0] if self.models else None
        self.feature_names = model_data["feature_names"]
        self.config = model_data.get("config", self.config)


# Global instance
_model: Optional[StockTradingModel] = None


def get_model() -> StockTradingModel:
    """Get global model instance."""
    global _model
    if _model is None:
        _model = StockTradingModel()
    return _model
