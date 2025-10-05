"""
Market Regime Classification Pipeline

This module provides classes and functions for market regime classification
using both machine learning models and pre-classified hierarchical regimes.

Updated to support the 1D momentum-based regime classification approach.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os
from typing import Tuple, Dict, List, Optional


class RegimeFeatureEnginer:
    """Feature engineering for market regime classification"""

    def __init__(self, lookback_windows: List[int] = [5, 10, 20, 50]):
        self.lookback_windows = lookback_windows

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for market regime classification

        Args:
            df: DataFrame with OHLCV and technical indicators

        Returns:
            DataFrame with additional regime classification features
        """
        df = df.copy()

        # Price-based features
        for window in self.lookback_windows:
            # Returns and volatility
            df[f'return_{window}d'] = df['Close'].pct_change(window)
            df[f'volatility_{window}d'] = df['Close'].pct_change().rolling(window).std()

            # Trend strength
            df[f'trend_strength_{window}d'] = (df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)

            # Price position relative to moving averages
            sma_col = f'sma_{window}d' if f'sma_{window}d' in df.columns else None
            if sma_col:
                df[f'price_vs_sma_{window}d'] = (df['Close'] - df[sma_col]) / df[sma_col]

        # Technical indicator features
        if 'rsi_14d' in df.columns:
            df['rsi_normalized'] = (df['rsi_14d'] - 50) / 50
            df['rsi_momentum'] = df['rsi_14d'].diff()

        if 'macd_hist_12_26' in df.columns:
            df['macd_trend'] = np.where(df['macd_hist_12_26'] > 0, 1, -1)
            df['macd_momentum'] = df['macd_hist_12_26'].diff()

        # Volume features
        if 'Volume' in df.columns:
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

        # Fractal dimension features (if available)
        if 'fd_14d' in df.columns:
            df['fd_trend'] = df['fd_14d'].diff()
            df['fd_normalized'] = (df['fd_14d'] - df['fd_14d'].rolling(50).mean()) / df['fd_14d'].rolling(50).std()

        return df

    def create_regime_labels(self, df: pd.DataFrame,
                           bull_threshold: float = 0.02,
                           bear_threshold: float = -0.015,
                           lookforward_days: int = 10,
                           volatility_threshold: float = 0.03) -> pd.DataFrame:
        """
        Create market regime labels based on future returns and volatility

        Args:
            df: DataFrame with price data
            bull_threshold: Minimum return for bull market classification
            bear_threshold: Maximum return for bear market classification
            lookforward_days: Days to look forward for regime determination
            volatility_threshold: Volatility threshold for regime classification

        Returns:
            DataFrame with regime labels (0: Bear, 1: Sideways, 2: Bull)
        """
        df = df.copy()

        # Calculate future returns
        df['future_return'] = df['Close'].shift(-lookforward_days).pct_change(lookforward_days)

        # Calculate rolling volatility
        df['rolling_volatility'] = df['Close'].pct_change().rolling(lookforward_days).std()

        # Define regimes
        conditions = [
            (df['future_return'] < bear_threshold) & (df['rolling_volatility'] > volatility_threshold),
            (df['future_return'] > bull_threshold) & (df['rolling_volatility'] < volatility_threshold * 2),
        ]

        choices = [0, 2]  # Bear, Bull
        df['regime'] = np.select(conditions, choices, default=1)  # Default to Sideways

        # Add regime smoothing to reduce noise
        df['regime_smooth'] = df['regime'].rolling(5, center=True).apply(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[2], raw=False
        )

        return df


class PreClassifiedRegimeLoader:
    """Loader for pre-classified regime data from enhanced regime classification"""

    def __init__(self, data_path: str = './data/BTCUSD_2023_1min_enhanced_regimes_1D_momentum.csv'):
        self.data_path = data_path
        self.metadata_path = './data/enhanced_regime_classification_metadata.json'

    def load_regime_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Load pre-classified regime data and metadata

        Returns:
            Tuple of (dataframe with regime column, metadata dict)
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Pre-classified regime data not found: {self.data_path}")

        # Load data
        df = pd.read_csv(self.data_path)

        # Handle timestamp column
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'Timestamp'
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col).sort_index()

        # Clean up unnecessary columns
        cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
        df = df.drop(columns=cols_to_drop)

        # Load metadata if available
        metadata = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

        return df, metadata

    def validate_regime_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe contains proper regime classifications

        Args:
            df: DataFrame to validate

        Returns:
            True if valid regime data
        """
        if 'regime' not in df.columns:
            return False

        # Check regime values are in expected range
        regime_values = df['regime'].unique()
        expected_regimes = {0, 1, 2}  # Bear, Sideways, Bull

        if not set(regime_values).issubset(expected_regimes):
            return False

        # Check we have all three regimes represented
        if len(set(regime_values)) < 3:
            print(f"Warning: Only {len(set(regime_values))} regimes found: {sorted(regime_values)}")

        return True


class RegimeClassifierPipeline:
    """Complete pipeline for market regime classification"""

    def __init__(self, model_type: str = 'RandomForest', use_preclassified: bool = False):
        self.model_type = model_type
        self.use_preclassified = use_preclassified
        self.model = None
        self.feature_cols = None
        self.feature_engineer = RegimeFeatureEnginer()
        self.scaler = StandardScaler()
        self.preclassified_loader = PreClassifiedRegimeLoader()

        # Define model configurations
        self.model_configs = {
            'RandomForest': Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
            ]),
            'GradientBoosting': Pipeline([
                ('scaler', StandardScaler()),
                ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42))
            ]),
            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42))
            ])
        }

    def prepare_training_data(self, df: pd.DataFrame,
                            target_col: str = 'regime',
                            test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Prepare data for ML training with time-aware split

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            test_size: Fraction of data for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_cols)
        """
        # Engineer features
        df_features = self.feature_engineer.engineer_features(df)
        df_labeled = self.feature_engineer.create_regime_labels(df_features)

        # Select feature columns (exclude target and derived columns)
        exclude_cols = [target_col, 'regime_smooth', 'future_return', 'rolling_volatility',
                       'Open', 'High', 'Low', 'Close', 'Volume']

        feature_cols = [col for col in df_labeled.columns
                       if col not in exclude_cols and not col.startswith('future_')]

        # Remove rows with NaN values
        df_clean = df_labeled.dropna(subset=feature_cols + [target_col])

        X = df_clean[feature_cols]
        y = df_clean[target_col]

        # Time-aware split (no shuffling for time series)
        split_idx = int(len(df_clean) * (1 - test_size))

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.feature_cols = feature_cols

        return X_train, X_test, y_train, y_test, feature_cols

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Train the regime classification model

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary with training metrics
        """
        if self.model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = self.model_configs[self.model_type]

        # Train model
        self.model.fit(X_train, y_train)

        # Cross-validation score
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=tscv, scoring='accuracy')

        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_type': self.model_type
        }

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate the trained model

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, target_names=['Bear', 'Sideways', 'Bull'])
        }

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict market regime for new data

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.use_preclassified:
            # Use pre-classified regimes if available
            if 'regime' in df.columns:
                predictions = df['regime'].values
                # Create dummy probabilities (high confidence for the classified regime)
                n_samples = len(predictions)
                probabilities = np.zeros((n_samples, 3))
                for i, regime in enumerate(predictions):
                    probabilities[i, int(regime)] = 0.9
                    # Distribute remaining probability
                    for j in range(3):
                        if j != int(regime):
                            probabilities[i, j] = 0.05
                return predictions, probabilities
            else:
                raise ValueError("Pre-classified mode selected but no 'regime' column found in data")

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Engineer features
        df_features = self.feature_engineer.engineer_features(df)
        X = df_features[self.feature_cols].fillna(method='ffill').fillna(0)

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return predictions, probabilities

    def save_model(self, filepath: str):
        """Save the trained model and metadata"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Save model
        model_path = filepath.replace('.pkl', '_model.pkl')
        joblib.dump(self.model, model_path)

        # Save feature columns and metadata
        metadata = {
            'model_type': self.model_type,
            'feature_cols': self.feature_cols,
            'lookback_windows': self.feature_engineer.lookback_windows
        }

        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved: {model_path}")
        print(f"Metadata saved: {metadata_path}")

    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model and metadata"""
        # Load metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(model_type=metadata['model_type'])
        instance.feature_cols = metadata['feature_cols']
        instance.feature_engineer.lookback_windows = metadata['lookback_windows']

        # Load model
        model_path = filepath.replace('.pkl', '_model.pkl')
        instance.model = joblib.load(model_path)

        return instance


def create_preclassified_pipeline() -> RegimeClassifierPipeline:
    """
    Create a pipeline configured for pre-classified regime data

    Returns:
        RegimeClassifierPipeline configured for pre-classified regimes
    """
    pipeline = RegimeClassifierPipeline(use_preclassified=True)

    # Load and validate pre-classified data
    try:
        df, metadata = pipeline.preclassified_loader.load_regime_data()

        if pipeline.preclassified_loader.validate_regime_data(df):
            print("✅ Pre-classified regime data loaded and validated")
            print(f"Dataset shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")

            # Print regime distribution
            regime_counts = df['regime'].value_counts().sort_index()
            regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}

            print("Regime distribution:")
            for regime, count in regime_counts.items():
                pct = count / len(df) * 100
                print(f"  {regime_names.get(regime, regime)}: {count:,} ({pct:.1f}%)")

            if metadata:
                print(f"Classification method: {metadata.get('optimal_method', 'Unknown')}")
                print(f"Classification timeframe: {metadata.get('optimal_timeframe', 'Unknown')}")

        else:
            print("❌ Invalid pre-classified regime data")

    except Exception as e:
        print(f"Error loading pre-classified data: {e}")
        print("Pipeline created but will need data loaded manually")

    return pipeline


def train_multiple_models(df: pd.DataFrame, test_size: float = 0.3, use_preclassified: bool = False) -> Dict[str, Dict]:
    """
    Train and compare multiple regime classification models

    Args:
        df: DataFrame with OHLCV and technical indicators
        test_size: Fraction of data for testing
        use_preclassified: Whether to use pre-classified regimes

    Returns:
        Dictionary with results for each model type
    """
    if use_preclassified:
        print("Using pre-classified regime approach - skipping ML training")
        pipeline = create_preclassified_pipeline()
        return {'PreClassified': {'pipeline': pipeline}}

    model_types = ['RandomForest', 'GradientBoosting', 'SVM']
    results = {}

    # Prepare data once
    pipeline = RegimeClassifierPipeline('RandomForest')
    X_train, X_test, y_train, y_test, feature_cols = pipeline.prepare_training_data(df, test_size=test_size)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of features: {len(feature_cols)}")

    for model_type in model_types:
        print(f"\nTraining {model_type}...")

        # Create and train pipeline
        pipeline = RegimeClassifierPipeline(model_type)
        pipeline.feature_cols = feature_cols

        # Train
        train_metrics = pipeline.train(X_train, y_train)

        # Evaluate
        eval_metrics = pipeline.evaluate(X_test, y_test)

        # Store results
        results[model_type] = {
            'pipeline': pipeline,
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics
        }

        print(f"{model_type} - Accuracy: {eval_metrics['accuracy']:.3f}")
        print(f"{model_type} - CV Score: {train_metrics['cv_mean']:.3f} ± {train_metrics['cv_std']:.3f}")

    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['eval_metrics']['accuracy'])
    print(f"\nBest model: {best_model} (Accuracy: {results[best_model]['eval_metrics']['accuracy']:.3f})")

    return results


if __name__ == "__main__":
    # Example usage
    print("Market Regime Classification Pipeline")
    print("====================================")

    # This would typically load real data
    # df = pd.read_csv('path_to_your_data.csv')
    # results = train_multiple_models(df)

    print("Pipeline components defined and ready to use.")