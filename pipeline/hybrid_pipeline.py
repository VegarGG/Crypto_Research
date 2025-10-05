"""
Hybrid ML + Rule-Based Trading Pipeline

This module combines market regime classification with rule-based trading strategies
to create a comprehensive trading system.

Updated to support pre-classified 1D momentum-based regime data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import joblib
import json
from datetime import datetime

from regime_classifier import RegimeClassifierPipeline, create_preclassified_pipeline
from trading_strategies import (
    StrategyManager, BullMarketStrategy, BearMarketStrategy,
    SidewaysMarketStrategy, TradeSignal, Trade
)


class HybridTradingPipeline:
    """Main pipeline combining ML regime classification with rule-based strategies"""

    def __init__(self,
                 regime_model_path: Optional[str] = None,
                 strategy_config_path: Optional[str] = None,
                 use_preclassified_regimes: bool = True):
        """
        Initialize hybrid trading pipeline

        Args:
            regime_model_path: Path to saved regime classification model
            strategy_config_path: Path to strategy configuration file
            use_preclassified_regimes: Whether to use pre-classified 1D momentum regimes
        """
        self.regime_classifier = None
        self.strategy_manager = StrategyManager()
        self.backtest_results = {}
        self.use_preclassified_regimes = use_preclassified_regimes

        # Initialize regime classifier
        if use_preclassified_regimes:
            print("Initializing with pre-classified 1D momentum regime data...")
            self.regime_classifier = create_preclassified_pipeline()
        elif regime_model_path and os.path.exists(regime_model_path):
            self.load_regime_model(regime_model_path)

        if strategy_config_path and os.path.exists(strategy_config_path):
            self.strategy_manager = StrategyManager.load_strategies(strategy_config_path)
        else:
            self._create_default_strategies()

    def _create_default_strategies(self):
        """Create default strategy configurations"""
        # Bull market strategy - aggressive trend following
        bull_strategy = BullMarketStrategy(
            ma_fast=7, ma_slow=20, rsi_threshold=40,
            stop_loss=0.06, take_profit=0.12, trail_stop=0.04
        )
        self.strategy_manager.add_strategy('bull_trend', bull_strategy)

        # Bear market strategy - conservative mean reversion
        bear_strategy = BearMarketStrategy(
            rsi_oversold=25, rsi_overbought=60,
            stop_loss=0.04, take_profit=0.08, bb_deviation=1.0
        )
        self.strategy_manager.add_strategy('bear_reversion', bear_strategy)

        # Sideways market strategy - range trading
        sideways_strategy = SidewaysMarketStrategy(
            bb_period=20, rsi_oversold=30, rsi_overbought=70,
            stop_loss=0.03, take_profit=0.06, bb_threshold=0.01
        )
        self.strategy_manager.add_strategy('sideways_range', sideways_strategy)

    def train_regime_classifier(self, df: pd.DataFrame,
                               model_type: str = 'RandomForest',
                               test_size: float = 0.3) -> Dict:
        """
        Train market regime classification model

        Args:
            df: Training data with OHLCV and technical indicators
            model_type: Type of ML model to use
            test_size: Fraction of data for testing

        Returns:
            Training and evaluation metrics
        """
        if self.use_preclassified_regimes:
            print("Using pre-classified regimes - skipping ML training")
            return {
                'train_metrics': {'cv_mean': 1.0, 'cv_std': 0.0, 'model_type': 'PreClassified'},
                'eval_metrics': {'accuracy': 1.0},
                'feature_cols': ['regime']
            }

        print(f"Training {model_type} regime classifier...")

        self.regime_classifier = RegimeClassifierPipeline(model_type)

        # Prepare training data
        X_train, X_test, y_train, y_test, feature_cols = self.regime_classifier.prepare_training_data(
            df, test_size=test_size
        )

        # Train model
        train_metrics = self.regime_classifier.train(X_train, y_train)

        # Evaluate model
        eval_metrics = self.regime_classifier.evaluate(X_test, y_test)

        print(f"Training completed:")
        print(f"  - CV Score: {train_metrics['cv_mean']:.3f} ± {train_metrics['cv_std']:.3f}")
        print(f"  - Test Accuracy: {eval_metrics['accuracy']:.3f}")

        return {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'feature_cols': feature_cols
        }

    def predict_regime(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict market regime for given data

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.regime_classifier is None:
            raise ValueError("Regime classifier not trained. Call train_regime_classifier() first.")

        return self.regime_classifier.predict(df)

    def generate_signals(self, df: pd.DataFrame) -> Tuple[List[TradeSignal], np.ndarray]:
        """
        Generate trading signals using hybrid approach

        Args:
            df: DataFrame with OHLCV and technical indicators

        Returns:
            Tuple of (all_signals, regime_predictions)
        """
        # Predict market regimes
        regime_predictions, regime_probabilities = self.predict_regime(df)

        # Generate signals from all strategies
        all_signals = []
        strategy_signals = self.strategy_manager.generate_all_signals(df, regime_predictions)

        # Combine signals from all strategies
        for strategy_name, signals in strategy_signals.items():
            all_signals.extend(signals)

        # Sort signals by timestamp
        all_signals.sort(key=lambda x: x.timestamp)

        return all_signals, regime_predictions

    def backtest(self, df: pd.DataFrame,
                initial_capital: float = 10000,
                commission: float = 0.001,
                slippage: float = 0.0001) -> Dict:
        """
        Comprehensive backtesting of the hybrid strategy

        Args:
            df: DataFrame with OHLCV and technical indicators
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)
            slippage: Price slippage per trade (as fraction)

        Returns:
            Dictionary with backtest results
        """
        print(f"Running backtest with ${initial_capital:,.2f} initial capital...")

        # Generate signals
        signals, regime_predictions = self.generate_signals(df)

        # Initialize portfolio tracking
        portfolio = []
        cash = initial_capital
        position = 0
        position_value = 0
        trades = []

        # Track current trade
        current_trade = None

        # Create regime series for analysis
        regime_series = pd.Series(regime_predictions, index=df.index[:len(regime_predictions)])

        # Process each timestamp
        for i, (timestamp, row) in enumerate(df.iterrows()):
            current_price = row['Close']

            # Get regime for this timestamp
            current_regime = regime_series.get(timestamp, 1) if timestamp in regime_series.index else 1

            # Check for signals at this timestamp
            timestamp_signals = [s for s in signals if s.timestamp == timestamp]

            for signal in timestamp_signals:
                signal_price = signal.price * (1 + slippage if signal.action == 'buy' else 1 - slippage)

                if signal.action == 'buy' and position == 0:
                    # Open long position
                    position = (cash * (1 - commission)) / signal_price
                    cash = 0
                    position_value = position * signal_price

                    # Start new trade record
                    current_trade = Trade(
                        entry_time=signal.timestamp,
                        entry_price=signal_price,
                        exit_time=None,
                        exit_price=0,
                        entry_regime=signal.regime,
                        exit_regime=0,
                        pnl=0,
                        return_pct=0,
                        duration=0,
                        strategy_name=signal.strategy_name
                    )

                elif signal.action == 'sell' and position > 0:
                    # Close long position
                    cash = position * signal_price * (1 - commission)

                    # Complete trade record
                    if current_trade:
                        current_trade.exit_time = signal.timestamp
                        current_trade.exit_price = signal_price
                        current_trade.exit_regime = signal.regime
                        current_trade.pnl = cash - initial_capital
                        current_trade.return_pct = (cash - initial_capital) / initial_capital * 100
                        current_trade.duration = (signal.timestamp - current_trade.entry_time).days

                        trades.append(current_trade)
                        current_trade = None

                    position = 0
                    position_value = 0

            # Calculate portfolio value
            if position > 0:
                position_value = position * current_price
                total_value = position_value
            else:
                total_value = cash

            portfolio.append(total_value)

        # Create results DataFrame
        results_df = df.copy()
        results_df['portfolio_value'] = portfolio[:len(df)]
        results_df['regime'] = list(regime_predictions) + [regime_predictions[-1]] * (len(df) - len(regime_predictions))

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            results_df, trades, initial_capital
        )

        # Store results
        backtest_result = {
            'performance_metrics': performance_metrics,
            'trades': trades,
            'signals': signals,
            'regime_predictions': regime_predictions,
            'results_df': results_df
        }

        self.backtest_results[datetime.now().strftime("%Y%m%d_%H%M%S")] = backtest_result

        # Print summary
        print(f"Backtest completed:")
        print(f"  - Final Value: ${performance_metrics['final_value']:,.2f}")
        print(f"  - Total Return: {performance_metrics['total_return']:.2f}%")
        print(f"  - Number of Trades: {performance_metrics['num_trades']}")
        print(f"  - Win Rate: {performance_metrics['win_rate']:.2%}")
        print(f"  - Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
        print(f"  - Max Drawdown: {performance_metrics['max_drawdown']:.2%}")

        return backtest_result

    def _calculate_performance_metrics(self, results_df: pd.DataFrame,
                                     trades: List[Trade],
                                     initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        final_value = results_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100

        # Trade-based metrics
        num_trades = len(trades)
        if num_trades > 0:
            win_rate = np.mean([t.return_pct > 0 for t in trades])
            avg_return = np.mean([t.return_pct for t in trades])
            avg_duration = np.mean([t.duration for t in trades])
        else:
            win_rate = 0
            avg_return = 0
            avg_duration = 0

        # Portfolio-based metrics
        returns = results_df['portfolio_value'].pct_change().dropna()

        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0

        # Drawdown calculation
        rolling_max = results_df['portfolio_value'].cummax()
        drawdown = (results_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Regime-specific analysis
        regime_performance = {}
        for regime in [0, 1, 2]:
            regime_trades = [t for t in trades if t.entry_regime == regime]
            if regime_trades:
                regime_performance[regime] = {
                    'num_trades': len(regime_trades),
                    'win_rate': np.mean([t.return_pct > 0 for t in regime_trades]),
                    'avg_return': np.mean([t.return_pct for t in regime_trades])
                }

        return {
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_duration': avg_duration,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'regime_performance': regime_performance
        }

    def optimize_strategies(self, df: pd.DataFrame,
                          optimization_params: Dict) -> Dict:
        """
        Optimize strategy parameters using historical data

        Args:
            df: Historical data for optimization
            optimization_params: Parameter ranges for each strategy

        Returns:
            Optimized parameters for each strategy
        """
        print("Optimizing strategy parameters...")

        # Get regime predictions for optimization
        regime_predictions, _ = self.predict_regime(df)

        optimization_results = {}

        for strategy_name, strategy in self.strategy_manager.strategies.items():
            if strategy_name in optimization_params:
                print(f"Optimizing {strategy_name}...")

                from trading_strategies import StrategyOptimizer
                optimizer = StrategyOptimizer(type(strategy), regime_predictions)

                result = optimizer.optimize_parameters(df, optimization_params[strategy_name])
                optimization_results[strategy_name] = result

                # Update strategy with best parameters
                strategy.update_parameters(**result['best_parameters'])

                print(f"  Best Sharpe Ratio: {result['best_performance']:.3f}")

        return optimization_results

    def save_pipeline(self, base_path: str):
        """
        Save the complete pipeline

        Args:
            base_path: Base directory for saving pipeline components
        """
        os.makedirs(base_path, exist_ok=True)

        # Save regime classifier
        if self.regime_classifier:
            regime_model_path = os.path.join(base_path, 'regime_classifier.pkl')
            self.regime_classifier.save_model(regime_model_path)

        # Save strategy configurations
        strategy_config_path = os.path.join(base_path, 'strategy_config.json')
        self.strategy_manager.save_strategies(strategy_config_path)

        # Save pipeline metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'regime_model_type': self.regime_classifier.model_type if self.regime_classifier else None,
            'num_strategies': len(self.strategy_manager.strategies)
        }

        metadata_path = os.path.join(base_path, 'pipeline_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Pipeline saved to: {base_path}")

    def load_regime_model(self, model_path: str):
        """Load a trained regime classification model"""
        self.regime_classifier = RegimeClassifierPipeline.load_model(model_path)
        print(f"Regime classifier loaded from: {model_path}")

    @classmethod
    def load_pipeline(cls, base_path: str):
        """
        Load a complete saved pipeline

        Args:
            base_path: Base directory containing pipeline components

        Returns:
            Loaded HybridTradingPipeline instance
        """
        regime_model_path = os.path.join(base_path, 'regime_classifier.pkl')
        strategy_config_path = os.path.join(base_path, 'strategy_config.json')

        return cls(regime_model_path, strategy_config_path)


def run_preclassified_regime_analysis(data_path: str = './data/BTCUSD_2023_1min_enhanced_regimes_1D_momentum.csv',
                                     initial_capital: float = 10000) -> Dict:
    """
    Run hybrid strategy analysis with pre-classified 1D momentum regime data

    Args:
        data_path: Path to pre-classified regime dataset
        initial_capital: Starting capital for backtest

    Returns:
        Dictionary with analysis results
    """
    print("Running hybrid strategy analysis with pre-classified 1D momentum regimes...")
    print("=" * 80)

    try:
        # Load pre-classified regime data
        df = pd.read_csv(data_path)
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'Timestamp'
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col).sort_index()

        # Clean up unnecessary columns
        cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
        df = df.drop(columns=cols_to_drop)

        print(f"Loaded pre-classified regime data: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        # Validate regime data
        if 'regime' not in df.columns:
            raise ValueError("No 'regime' column found in the dataset")

        regime_counts = df['regime'].value_counts().sort_index()
        regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}

        print("Regime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(df) * 100
            print(f"  {regime_names.get(regime, regime)}: {count:,} ({pct:.1f}%)")

        # Create hybrid pipeline with pre-classified regimes
        pipeline = HybridTradingPipeline(use_preclassified_regimes=True)

        # Run backtest
        backtest_result = pipeline.backtest(df, initial_capital=initial_capital)

        # Additional analysis
        results = {
            'data_info': {
                'shape': df.shape,
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'regime_distribution': dict(regime_counts)
            },
            'backtest_result': backtest_result,
            'pipeline': pipeline
        }

        print("\nAnalysis completed successfully!")
        return results

    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def run_multi_timeframe_analysis(data_dir: str,
                                timeframes: List[str] = ['1min', '15min', '30min', '1day'],
                                initial_capital: float = 10000,
                                use_preclassified: bool = True) -> Dict:
    """
    Run hybrid strategy analysis across multiple timeframes

    Args:
        data_dir: Directory containing cleaned datasets
        timeframes: List of timeframes to analyze
        initial_capital: Starting capital for each backtest
        use_preclassified: Whether to use pre-classified regimes

    Returns:
        Dictionary with results for each timeframe
    """
    if use_preclassified:
        print("Running analysis with pre-classified 1D momentum regime data...")
        return run_preclassified_regime_analysis(
            data_path='./data/BTCUSD_2023_1min_enhanced_regimes_1D_momentum.csv',
            initial_capital=initial_capital
        )

    print("Running multi-timeframe hybrid strategy analysis...")
    print("=" * 60)

    results = {}

    for timeframe in timeframes:
        print(f"\nAnalyzing {timeframe} timeframe...")

        # Load data
        if timeframe == '1min':
            filename = 'BTCUSD_2023_1min_cleaned.csv'
        else:
            filename = f'BTCUSD_2023_{timeframe}_cleaned.csv'

        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"  Data file not found: {filepath}")
            continue

        try:
            # Load and prepare data
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

            # Drop unnecessary columns
            cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
            df = df.drop(columns=cols_to_drop)

            print(f"  Loaded data: {df.shape} ({df.index.min()} to {df.index.max()})")

            # Create and train pipeline
            pipeline = HybridTradingPipeline(use_preclassified_regimes=False)

            # Train regime classifier (using first 70% of data)
            train_size = int(len(df) * 0.7)
            train_df = df.iloc[:train_size]

            train_result = pipeline.train_regime_classifier(train_df, test_size=0.3)

            # Run backtest on full dataset
            backtest_result = pipeline.backtest(df, initial_capital=initial_capital)

            # Store results
            results[timeframe] = {
                'train_result': train_result,
                'backtest_result': backtest_result,
                'data_shape': df.shape,
                'pipeline': pipeline
            }

            # Print summary
            metrics = backtest_result['performance_metrics']
            print(f"  Results: {metrics['total_return']:.2f}% return, {metrics['num_trades']} trades, {metrics['win_rate']:.2%} win rate")

        except Exception as e:
            print(f"  Error processing {timeframe}: {str(e)}")
            continue

    return results


if __name__ == "__main__":
    print("Hybrid ML Trading Pipeline")
    print("=========================")

    # Example usage with pre-classified regime data
    print("Running analysis with pre-classified 1D momentum regime data...")

    # Test the pre-classified regime analysis
    result = run_preclassified_regime_analysis()

    if result:
        print("\nPre-classified regime analysis completed successfully!")

        # Show summary if backtest results are available
        if 'backtest_result' in result:
            metrics = result['backtest_result']['performance_metrics']
            print("\nPerformance Summary:")
            print("-" * 40)
            print(f"Total Return: {metrics['total_return']:>7.2f}%")
            print(f"Number of Trades: {metrics['num_trades']:>3}")
            print(f"Win Rate: {metrics['win_rate']:>5.1%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:>6.3f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:>6.2%}")
    else:
        print("Analysis failed - check data availability and file paths")

        # Fallback to traditional multi-timeframe analysis
        print("\nFalling back to traditional multi-timeframe analysis...")
        data_dir = "../data"

        if os.path.exists(data_dir):
            results = run_multi_timeframe_analysis(data_dir, use_preclassified=False)

            if results:
                print(f"\nAnalysis completed for {len(results)} timeframes")

                # Show summary
                print("\nSummary Results:")
                print("-" * 50)
                for tf, result in results.items():
                    metrics = result['backtest_result']['performance_metrics']
                    print(f"{tf:>8}: {metrics['total_return']:>7.2f}% | {metrics['num_trades']:>3} trades | {metrics['win_rate']:>5.1%} win")
            else:
                print("No results generated - check data availability")
        else:
            print(f"Data directory not found: {data_dir}")
            print("Please ensure datasets are available in the data directory")