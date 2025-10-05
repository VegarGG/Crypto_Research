"""
Hybrid ML Trading Pipeline Package

This package provides a complete trading system that combines machine learning
market regime classification with rule-based trading strategies.

Components:
- regime_classifier: ML models for market regime classification
- trading_strategies: Rule-based strategies for different market conditions
- hybrid_pipeline: Main pipeline combining all components
- backtesting: Comprehensive backtesting utilities

Usage:
    from pipeline import HybridTradingPipeline

    # Create and train pipeline
    pipeline = HybridTradingPipeline()
    pipeline.train_regime_classifier(training_data)

    # Run backtest
    results = pipeline.backtest(test_data)
"""

from .regime_classifier import (
    RegimeClassifierPipeline,
    RegimeFeatureEnginer,
    train_multiple_models
)

from .trading_strategies import (
    BaseStrategy,
    BullMarketStrategy,
    BearMarketStrategy,
    SidewaysMarketStrategy,
    StrategyManager,
    StrategyOptimizer,
    TradeSignal,
    Trade,
    create_default_strategies
)

from .hybrid_pipeline import (
    HybridTradingPipeline,
    run_multi_timeframe_analysis
)

__version__ = "1.0.0"
__author__ = "Crypto Research Team"

__all__ = [
    # Regime Classification
    'RegimeClassifierPipeline',
    'RegimeFeatureEnginer',
    'train_multiple_models',

    # Trading Strategies
    'BaseStrategy',
    'BullMarketStrategy',
    'BearMarketStrategy',
    'SidewaysMarketStrategy',
    'StrategyManager',
    'StrategyOptimizer',
    'TradeSignal',
    'Trade',
    'create_default_strategies',

    # Main Pipeline
    'HybridTradingPipeline',
    'run_multi_timeframe_analysis'
]