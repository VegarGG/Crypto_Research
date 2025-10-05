# Hybrid ML Trading Pipeline

A comprehensive trading system that combines machine learning market regime classification with rule-based trading strategies for cryptocurrency markets.

## Overview

This pipeline integrates:
- **Machine Learning Models** for market regime classification (Bull/Bear/Sideways)
- **Rule-Based Strategies** optimized for different market conditions
- **Advanced Backtesting** with comprehensive performance metrics
- **Multi-Timeframe Analysis** across different trading intervals

## Architecture

```
pipeline/
├── __init__.py                 # Package initialization
├── regime_classifier.py       # ML models for market regime classification
├── trading_strategies.py      # Rule-based trading strategies
├── hybrid_pipeline.py         # Main pipeline combining all components
├── backtesting.py             # Advanced backtesting utilities
├── demo.py                    # Demonstration script
└── README.md                  # This file
```

## Quick Start

### 1. Basic Usage

```python
from pipeline import HybridTradingPipeline
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')
df = df.set_index('timestamp')

# Create and train pipeline
pipeline = HybridTradingPipeline()
pipeline.train_regime_classifier(df, model_type='RandomForest')

# Run backtest
results = pipeline.backtest(df, initial_capital=10000)

# View results
print(f"Total Return: {results['performance_metrics']['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.3f}")
```

### 2. Multi-Timeframe Analysis

```python
from pipeline import run_multi_timeframe_analysis

# Run analysis across multiple timeframes
results = run_multi_timeframe_analysis(
    data_dir='../data',
    timeframes=['1min', '15min', '30min', '1day'],
    initial_capital=10000
)
```

### 3. Run Demo

```bash
cd pipeline
python demo.py
```

## Components

### 1. Regime Classifier (`regime_classifier.py`)

Machine learning models for classifying market regimes:

- **Models**: Random Forest, Gradient Boosting, SVM
- **Features**: Technical indicators, price patterns, volatility metrics
- **Regimes**: 0 (Bear), 1 (Sideways), 2 (Bull)

```python
from pipeline.regime_classifier import RegimeClassifierPipeline

classifier = RegimeClassifierPipeline('RandomForest')
X_train, X_test, y_train, y_test, features = classifier.prepare_training_data(df)
classifier.train(X_train, y_train)
predictions, probabilities = classifier.predict(new_data)
```

### 2. Trading Strategies (`trading_strategies.py`)

Regime-specific trading strategies:

#### Bull Market Strategy
- **Approach**: Aggressive trend-following
- **Entry**: EMA crossover + RSI momentum + MACD confirmation
- **Exit**: Stop-loss, take-profit, trailing stop, trend reversal

#### Bear Market Strategy
- **Approach**: Conservative mean reversion
- **Entry**: Oversold RSI + price below Bollinger lower band
- **Exit**: Stop-loss, take-profit, overbought conditions

#### Sideways Market Strategy
- **Approach**: Range-bound trading
- **Entry**: Price near Bollinger bands + RSI extremes
- **Exit**: Stop-loss, take-profit, opposite band touch

```python
from pipeline.trading_strategies import BullMarketStrategy, create_default_strategies

# Create individual strategy
bull_strategy = BullMarketStrategy(ma_fast=7, ma_slow=20, stop_loss=0.06)

# Or use default set
strategy_manager = create_default_strategies()
```

### 3. Hybrid Pipeline (`hybrid_pipeline.py`)

Main pipeline combining regime classification with trading strategies:

```python
from pipeline.hybrid_pipeline import HybridTradingPipeline

pipeline = HybridTradingPipeline()

# Train regime classifier
pipeline.train_regime_classifier(training_data)

# Generate signals
signals, regime_predictions = pipeline.generate_signals(test_data)

# Run backtest
results = pipeline.backtest(test_data)

# Save pipeline for later use
pipeline.save_pipeline('saved_models/')
```

### 4. Advanced Backtesting (`backtesting.py`)

Comprehensive backtesting with advanced analytics:

```python
from pipeline.backtesting import BacktestEngine, PerformanceAnalyzer

# Create backtest engine
engine = BacktestEngine(
    initial_capital=10000,
    commission=0.001,
    slippage=0.0001
)

# Run backtest
results = engine.run_backtest(df, signals)

# Generate performance report
report = PerformanceAnalyzer.generate_performance_report(results)
print(report)

# Plot equity curve
PerformanceAnalyzer.plot_equity_curve(results['equity_curve'])
```

## Data Requirements

The pipeline expects data with the following columns:

### Required OHLCV Data
- `Open`, `High`, `Low`, `Close`, `Volume`
- `timestamp` or datetime index

### Required Technical Indicators
- **Trend**: `ema_7d`, `ema_20d`, `ema_30d`, `sma_7d`, `sma_20d`, `sma_30d`
- **Momentum**: `macd_12_26`, `macd_sig_12_26`, `macd_hist_12_26`, `rsi_14d`
- **Volatility**: `bb_mid_20d`, `bb_upper_20d`, `bb_lower_20d`, `atr_14d`
- **Optional**: Fractal dimensions (`fd_7d`, `fd_14d`, `fd_30d`)

### Data Preparation

Use the data cleaning notebook or prepare data with:

```python
# Ensure proper datetime index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# Remove any unnamed columns
cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
df = df.drop(columns=cols_to_drop)
```

## Performance Metrics

The pipeline calculates comprehensive performance metrics:

### Return Metrics
- Total Return (%)
- Annualized Return (%)
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

### Risk Metrics
- Maximum Drawdown (%)
- Volatility (%)
- Value at Risk (VaR)

### Trade Metrics
- Number of Trades
- Win Rate (%)
- Profit Factor
- Average Win/Loss
- Average Trade Duration

### Regime-Specific Metrics
- Performance by market regime
- Trade distribution across regimes
- Regime prediction accuracy

## Configuration

### Strategy Parameters

Strategies can be customized with various parameters:

```python
# Bull market strategy
bull_strategy = BullMarketStrategy(
    ma_fast=7,           # Fast moving average period
    ma_slow=20,          # Slow moving average period
    rsi_threshold=40,    # RSI threshold for entry
    stop_loss=0.08,      # Stop loss percentage
    take_profit=0.15,    # Take profit percentage
    trail_stop=0.05      # Trailing stop percentage
)

# Bear market strategy
bear_strategy = BearMarketStrategy(
    rsi_oversold=25,     # RSI oversold threshold
    rsi_overbought=60,   # RSI overbought threshold
    stop_loss=0.05,      # Stop loss percentage
    take_profit=0.08,    # Take profit percentage
    bb_deviation=1.0     # Bollinger band deviation multiplier
)
```

### Model Parameters

ML models can be configured:

```python
classifier = RegimeClassifierPipeline(
    model_type='RandomForest',  # 'RandomForest', 'GradientBoosting', 'SVM'
)

# Custom model parameters
from sklearn.ensemble import RandomForestClassifier
custom_model = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=15))
])
```

## Examples

### Example 1: Basic Strategy Testing

```python
import pandas as pd
from pipeline import HybridTradingPipeline

# Load data
df = pd.read_csv('BTCUSD_2023_15min_cleaned.csv')
df = df.set_index('timestamp')

# Create pipeline
pipeline = HybridTradingPipeline()

# Train on first 70% of data
train_size = int(len(df) * 0.7)
pipeline.train_regime_classifier(df.iloc[:train_size])

# Test on remaining 30%
test_results = pipeline.backtest(df.iloc[train_size:])

print(f"Test Period Return: {test_results['performance_metrics']['total_return']:.2f}%")
```

### Example 2: Strategy Optimization

```python
from pipeline.trading_strategies import StrategyOptimizer

# Define parameter ranges to test
param_ranges = {
    'ma_fast': [5, 7, 10],
    'ma_slow': [15, 20, 25],
    'stop_loss': [0.04, 0.06, 0.08],
    'take_profit': [0.10, 0.12, 0.15]
}

# Optimize bull strategy
regime_predictions, _ = pipeline.predict_regime(df)
optimizer = StrategyOptimizer(BullMarketStrategy, regime_predictions)
best_params = optimizer.optimize_parameters(df, param_ranges)

print(f"Best parameters: {best_params['best_parameters']}")
```

### Example 3: Walk-Forward Analysis

```python
from pipeline.backtesting import WalkForwardAnalysis

def strategy_generator(train_data):
    \"\"\"Generate strategy from training data\"\"\"
    pipeline = HybridTradingPipeline()
    pipeline.train_regime_classifier(train_data)
    return pipeline

# Run walk-forward analysis
wfa = WalkForwardAnalysis(training_window=252, rebalance_freq=63)
wf_results = wfa.run_analysis(df, strategy_generator)

print(f"Average Return: {wf_results['summary']['avg_return']:.2f}%")
print(f"Consistency: {wf_results['summary']['consistency_ratio']:.2%}")
```

## Output Files

The pipeline generates several output files:

### Saved Models
- `regime_classifier_model.pkl` - Trained ML model
- `regime_classifier_metadata.json` - Model metadata and feature columns
- `strategy_config.json` - Strategy configurations

### Backtest Results
- `equity_curve.csv` - Portfolio value over time
- `trades.csv` - Individual trade records
- `performance_summary.csv` - Performance metrics summary

### Analysis Reports
- Text-based performance reports
- Visualization plots (equity curves, drawdowns, trade analysis)

## Requirements

### Python Packages
```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
joblib>=1.0.0
```

### Data Requirements
- Clean OHLCV data with technical indicators
- Minimum 1000 data points for reliable training
- Regular time intervals (1min, 5min, 15min, etc.)

## Troubleshooting

### Common Issues

1. **"Feature columns not found"**
   - Ensure your data has all required technical indicators
   - Check column naming matches expected format

2. **"Insufficient training data"**
   - Use at least 1000 data points for training
   - Consider using longer lookback windows

3. **"Poor model performance"**
   - Try different model types ('RandomForest', 'GradientBoosting', 'SVM')
   - Adjust regime classification thresholds
   - Use more features or longer training period

4. **"No trades generated"**
   - Check strategy parameters (may be too restrictive)
   - Verify regime predictions are reasonable
   - Ensure data has sufficient volatility

### Performance Tips

1. **For faster training**: Use smaller datasets or subsample data
2. **For better accuracy**: Use longer training periods and more features
3. **For more trades**: Relax strategy entry/exit conditions
4. **For better risk management**: Tighten stop-loss and position sizing

## Contributing

To add new features:

1. **New Strategies**: Inherit from `BaseStrategy` class
2. **New Models**: Add to `model_configs` in `RegimeClassifierPipeline`
3. **New Metrics**: Add to `_calculate_performance_metrics` method
4. **New Features**: Add to `engineer_features` method

## License

This pipeline is part of the Crypto Research project. See main project for license details.

## Support

For questions or issues:
1. Check this README for common solutions
2. Review the demo script for usage examples
3. Examine the notebook examples in the strategy directory