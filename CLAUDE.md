# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

- **Purpose**: Quantitative analysis and trading models for cryptocurrency markets
- **Main Technologies**: Python, Jupyter Notebooks, pandas, scikit-learn, PyCaret, customized-technical-indicators pipeline
- **Data Sources**: BTC/USD data, market indicators
- **Current Focus**: ML model comparison and backtesting

## Architecture & Workflow

### Data Pipeline
- **Raw Data**: 1-minute OHLCV Bitcoin data and S&P 500 data in `data/` directory
- **Data Processing**: Monthly split datasets with market regime classification (Bull/Bear/Side markets)
- **Feature Engineering**: Technical indicators pipeline using both custom math implementations and `ta` package
- **Storage**: ArcticDB for efficient time-series data storage and retrieval

### Core Components

1. **Feature Engineering** (`features/`)
   - `trend_indicator_pipeline_math.py`: Custom mathematical implementations of SMA, EMA, ADX, RSI, MACD, Bollinger Bands, ATR
   - `trend_indicator_pipeline_pkg.py`: Package-based implementations using `ta` library
   - Technical indicators include trend, momentum, volatility, correlation, and fractal dimension metrics

2. **Model Training** (`models/`)
   - Market regime classification using multiple ML algorithms
   - Strategy training and backtesting
   - Model comparison and evaluation notebooks

3. **Strategy Development** (`strategy/`)
   - Trading signal generation based on ML predictions
   - Portfolio backtesting and performance analysis
   - Trade logging and aggregation

4. **Data Management** (`utils/`)
   - Data cleaning and preprocessing utilities

### Key Dependencies
- `pandas`, `numpy`: Data manipulation
- `matplotlib`: Visualization
- `arcticdb`: Time-series database
- `ta`: Technical analysis indicators
- `pycaret`: Machine learning pipeline (used in notebooks)

## Development Workflow

### Running Jupyter Notebooks
The primary development environment uses Jupyter notebooks. To start working:

```bash
jupyter notebook
```

Key notebooks by category:
- **EDA**: `features/feature_engineering.ipynb`
- **Feature Engineering**: `features/trend_indicator.ipynb`, `features/btc_sp500_correlation.ipynb`
- **Modeling**: `models/code/market_regime_classification.ipynb`, `models/code/btc_ml_comparison.ipynb`
- **Strategy**: `strategy/strategy_training.ipynb`, `strategy/ml_datasets_prep.ipynb`

### Data Structure
- **1-minute data**: Used for high-frequency feature engineering (window calculations use `days * 1440` minutes)
- **Daily aggregated data**: For regime classification and strategy backtesting
- **Market regimes**: Bull/Bear/Side market classifications stored in `data/BTC_montly/Classified/`

### ArcticDB Configuration
ArcticDB stores are configured with path: `arctic_store`
Libraries created automatically for different indicator types:
- `trend_indicators`
- `momentum_indicators` 
- `volatility_indicators`
- `correlation_indicators`
- `fractal_indicators`

### Feature Engineering Pipeline
Technical indicators are computed using sliding windows:
- Trend: SMA, EMA, ADX
- Momentum: RSI, Stochastic, MACD
- Volatility: Bollinger Bands, ATR
- Correlation: Rolling correlation between BTC and S&P 500
- Complexity: Fractal dimension using Keltner band crossings

### Model Training Flow
1. Load and preprocess market data
2. Engineer technical features 
3. Split data by market regimes
4. Train classification models for regime prediction
5. Generate trading signals based on predictions
6. Backtest strategy performance

## Coding Standards
- Use descriptive variable names
- Document complex algorithms
- Follow PEP 8 for Python code
- Include docstrings for functions or markdown in notebook
- No emoji in any code/comments/documents/git commit


## Important Notes
- ArcticDB path is hardcoded to `/Users/zway/Desktop/BTC_Project/DB` - may need adjustment for different environments
- The project uses 1-minute resolution data, so window calculations multiply days by 1440
- Market regime classification is central to the strategy - data is split into Bull/Bear/Side market periods
- No formal dependency management file exists - dependencies must be installed manually