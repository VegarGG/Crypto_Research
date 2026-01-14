# Bitcoin Quantitative Trading Strategy Research

A comprehensive quantitative research project focused on developing machine learning-based trading strategies for Bitcoin (BTC/USD) using 1-minute OHLCV data. This project demonstrates the complete workflow from feature engineering to strategy deployment, with emphasis on avoiding common pitfalls like look-ahead bias and overfitting.

## Project Evolution

This repository chronicles a methodical journey from basic technical analysis to sophisticated hybrid ML strategies:

### Phase 1: Data Foundation (Feb-Jun 2023)
**Exploratory Data Analysis and Feature Engineering**

Started with fundamental data cleaning and exploration of 1-minute Bitcoin price data covering 2023. Built custom technical indicator pipelines using both mathematical formulas and the `ta` package to ensure accuracy and understanding.

- [features/feature_engineering.ipynb](features/feature_engineering.ipynb) - Initial EDA and feature exploration
- [utils/data_cleaning.ipynb](utils/data_cleaning.ipynb) - Data preprocessing and quality checks
- [features/trend_indicator.ipynb](features/trend_indicator.ipynb) - Custom technical indicator implementations

**Key Achievement**: Developed dual-pipeline feature engineering system (math-based and package-based) with validation between implementations.

### Phase 2: Technical Indicator Refinement (Jun-Jul 2023)
**Advanced Indicator Development**

Expanded feature engineering to include trend (SMA, EMA, ADX), momentum (RSI, Stochastic, MACD), volatility (Bollinger Bands, ATR), correlation (BTC-SP500), and fractal dimension indicators. Optimized computation using Numba JIT compilation and multi-threading.

- [features/trend_indicator_pipeline_math.py](features/trend_indicator_pipeline_math.py) - Math-based indicator pipeline
- [features/trend_indicator_pipeline_pkg.py](features/trend_indicator_pipeline_pkg.py) - Package-based indicator pipeline
- [features/btc_sp500_correlation.ipynb](features/btc_sp500_correlation.ipynb) - Market correlation analysis

**Key Achievement**: Successfully optimized fractal dimension computation, reducing calculation time by 80% through Numba JIT and parallel processing.

### Phase 3: Market Regime Classification (Jul-Aug 2023)
**Systematic Market State Identification**

Manually labeled monthly datasets (Mar-Dec 2023) into Bull/Bear/Sideways regimes based on visual inspection. Developed ML classifiers to automatically identify market regimes using technical indicators.

- [strategy/monthly_visualization.ipynb](strategy/monthly_visualization.ipynb) - Monthly regime labeling
- [strategy/ml_datasets_prep.ipynb](strategy/ml_datasets_prep.ipynb) - Dataset preparation for ML training
- [models/code/market_regime_classification.ipynb](models/code/market_regime_classification.ipynb) - Regime classification models

**Key Achievement**: Built reliable market regime classifier achieving high accuracy in identifying bull/bear/sideways markets using MACD and fractal dimension indicators.

### Phase 4: Initial Strategy Development (Jul-Aug 2023)
**Rule-Based and Simple ML Strategies**

Created initial rule-based trading strategies with progressively refined entry/exit logic. Discovered critical bugs in portfolio return calculations and the importance of transaction costs.

- [strategy/strategy_training.ipynb](strategy/strategy_training.ipynb) - Initial strategy prototypes
- [strategy/strategy_training_p2.ipynb](strategy/strategy_training_p2.ipynb) - Enhanced v3.5 and v3.6 strategies

**Key Achievement**: V3.5 strategy achieved 25.88% gain on test data, but identified need for better risk management (low Sharpe ratio).

### Phase 5: ML Model Comparison (Aug 2023)
**Comprehensive Model Evaluation**

Systematically compared 20+ machine learning classifiers using PyCaret for predicting price trends. Implemented proper backtesting with transaction costs, exit logic refinement, and trade logging.

- [models/code/btc_ml_comparison.ipynb](models/code/btc_ml_comparison.ipynb) - Comprehensive ML model comparison
- [models/code/btc_ml_training_LGBM_GBC.ipynb](models/code/btc_ml_training_LGBM_GBC.ipynb) - LGBM and GradientBoosting deep dive
- [strategy/trade_log_analysis.ipynb](strategy/trade_log_analysis.ipynb) - Trade performance analysis

**Critical Discovery**: Initial models achieved >99% accuracy but terrible returns due to poor exit logic. Transaction fees awareness completely changed strategy design - discovered that avoiding high-frequency trading improved win rate and trade quality.

### Phase 6: Hybrid ML Strategies (Oct 2023)
**Combining Rules and Machine Learning**

Developed sophisticated hybrid approaches combining rule-based regime detection with ML-based trade selection. Progressed from classification to regression models for more nuanced return prediction.

- [strategy/hybrid_ml_strategy.ipynb](strategy/hybrid_ml_strategy.ipynb) - Initial hybrid approach
- [strategy/hybrid_ml_strategy_2.ipynb](strategy/hybrid_ml_strategy_2.ipynb) - Improved workflow structure
- [strategy/hybrid_ml_strategy_optimized.ipynb](strategy/hybrid_ml_strategy_optimized.ipynb) - Performance optimizations

**Key Achievement**: Successfully integrated rule-based regime detection with ML models, achieving better risk-adjusted returns than either approach alone.

### Phase 7: Regression-Based Strategies (Oct-Nov 2023)
**Advanced Return Prediction**

Transitioned from classification (up/down) to regression (predicted return magnitude) approaches. Implemented sophisticated A/B testing framework to compare different target formulations.

- [strategy/hybrid_rule_ml_regression_strategy.ipynb](strategy/hybrid_rule_ml_regression_strategy.ipynb) - Regression-based strategy with PyCaret
- [strategy/hybrid_rule_ml_regression_strategy_2.ipynb](strategy/hybrid_rule_ml_regression_strategy_2.ipynb) - Fixed version with A/B testing

**Achievement (Nov 2023)**: Created comprehensive A/B testing framework comparing Close-to-Close vs VWAP return targets, with rigorous fixes for:
- Look-ahead bias (1-day regime lag)
- Realistic execution (next-bar entry with slippage)
- Proper intra-bar stop loss checking
- Transaction cost awareness

**Critical Discovery**: Market regime classification models showed poor predictive performance, generating either no signals or unprofitable trades. This led to a strategic pivot toward time-series forecasting approaches.

### Phase 8: Time-Series Forecasting (Dec 2025)
**Hybrid Forecasting as Primary Approach**

After discovering that market regime classification fails to generate profitable signals, shifted to hybrid time-series forecasting combining trend decomposition with residual modeling. This approach serves as both the primary strategy and benchmark.

- [models/code/timeseries_forecasting.ipynb](models/code/timeseries_forecasting.ipynb) - Hybrid Prophet + ARIMA baseline
- [strategy/integrated_strategy_v2.ipynb](strategy/integrated_strategy_v2.ipynb) - Integrated strategy with regime filter (4H timeframe)
- [strategy/integrated_strategy_v3.ipynb](strategy/integrated_strategy_v3.ipynb) - Enhanced with GARCH volatility filtering (1m data, standardized features)
- [strategy/integrated_strategy_v4.ipynb](strategy/integrated_strategy_v4.ipynb) - **LATEST**: A/B testing Prophet+ARIMA vs Prophet+SARIMA (1m resolution)

**Key Achievements**:
- **Hybrid Forecasting**: Prophet (trend/seasonality) + ARIMA/SARIMA (residuals) for complete price prediction
- **Multi-Timeframe**: Testing from 1-minute to 4-hour resolutions to balance signal quality and execution feasibility
- **GARCH Integration**: Volatility regime filtering to avoid trading during extreme volatility periods
- **Leakage Prevention**: Strict shift(1) on all higher-timeframe signals before broadcasting to 1m execution
- **Model Comparison**: A/B testing ARIMA vs SARIMA for residual modeling at 1-minute resolution

**Current Results**: Time-series only strategies (without regime filters) generate consistent signals and provide measurable baseline performance. ARIMA-based residual modeling shows promise but requires threshold optimization to balance signal frequency with profitability after 0.75% transaction costs.

### Current State

The latest iteration ([strategy/integrated_strategy_v4.ipynb](strategy/integrated_strategy_v4.ipynb)) represents a shift in approach:
- **Primary Method**: Hybrid time-series forecasting (Prophet + ARIMA/SARIMA)
- **Benchmark Role**: Time-series forecasts serve as performance baseline
- **No Regime Dependency**: Operates independently of regime classification
- **1-Minute Resolution**: Direct forecasting on 1m data for maximum signal granularity
- **Transaction Cost Aware**: 0.75% commission per side + slippage modeling

**Key Insight**: Market regime classification, while theoretically sound, fails to generate actionable trading signals in practice. Time-series forecasting provides a more reliable foundation by directly predicting price movements and expected returns, enabling measurable strategy performance even if absolute returns are modest.

## Repository Structure

```
Crypto_Research/
├── data/                           # Raw and processed datasets
│   ├── BTC_montly/                # Monthly split datasets
│   │   └── Classified/            # Market regime labeled data
│   └── BTCUSD_2023_1min_cleaned.csv
│
├── features/                       # Feature engineering
│   ├── trend_indicator_pipeline_math.py    # Math-based indicators
│   ├── trend_indicator_pipeline_pkg.py     # Package-based indicators
│   ├── feature_engineering.ipynb           # Initial feature exploration
│   ├── trend_indicator.ipynb               # Indicator testing
│   └── btc_sp500_correlation.ipynb         # Correlation analysis
│
├── models/                         # Model training and evaluation
│   └── code/
│       ├── timeseries_forecasting.ipynb            # BASELINE: Hybrid Prophet + ARIMA
│       ├── market_regime_classification.ipynb      # Regime classifiers (deprecated)
│       ├── btc_ml_comparison.ipynb                # Model comparison
│       ├── btc_ml_training_LGBM_GBC.ipynb        # Specific model training
│       └── price_confidence.ipynb                 # Confidence-based trading
│
├── strategy/                       # Trading strategy development
│   ├── integrated_strategy_v4.ipynb                # LATEST: Prophet+ARIMA vs SARIMA A/B
│   ├── integrated_strategy_v3.ipynb                # GARCH volatility filtering
│   ├── integrated_strategy_v2.ipynb                # Regime + forecasting integration
│   ├── hybrid_rule_ml_regression_strategy_2.ipynb  # Regression-based ML strategy
│   ├── hybrid_ml_strategy_optimized.ipynb         # Hybrid classification
│   ├── strategy_training.ipynb                    # Rule-based strategies
│   ├── ml_datasets_prep.ipynb                     # ML dataset preparation
│   └── trade_log_analysis.ipynb                   # Performance analysis
│
├── utils/                          # Utilities
│   ├── data_cleaning.ipynb
│   ├── monthly_split.ipynb
│   └── data_cleaning_aggregation.ipynb
│
├── arctic_store/                   # ArcticDB time-series storage
├── setup_environment.py            # Environment setup helper
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Technologies

### Core Stack
- **Python 3.8+**: Primary language
- **Jupyter Notebooks**: Interactive development and analysis
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Matplotlib & Seaborn**: Visualization

### Machine Learning
- **PyCaret**: Automated ML pipeline for model comparison
- **Scikit-learn**: Classical ML algorithms
- **LightGBM / XGBoost**: Gradient boosting frameworks

### Time-Series Forecasting
- **Prophet**: Facebook's time-series forecasting library (trend/seasonality)
- **Statsmodels**: ARIMA/SARIMA/GARCH models
- **Arch**: GARCH volatility modeling

### Technical Analysis
- **TA**: Technical analysis indicator library
- **Custom implementations**: Math-based indicator pipelines

### Performance & Storage
- **ArcticDB**: High-performance time-series database
- **Numba**: JIT compilation for computational optimization
- **Multiprocessing**: Parallel feature computation

## Getting Started

### 1. Environment Setup

Clone the repository:
```bash
git clone https://github.com/VegarGG/Crypto_Research.git
cd Crypto_Research
```

Run the automated setup script:
```bash
python setup_environment.py
```

This will:
- Check your Python environment
- Install missing dependencies
- Configure ArcticDB storage
- Generate requirements.txt

Or manually install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Required Packages

**Core Dependencies:**
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- scipy >= 1.9.0

**Machine Learning:**
- scikit-learn >= 1.2.0
- pycaret >= 3.0.0
- lightgbm >= 3.3.0
- xgboost >= 1.7.0

**Time-Series Forecasting:**
- prophet >= 1.1.0
- statsmodels >= 0.14.0
- arch >= 6.0.0

**Technical Analysis:**
- ta >= 0.10.0

**Database:**
- arcticdb >= 4.0.0

**Performance:**
- numba >= 0.56.0

**Jupyter:**
- jupyter >= 1.0.0
- ipykernel >= 6.0.0

### 3. Data Preparation

Place your BTC/USD 1-minute OHLCV data in the `data/` directory:
```
data/BTCUSD_2023_1min_cleaned.csv
```

Expected columns: `timestamp, Open, High, Low, Close, Volume`

### 4. Running Notebooks

Start Jupyter:
```bash
jupyter notebook
```

**Recommended execution order:**

1. **Feature Engineering**: [features/trend_indicator.ipynb](features/trend_indicator.ipynb)
2. **Time-Series Baseline**: [models/code/timeseries_forecasting.ipynb](models/code/timeseries_forecasting.ipynb)
3. **Latest Strategy (A/B Test)**: [strategy/integrated_strategy_v4.ipynb](strategy/integrated_strategy_v4.ipynb)
4. **GARCH Volatility Filter**: [strategy/integrated_strategy_v3.ipynb](strategy/integrated_strategy_v3.ipynb)

## Key Learnings

1. **Transaction Costs Matter**: 0.75% commission + slippage drastically changes strategy viability. High-frequency trading becomes unprofitable quickly.

2. **Look-Ahead Bias is Subtle**: Even regime classification can introduce look-ahead bias. Always use lagged features (shift(1) on higher timeframes before broadcasting).

3. **Regime Classification Limitations**: Market regime models (Bull/Bear/Sideways) achieve high accuracy in backtests but fail to generate profitable signals in forward testing. Classification precision doesn't translate to trading edge.

4. **Time-Series Forecasting as Benchmark**: Hybrid Prophet + ARIMA/SARIMA provides measurable baseline performance. Direct price forecasting generates consistent signals and quantifiable expected returns, making it superior to regime-based approaches.

5. **Exit Logic > Entry Logic**: Discovered that 99%+ accuracy models had negative returns due to poor exit strategy. Transaction fee awareness in exits is critical.

6. **Feature Engineering > Model Selection**: Quality features (fractal dimension, regime-aware indicators) contribute more to performance than sophisticated models.

7. **Backtesting Realism**: Must simulate:
   - Next-bar entry (no instant fills)
   - Intra-bar stop loss checks
   - Slippage on both entry and exit
   - Realistic volume-weighted exits
   - Strict temporal ordering (shift signals before execution)

8. **Multi-Timeframe Tradeoffs**: Higher timeframes (4H) provide cleaner signals but fewer trading opportunities. 1-minute resolution maximizes granularity but increases noise and overfitting risk.

9. **Volatility Filtering**: GARCH-based volatility regime detection can improve risk management but may eliminate profitable trades during volatile periods. Threshold selection is critical.

## Performance Notes

**ArcticDB Configuration**:
- Default path: `./arctic_store`
- Configured via [config.py](config.py) (auto-generated by setup script)
- Libraries: trend_indicators, momentum_indicators, volatility_indicators, correlation_indicators, fractal_indicators

**Window Calculations**:
- 1-minute resolution: multiply days by 1,440 (e.g., 7-day SMA = 10,080 minutes)

**Computational Optimization**:
- Fractal dimension: 80% speedup using Numba JIT
- Feature pipelines: Parallel processing using ProcessPoolExecutor

## Future Work

- [ ] Optimize ARIMA/SARIMA hyperparameters for 1-minute data
- [ ] Rolling forecast windows vs static train/test split
- [ ] Ensemble forecasting: combine multiple time-series models
- [ ] Incorporate additional data sources (order book, funding rates, social sentiment)
- [ ] Deep learning models (LSTM, Transformers) for sequence prediction
- [ ] Multi-asset portfolio strategies (BTC/ETH/altcoins)
- [ ] Real-time data integration and paper trading
- [ ] Risk parity and dynamic position sizing
- [ ] Signal threshold optimization using walk-forward analysis

## Contributing

This is a research project for learning and experimentation. Contributions, suggestions, and discussions are welcome!

## Disclaimer

This project is for **educational and research purposes only**. The strategies and code are not financial advice. Cryptocurrency trading carries substantial risk. Past performance does not guarantee future results.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Data processing powered by **Pandas** and **NumPy**
- Time-series forecasting via **Prophet** and **Statsmodels**
- Volatility modeling using **Arch**
- ML workflows simplified by **PyCaret**
- Time-series storage via **ArcticDB**
- Technical indicators from **TA** library and custom implementations

---

**Last Updated**: January 2026
**Project Status**: Active Research
**Latest Notebook**: [strategy/integrated_strategy_v4.ipynb](strategy/integrated_strategy_v4.ipynb)
**Baseline Benchmark**: [models/code/timeseries_forecasting.ipynb](models/code/timeseries_forecasting.ipynb)
