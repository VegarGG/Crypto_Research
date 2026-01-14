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
- [strategy/hybrid_rule_ml_regression_strategy_2.ipynb](strategy/hybrid_rule_ml_regression_strategy_2.ipynb) - **LATEST**: Fixed version with A/B testing

**Latest Achievement (Nov 2023)**: Created comprehensive A/B testing framework comparing Close-to-Close vs VWAP return targets, with rigorous fixes for:
- Look-ahead bias (1-day regime lag)
- Realistic execution (next-bar entry with slippage)
- Proper intra-bar stop loss checking
- Transaction cost awareness

### Current State

The latest iteration ([hybrid_rule_ml_regression_strategy_2.ipynb](strategy/hybrid_rule_ml_regression_strategy_2.ipynb)) represents a production-ready framework with:
- No data leakage or look-ahead bias
- Realistic execution modeling
- Comprehensive A/B testing
- Full transaction cost accounting
- Proper statistical model selection using BIC

**Key Insight**: While ML models provide sophisticated predictions, the rule-based benchmark often outperforms due to overfitting and prediction threshold challenges. The journey demonstrates that simpler approaches with proper risk management can outperform complex ML in live trading.

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
│       ├── market_regime_classification.ipynb  # Regime classifiers
│       ├── btc_ml_comparison.ipynb            # Model comparison
│       ├── btc_ml_training_LGBM_GBC.ipynb    # Specific model training
│       └── price_confidence.ipynb             # Confidence-based trading
│
├── strategy/                       # Trading strategy development
│   ├── hybrid_rule_ml_regression_strategy_2.ipynb  # LATEST: Fixed A/B test
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

### Technical Analysis
- **TA-Lib**: Technical analysis indicator library
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
2. **Regime Classification**: [models/code/market_regime_classification.ipynb](models/code/market_regime_classification.ipynb)
3. **ML Model Comparison**: [models/code/btc_ml_comparison.ipynb](models/code/btc_ml_comparison.ipynb)
4. **Latest Strategy**: [strategy/hybrid_rule_ml_regression_strategy_2.ipynb](strategy/hybrid_rule_ml_regression_strategy_2.ipynb)

## Key Learnings

1. **Transaction Costs Matter**: 0.75% commission + slippage drastically changes strategy viability. High-frequency trading becomes unprofitable quickly.

2. **Look-Ahead Bias is Subtle**: Even regime classification can introduce look-ahead bias. Always use lagged features (previous day's regime).

3. **ML Isn't Always Better**: Simple rule-based strategies with proper risk management often outperform complex ML models due to overfitting and threshold sensitivity.

4. **Exit Logic > Entry Logic**: Discovered that 99%+ accuracy models had negative returns due to poor exit strategy. Transaction fee awareness in exits is critical.

5. **Feature Engineering > Model Selection**: Quality features (fractal dimension, regime-aware indicators) contribute more to performance than sophisticated models.

6. **Backtesting Realism**: Must simulate:
   - Next-bar entry (no instant fills)
   - Intra-bar stop loss checks
   - Slippage on both entry and exit
   - Realistic volume-weighted exits

7. **A/B Testing is Essential**: Different target formulations (Close-to-Close vs VWAP) can lead to completely different model behaviors.

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

- [ ] Incorporate additional data sources (order book, funding rates, social sentiment)
- [ ] Deep learning models (LSTM, Transformers) for sequence prediction
- [ ] Multi-asset portfolio strategies (BTC/ETH/altcoins)
- [ ] Real-time data integration and paper trading
- [ ] Risk parity and dynamic position sizing
- [ ] Reinforcement learning for adaptive strategies

## Contributing

This is a research project for learning and experimentation. Contributions, suggestions, and discussions are welcome!

## Disclaimer

This project is for **educational and research purposes only**. The strategies and code are not financial advice. Cryptocurrency trading carries substantial risk. Past performance does not guarantee future results.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Data processing powered by **Pandas** and **NumPy**
- ML workflows simplified by **PyCaret**
- Time-series storage via **ArcticDB**
- Technical indicators from **TA-Lib** and custom implementations

---

**Last Updated**: November 2024
**Project Status**: Active Research
**Latest Notebook**: [strategy/hybrid_rule_ml_regression_strategy_2.ipynb](strategy/hybrid_rule_ml_regression_strategy_2.ipynb)
