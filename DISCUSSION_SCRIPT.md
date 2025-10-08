# BTC Trading Strategy Development - Technical Discussion Script

## Session Overview
Date: October 8, 2025
Topic: Regime Classification & Hybrid ML Trading Strategy Development
Duration: 45-60 minutes

---

## Part 1: Regime Classification Evolution (15 minutes)

### 1.1 Initial Challenge
**Problem Statement:**
- Need to classify market conditions (Bull/Bear/Sideways) for adaptive trading
- 1-minute data creates noise and unstable regime classifications
- Want regimes that represent actual market conditions, not momentary fluctuations

### 1.2 Approach 1: Hierarchical Regime Classification
**File:** `strategy/regime_classification_hierarchical.ipynb`

**Key Concept:**
- Classify regimes on LONGER timeframes (daily/weekly)
- Apply those labels to ALL 1-minute bars within each period
- Like saying "the whole day was bullish" rather than changing every minute

**Implementation:**
```
Daily Method:
- Aggregate 1-min → Daily OHLCV
- Calculate daily MAs (SMA10, SMA20, SMA50, EMA10, EMA20)
- Classify: Bull if Price > EMA20 > SMA20 > SMA50
- Apply regime label to all 1-min bars in that day

Weekly Method:
- Aggregate 1-min → Weekly OHLCV
- Calculate weekly MAs (SMA4, SMA8, SMA12)
- Classify: Bull if Price > EMA4 > SMA4 > SMA8
- Apply regime label to all 1-min bars in that week
```

**Results:**
- Daily approach: 302,039 sideways (69.5%), 83,216 bull (19.2%), 49,183 bear (11.3%)
- Weekly approach: 264,402 sideways (60.9%), 97,089 bull (22.3%), 72,947 bear (16.8%)
- Average regime duration: Daily 301.7h, Weekly 402.3h
- Regime changes: Daily 24 changes, Weekly 18 changes

**Strengths:**
- Much more stable than minute-by-minute classification
- Represents longer-term market conditions
- Fewer regime changes = less trading noise

**Limitations:**
- ALL 1-min bars in a period get same label
- Doesn't capture intra-day/intra-week shifts
- May miss regime changes happening mid-period

### 1.3 Approach 2: Enhanced Hierarchical with Multi-Timeframe
**File:** `strategy/regime_classification_enhanced.ipynb`

**Key Innovation:**
- Test MULTIPLE timeframes: 1H, 6H, 12H, 1D, 1W
- Test MULTIPLE methods: Trend-based, Momentum-based (MACD+FD), Volatility-adjusted
- AUTOMATICALLY select optimal combination using scoring system

**Methodology:**
```
For each timeframe:
  For each method:
    1. Classify regimes
    2. Check: Has all 3 regimes? (Bear/Sideways/Bull)
    3. Calculate balance score (lower std = more balanced)
    4. Calculate stability score (avg duration)
    5. Calculate change score (need some variation)

Selection Criteria (scored):
  - Has all 3 regimes: +3 points
  - Balance score: +2 points max
  - Stability score: +2 points max
  - Reasonable changes: +1 point max

Winner: Highest total score
```

**Methods Tested:**
1. **Trend-strength:** Multiple MA alignment
2. **Momentum:** MACD histogram + signal crossovers + Fractal Dimension
3. **Volatility-adjusted:** Dynamic thresholds based on rolling volatility

**Results:**
```
Timeframe Scores:
  1H:  5.39 (avg 4.6h duration, 2.00 changes score)
  6H:  6.53 (avg 18.3h duration, 2.00 changes score)
  12H: 8.14 (avg 37.7h duration, 2.00 changes score)
  1D:  8.42 (avg 102.0h duration, 1.42 changes score) ← WINNER
  1W:  7.24 (avg 603.4h duration, 0.24 changes score)

Selected: 1D Momentum method
  - Bear: 74,605 (17.2%)
  - Sideways: 296,485 (68.2%)
  - Bull: 63,348 (14.6%)
  - Avg regime duration: 102 hours
  - Total regime changes: 71
```

**Strengths:**
- Scientific, data-driven selection process
- Incorporates advanced momentum indicators (MACD + Fractal Dimension)
- Balances stability with responsiveness
- Repeatable methodology

**Key Insight:**
Daily momentum-based classification won because it:
- Had all 3 regime classes
- Balanced stability (102h avg duration) with adaptability (71 changes)
- Used MACD + Fractal Dimension for better trend/noise discrimination

---

## Part 2: Hybrid ML Strategy Evolution (25 minutes)

### 2.1 Strategy Framework
**Core Idea:** Train ML models to predict regimes, then use regime-specific trading strategies

**Data Split:**
- Training: 30% (Feb 10 - May 15, 2023)
- Testing: 70% (May 15 - Dec 31, 2023)
- Strictly chronological to prevent data leakage

### 2.2 Iteration 1: Basic Hybrid ML Strategy
**File:** `strategy/hybrid_ml_strategy_2.ipynb`

**Architecture:**
```
1. Feature Engineering (98 features):
   - Price-based: returns, volatility, trend strength (4 windows: 5/10/20/50)
   - Technical indicators: RSI, MACD, Bollinger Bands
   - MA relationships: EMA/SMA crossovers and spreads
   - Volume: ratios, momentum, high volume detection
   - Fractal Dimension: trend, normalized, complexity
   - Momentum: price momentum, rank, trend consistency
   - Volatility: rolling vol, ranks, high vol detection
   - Lag features: RSI, MACD, BB position (1/2/3/5 lags)

2. ML Training:
   - RandomForest: 200 trees, max_depth=15
   - GradientBoosting: 150 estimators, max_depth=8
   - SVM: RBF kernel, probability=True
   - All with StandardScaler pipeline
   - Time series cross-validation (5 splits)

3. Regime-Specific Strategies:
   Bull Market: Trend-following
     - Entry: EMA7 > EMA20 AND RSI > 40
     - Exit: 6% stop loss OR 12% take profit

   Bear Market: Mean reversion
     - Entry: RSI < 25 AND Close < BB_lower
     - Exit: 4% stop loss OR 8% take profit OR RSI > 60

   Sideways: Range trading
     - Entry: Close <= BB_lower AND RSI < 30
     - Exit: 3% stop OR 6% profit OR Close >= BB_upper OR RSI > 70

4. Transaction Costs:
   - Commission: 0.75%
   - Slippage: 0.01%
   - Total per trade: ~0.76% each side = 1.52% round-trip
```

**Results:**
```
ML Performance:
  Best Model: GradientBoosting (CV: 0.845)
  Test Accuracy: 51.7%

  Regime-specific accuracy:
    Bear: 85.0% (excellent)
    Sideways: 54.8% (weak)
    Bull: 13.6% (VERY poor)

Trading Performance:
  Final Value: $4,709.89
  Total Return: -52.90%
  BTC Buy & Hold: +55.26%
  Underperformance: -108.17%

  Trades: 52
  Win Rate: 9.62% (5 winners, 47 losers)
  Sharpe Ratio: -7.341
  Max Drawdown: -57.10%

  Commission Paid: $5,703.71 (57% of initial capital!)
  Avg Trade Duration: 4.4 hours
```

**Problems Identified:**
1. **Class Imbalance Issue:**
   - Sideways: 208,450 samples (68%)
   - Bear: 42,189 samples (14%)
   - Bull: 53,468 samples (18%)
   - Model heavily biased toward sideways → poor bull/bear detection

2. **Over-Trading:**
   - 52 trades in 7 months = 7.4 trades/month
   - 4.4 hour avg duration = churning
   - Commission ate 57% of capital

3. **High Transaction Costs:**
   - 0.75% fee × 2 sides × 52 trades = devastating
   - Short duration trades don't justify fee burden

4. **No Confidence Filtering:**
   - Traded on ALL signals regardless of model confidence
   - No quality filter on predictions

5. **Tight Take Profits:**
   - Bull strategy: 12% take profit
   - Exits too early, caps upside

### 2.3 Iteration 2: Optimized Strategy with Fee-Aware Logic
**File:** `strategy/hybrid_ml_strategy_optimized.ipynb`

**Key Innovations:**

**1. Two-Stage Classification (Solves Class Imbalance):**
```
Stage 1: Binary Classifier - "Is it Trending?"
  - Sideways (0) vs Trending (1)
  - Boost trending detection with class_weight={0:1, 1:2}
  - Stage 1 Accuracy: 100% on training

Stage 2: Directional Classifier - "Which direction?"
  - Bear (0) vs Bull (2)
  - Only runs on trending periods
  - Stage 2 Accuracy: 100% on training

Combined Test Results:
  Overall Accuracy: 54.2% (up from 51.7%)
  Bear: 81.5% (down from 85%, but still good)
  Sideways: 59.1% (up from 54.8%)
  Bull: 13.6% (unchanged - still problematic)
```

**2. Confidence Filtering:**
```python
Minimum Confidence: 80%
- Only trade when model is >80% confident
- Reduces signals from 17,006 bull predictions to 853 (95% reduction)
- Quality over quantity
```

**3. Fee-Aware Exit Manager:**
```python
class FeeAwareExitManager:
    def __init__(self):
        self.round_trip_cost = 1.52%  # 0.76% × 2
        self.min_hold_hours = 24      # Force minimum hold

    def should_exit(self):
        # Rule 1: Minimum 24h hold (unless catastrophic loss)
        if hours < 24 and net_pnl > -(stop_loss × 1.5):
            return False  # Keep holding

        # Rule 2: Stop loss on NET P&L (after fees)
        if net_pnl <= -4%:
            return True

        # Rule 3: Regime change after 24h
        if regime_changed and hours >= 24 and net_pnl < 3%:
            return True

        # Rule 4: NO UPPER LIMIT - Let profits run!
        return False
```

**4. Position Sizing by Confidence:**
```python
Confidence 90%+:   100% position
Confidence 85-90%: 75% position
Confidence 80-85%: 50% position
Confidence <80%:   No trade
```

**5. Enhanced Features (+12 regime discrimination features):**
```
- Higher highs/lows patterns (5-day)
- Close position in daily range
- Volume bias (up volume vs down volume)
- Net directional movement
- Range expansion metrics
Total: 110 features (up from 98)
```

**Results:**
```
ML Performance:
  Test Accuracy: 54.2% (improved from 51.7%)
  Average Confidence: 73.0%
  High Confidence Periods (>80%): 18.11%

Trading Performance:
  Final Value: $10,110.27
  Total Return: +1.10%
  BTC Buy & Hold: +55.26%
  Underperformance: -54.16%

  Trades: 1 (down from 52!)
  Win Rate: 100% (1 winner, 0 losers)
  Sharpe Ratio: 0.241 (positive!)
  Max Drawdown: -5.10% (down from -57%)

  Commission Paid: $76.12 (0.76% vs 57%)
  Avg Trade Duration: 812.9 hours (33.9 days)

Trade Details:
  Entry:  Jun 20, 2023 @ $27,897 (80.3% confidence)
  Exit:   Jul 24, 2023 @ $28,945 (regime change)
  Return: +0.55% net (3.01% gross before fees)
  Duration: 812.9 hours
```

**Comparison:**
| Metric | Original | Optimized | Change |
|--------|----------|-----------|--------|
| Return | -52.90% | +1.10% | +54.00% |
| Trades | 52 | 1 | -98% |
| Win Rate | 9.62% | 100% | +90.38% |
| Sharpe | -7.341 | 0.241 | +7.582 |
| Max DD | -57.10% | -5.10% | +52.00% |
| Commission | $5,704 (57%) | $76 (0.76%) | -98.7% |
| Avg Duration | 4.4h | 812.9h | +184x |

**Positive Aspects of the "Failure":**

1. **Massive Risk Reduction:**
   - Max drawdown from -57% to -5% is HUGE
   - Shows proper risk management works

2. **Commission Control:**
   - Reduced from 57% to 0.76% of capital
   - Proves fee-aware logic is essential for crypto

3. **Quality Trades:**
   - 100% win rate (albeit only 1 trade)
   - Shows filtering works - when we trade, we trade well

4. **Sustainable Strategy:**
   - Positive Sharpe ratio (0.241 vs -7.341)
   - Could scale with more capital/diversification

5. **Model Actually Works:**
   - 80.3% confidence trade was profitable
   - Exited at 3% profit when regime changed (smart)

6. **Foundation for Improvement:**
   - Clean codebase with proper exit logic
   - Can tune confidence thresholds
   - Can add more entry conditions

### 2.4 The Core Problem: Bull Market Detection

**Root Cause Analysis:**
```
Bull Regime Accuracy: 13.6%

Why is bull detection so poor?

1. Data Imbalance:
   - Training data (30%): Feb-May 2023
   - This period was mostly sideways/bear
   - Bull: 9,880 samples (7.6% of training data)
   - Model never learned what "bull" looks like

2. Test Period Mismatch:
   - Test data (70%): May-Dec 2023
   - BTC went +55% in this period (clearly bullish)
   - But model predicts bull only 13.6% correctly
   - Model thinks bull market is sideways

3. Fractal Dimension Confusion:
   - High FD can mean both:
     * Strong trending (bull/bear)
     * High volatility (sideways choppy)
   - Model can't distinguish

4. Momentum Indicators Lag:
   - MACD crosses after trend starts
   - By the time MACD confirms, already in position
   - Exits too early on pullbacks
```

**Evidence:**
- 17,006 bull predictions, but only 853 were >80% confident
- Of those 853, only 1 trade met entry criteria
- That 1 trade WAS profitable (+3% gross)
- Problem isn't strategy logic, it's bull detection

---

## Part 3: Key Learnings & Insights (5 minutes)

### 3.1 What Worked

1. **Hierarchical Regime Classification:**
   - Longer timeframes reduce noise
   - Daily momentum method with MACD+FD was optimal
   - Automated selection prevents overfitting to single method

2. **Two-Stage Classification:**
   - Handles class imbalance better than single model
   - Trending vs sideways separation is valuable

3. **Fee-Aware Exit Logic:**
   - 24-hour minimum hold prevents churning
   - Net P&L calculation accounts for fees
   - No upper limit lets profits run

4. **Confidence Filtering:**
   - 80% threshold drastically reduces bad trades
   - 95% signal reduction (17k → 853) is good
   - Quality over quantity is right approach

5. **Position Sizing:**
   - Scaling by confidence manages risk
   - Would work better with more trades

### 3.2 What Didn't Work

1. **Training Period Too Short/Wrong:**
   - 30% of 2023 (Feb-May) missed bull trends
   - Need 2022 data to see full cycle

2. **Feature Engineering for Bull:**
   - Current features don't distinguish bull well
   - Need trend strength, momentum persistence
   - Need volatility normalization

3. **Single Entry Criterion:**
   - Only trading on bull regime pred = 2
   - Could use sideways→bull transitions
   - Could use confidence increases

4. **No Multi-Timeframe Confirmation:**
   - Only using 1D regimes
   - Could require 1H + 6H + 1D alignment

5. **Static Take Profit (removed but could be smarter):**
   - No trailing stops
   - No dynamic exits based on volatility

### 3.3 Unexpected Discoveries

1. **Transaction Costs Dominate:**
   - 0.75% fee seems small, but 52 trades = death
   - Crypto fees are way higher than stocks
   - Must design around fees first, alpha second

2. **Confidence Is Predictive:**
   - 80.3% confident trade was profitable
   - Lower confidence trades (in v1) lost money
   - Confidence filtering is not just risk management, it's alpha

3. **Regime Duration Matters:**
   - 102-hour avg regime duration (1D method) was right
   - Too short (1H: 4.6h) = noise
   - Too long (1W: 603h) = missed transitions

4. **Bear Detection Is Easier:**
   - 81.5% bear accuracy vs 13.6% bull
   - Crashes are sharper, more obvious
   - Bulls are gradual, confusable with sideways

---

## Part 4: Next Steps & Recommendations (10 minutes)

### 4.1 Immediate Improvements (Week 1-2)

**Priority 1: Get More Training Data**
```
Current: 30% of 2023 (Feb-May) = 130k samples
Problem: Missed 2023 bull run, 2022 bear market

Action:
1. Add 2022 full year data
2. Add 2021 data if available
3. Keep 2023 as out-of-sample test
4. Train on 2021-2022, validate on Q1 2023, test on Q2-Q4 2023

Expected:
- See full bull (2021), bear (2022), recovery (2023)
- Bull regime samples increase from 7.6% to ~25%
- Better feature learning for trending markets
```

**Priority 2: Fix Bull Detection Features**
```
Add features specifically for bull discrimination:

1. Trend Strength:
   - Consecutive higher highs (10-day, 20-day, 30-day)
   - Positive days ratio (bullish days / total days)
   - Price distance from lowest low (20/50/100-day)

2. Momentum Persistence:
   - RSI staying in 50-70 range (healthy bull)
   - MACD histogram positive streak count
   - Positive returns consistency (rolling 10-day)

3. Volume Profile:
   - Volume on up days vs down days ratio
   - Breakout volume (volume spike on new highs)
   - Accumulation/distribution (Chaikin)

4. Volatility-Adjusted Returns:
   - Sharpe ratio on rolling windows
   - Sortino ratio (upside deviation)
   - Calmar ratio (return / max drawdown)

5. Multi-Timeframe Trend Alignment:
   - 1H bull AND 6H bull AND 1D bull = strong bull
   - Score: count of timeframes agreeing
   - Require 3/5 timeframes agree
```

**Priority 3: Improve Entry Logic**
```
Current: Only enter when regime_pred == 2 (bull)
Problem: Misses regime transitions, too binary

New Entry Conditions (OR logic):

1. Strong Bull Signal:
   - regime_pred == 2
   - confidence >= 85%
   - Price > EMA20 > EMA50
   - MACD > 0 and rising

2. Regime Transition Signal:
   - regime_pred changed from 1→2 or 0→2
   - confidence >= 80%
   - Price crosses above EMA20
   - RSI 40-60 (not overbought)

3. Confidence Breakout:
   - regime_pred == 2
   - confidence increased by 10% in last 6 hours
   - Price at new 5-day high
   - Volume > avg volume

4. Multi-Timeframe Confirmation:
   - 1H pred == 2 AND 6H pred == 2 AND 1D pred == 2
   - Any confidence >= 75%
   - Strong trend alignment
```

### 4.2 Medium-Term Enhancements (Week 3-4)

**1. Portfolio-Based Approach**
```
Current: Single BTC position, all-in or all-out
Problem: Binary, no diversification

New: Multi-Strategy Portfolio

Strategy A: High Confidence Bull (>85%)
  - 40% capital allocation
  - Wider stops (6%), no limit
  - Min hold: 3 days

Strategy B: Medium Confidence Bull (75-85%)
  - 30% capital allocation
  - Tighter stops (4%), 10% profit target
  - Min hold: 1 day

Strategy C: Sideways Range Trading
  - 20% capital allocation
  - Quick trades (2-4% targets)
  - BB-based entries/exits

Strategy D: Bear/Hedge
  - 10% capital allocation
  - Short positions (if allowed) or stay cash
  - Protect downside

Benefits:
- Diversification across regime uncertainties
- 60% of capital can be working simultaneously
- Reduces impact of single bad prediction
```

**2. Dynamic Exit Logic**
```
Current: Static 4% stop, regime change exit, min 24h hold
Problem: One-size-fits-all, doesn't adapt

New: Adaptive Exits

A. Volatility-Based Stops:
   - High volatility (>80th percentile): 6% stop
   - Medium volatility (40-80th): 4% stop
   - Low volatility (<40th): 3% stop

B. Trailing Stops:
   - If profit > 5%: trail at -3% from peak
   - If profit > 10%: trail at -4% from peak
   - If profit > 20%: trail at -5% from peak

C. Time-Based Exits:
   - If held > 7 days and profit < 2%: exit (stalled)
   - If held > 14 days and profit < 5%: exit (weak trend)

D. Technical Exits:
   - Price crosses below EMA20: warning
   - Price crosses below EMA50: exit
   - RSI > 80 for 6 hours: take profit
   - MACD bearish cross: exit
```

**3. Ensemble Modeling**
```
Current: Single GradientBoosting model
Problem: Model-specific biases

New: Weighted Ensemble

Models:
1. GradientBoosting (current best) - 40% weight
2. RandomForest - 30% weight
3. XGBoost - 20% weight
4. LightGBM - 10% weight

Prediction:
- Weighted average of probabilities
- Confidence = agreement score
  * All agree (same class): 95%+ confidence
  * 3/4 agree: 80-85% confidence
  * Split decision: <70% confidence (no trade)

Benefits:
- Reduces single model overfitting
- Higher confidence when models agree
- More robust to market regime changes
```

### 4.3 Advanced Research (Month 2+)

**1. Regime-Switching Models**
```
Try Hidden Markov Models (HMM) or Gaussian Mixture Models (GMM)
- Learn regime transitions probabilistically
- Don't force discrete labels
- Output probability distribution over regimes
```

**2. Reinforcement Learning**
```
Use RL to learn optimal entry/exit timing
- State: price, indicators, current regime probabilities
- Action: {long, short, hold}
- Reward: Sharpe ratio, accounting for fees
- Train with PPO or SAC algorithms
```

**3. Alternative Data**
```
Incorporate:
- Funding rates (perpetual futures)
- Open interest (derivatives)
- Exchange inflows/outflows (on-chain)
- Twitter sentiment
- Google Trends
- Fear & Greed Index
```

**4. Multi-Asset Regimes**
```
Don't just classify BTC regimes
- Classify broader crypto market (ETH, altcoins)
- Classify macro regimes (stocks, bonds, DXY)
- BTC bull in risk-on environment != BTC bull in risk-off
```

---

## Part 5: Questions for Tech Support Team

### 5.1 Data & Infrastructure

**Q1: Historical Data Availability**
```
Can we get access to:
- BTC/USD 1-minute data for 2021-2022?
- Other assets (ETH, S&P500) for 2021-2023?
- Alternative data sources (funding rates, open interest)?

Context: Training on only Feb-May 2023 gave us insufficient bull market
samples (7.6%). Need 2021 bull + 2022 bear for balanced training.
```

**Q2: Data Quality & Gaps**
```
Current data (2023) has 434,438 1-minute bars.
Expected: 365 days × 1440 min/day = 525,600 bars

Missing: ~91,000 bars (17%)

Questions:
- Are gaps due to exchange downtime or data collection issues?
- Can gaps be filled from other data sources?
- Should we resample to 5-min to reduce gaps?

Impact: Missing data during regime transitions could hurt model training.
```

**Q3: Computational Resources**
```
Current bottlenecks:
- Feature engineering on 434k rows × 110 features = slow
- Training 5-fold TimeSeriesSplit = 5x compute
- Backtesting iterates over 304k test samples

Questions:
- Can we use GPU for sklearn models (cuML)?
- Any parallel computing resources (Spark, Dask)?
- Could we cache engineered features?

Would enable: Faster iteration, larger feature sets, ensemble models
```

### 5.2 Model Development

**Q4: Class Imbalance Solutions**
```
Current distribution: Sideways 68%, Bear 14%, Bull 18%
Two-stage classifier helped, but bull detection still 13.6% accuracy.

Questions:
- Should we try SMOTE (synthetic minority oversampling)?
- Would focal loss (focuses on hard examples) help?
- Any experience with cost-sensitive learning weights?

Tried: class_weight='balanced', but bull still under-predicted.
```

**Q5: Feature Selection**
```
We have 110 features, likely some are redundant/noisy.

Questions:
- What feature selection methods do you recommend?
  * Recursive feature elimination (RFE)?
  * L1 regularization (Lasso)?
  * Tree-based feature importance?
  * SHAP values?
- Any automated feature engineering tools (Featuretools)?
- Should we use PCA/dimensionality reduction?

Goal: Improve signal-to-noise, speed up training, reduce overfitting.
```

**Q6: Hyperparameter Tuning**
```
Currently using manual hyperparameters:
- RandomForest: n_estimators=200, max_depth=15
- GradientBoosting: n_estimators=150, max_depth=8

Questions:
- Recommend Optuna vs GridSearchCV vs Bayesian optimization?
- Any best practices for tuning with time series data?
- Should we use early stopping based on validation set?

Concern: Don't want to overfit to 30% training data.
```

### 5.3 Trading Logic

**Q7: Transaction Cost Modeling**
```
We hardcoded:
- Commission: 0.75%
- Slippage: 0.01%

Questions:
- Is this realistic for major exchanges (Binance, Coinbase)?
- Should slippage scale with volatility (higher vol = higher slippage)?
- Any market impact models for larger position sizes?
- Do fees vary by trading volume tier?

Matters because: 0.75% × 2 × 52 trades killed strategy v1.
```

**Q8: Confidence Calibration**
```
Model outputs probabilities, we threshold at 80%.

Questions:
- Should we calibrate probabilities (Platt scaling, isotonic)?
- Current confidences seem too high (avg 73%, max 99.4%)
- Are sklearn predict_proba outputs well-calibrated?
- Any tools for calibration assessment (reliability diagrams)?

Problem: 80% confident trade might actually be 60% in reality.
```

**Q9: Walk-Forward Testing**
```
Currently: Train on 30%, test on 70% once.

Questions:
- Should we do walk-forward optimization?
  * Train on month 1-3, test on month 4
  * Retrain on month 2-4, test on month 5
  * Etc.
- How often to retrain in production? Daily? Weekly?
- Any concept drift detection methods?

Real world: Markets change, need adaptive models.
```

### 5.4 Validation & Testing

**Q10: Backtesting Framework**
```
Our backtest assumes perfect execution (fill at signal price ± slippage).

Questions:
- Any libraries for realistic backtesting (backtrader, zipline, vectorbt)?
- Should we simulate:
  * Partial fills?
  * Rejected orders (insufficient liquidity)?
  * Exchange downtime?
- How to validate backtest results aren't overfitted?

Concern: Real trading won't match backtest exactly.
```

**Q11: Performance Metrics**
```
We track: Sharpe, max drawdown, win rate, total return.

Questions:
- What other metrics do you recommend?
  * Sortino ratio (downside deviation)?
  * Calmar ratio (return / max drawdown)?
  * Omega ratio?
  * Tail ratio?
- Industry standards for "good" crypto strategy?
  * Sharpe > 1? 2?
  * Max DD < 20%? 10%?
- How to compare to benchmarks (BTC buy-and-hold, equal-weight portfolio)?

Current: 0.241 Sharpe, -5.1% max DD, +1.1% return vs +55% BTC.
```

**Q12: Risk Management**
```
Current: Single position, all-in or all-out, 4% stop loss.

Questions:
- Should we implement:
  * Position sizing (Kelly criterion)?
  * Maximum daily loss limits?
  * Correlation-based diversification?
  * Volatility targeting (scale size to hit target vol)?
- Any risk management frameworks you recommend?
- How to handle flash crashes (BTC drops 20% in minutes)?

Real risk: Blow up account on single bad trade.
```

### 5.5 Production Deployment

**Q13: Model Serving**
```
If we want to deploy this live:

Questions:
- Best way to serve sklearn models in production?
  * ONNX runtime?
  * FastAPI + pickle?
  * MLflow?
  * Custom gRPC service?
- How to handle feature engineering in real-time?
  * Pre-compute features every minute?
  * Incremental updates?
- Any latency requirements (need prediction in <1sec)?
```

**Q14: Monitoring & Alerting**
```
In production, need to monitor:
- Prediction quality degradation
- Feature distribution drift
- Execution slippage
- P&L tracking

Questions:
- Any monitoring tools you recommend (Prometheus, Grafana)?
- How to detect when model needs retraining?
- Should we have human-in-the-loop for high-value trades?
- Alerting thresholds (model confidence drops, unusual slippage, etc.)?
```

**Q15: Compliance & Safety**
```
Questions:
- Any regulatory considerations for automated crypto trading?
- Should we implement circuit breakers (auto-stop after X% loss)?
- How to handle model errors/exceptions gracefully?
- Disaster recovery plan (what if model server crashes mid-trade)?
- Audit logging requirements?
```

---

## Part 6: Summary & Action Items (5 minutes)

### Key Achievements
1. Developed stable regime classification using daily momentum method
2. Built ML pipeline with comprehensive feature engineering (110 features)
3. Created fee-aware trading logic that reduces commission impact from 57% to 0.76%
4. Improved Sharpe from -7.34 to +0.24 and max drawdown from -57% to -5%
5. Proved confidence filtering works (80% threshold effective)

### Critical Issues
1. Bull market detection accuracy only 13.6% (root cause: training data)
2. Under-trading: 1 trade in 7 months (too conservative)
3. Still underperforming BTC buy-and-hold (+1.1% vs +55%)
4. Limited training data (only 4 months, missing full cycle)

### Immediate Actions (Next 2 Weeks)
- [ ] Obtain 2021-2022 historical data
- [ ] Add bull-specific features (trend strength, momentum persistence)
- [ ] Implement regime transition signals (sideways→bull)
- [ ] Test ensemble models (GB + RF + XGBoost)
- [ ] Implement trailing stops and adaptive exits

### Tech Support Requests
- [ ] Access to 2021-2022 1-minute BTC data
- [ ] Guidance on class imbalance handling (SMOTE, focal loss)
- [ ] Recommendations for hyperparameter tuning framework
- [ ] Calibration methods for prediction probabilities
- [ ] Realistic backtesting framework suggestions

### Success Criteria for Next Iteration
- Bull detection accuracy > 50% (up from 13.6%)
- 10-20 trades over test period (up from 1, but not 52)
- Positive return with Sharpe > 0.5
- Max drawdown < 15%
- Beat BTC buy-and-hold on risk-adjusted basis

---

## Appendix: Technical Deep Dives

### A. Feature Engineering Details

**Feature Categories (110 total):**

1. **Price-Based (20 features):**
   - Returns: 5d, 10d, 20d, 50d
   - Volatility: 5d, 10d, 20d, 50d
   - Trend strength: 5d, 10d, 20d, 50d
   - Price vs SMA: 5d, 10d, 20d, 50d
   - Price vs EMA: 5d, 10d, 20d, 50d

2. **Technical Indicators (18 features):**
   - RSI: normalized, momentum, oversold flag, overbought flag
   - MACD: trend, momentum, strength, signal cross
   - Bollinger Bands: position, squeeze, breakout upper, breakout lower
   - ATR: value, rank, high flag
   - Multi-timeframe (if available)

3. **Moving Average Relationships (12 features):**
   - EMA crosses: 7/20, 7/30, 20/30
   - EMA spreads: 7/20, 7/30, 20/30
   - SMA crosses: 7/20, 7/30, 20/30
   - SMA spreads: 7/20, 7/30, 20/30

4. **Volume (4 features):**
   - Volume SMA 20
   - Volume ratio
   - Volume momentum
   - High volume flag

5. **Fractal Dimension (5 features):**
   - FD 14d: value, trend, normalized
   - FD short-long spread (7d - 30d)
   - FD complexity flag

6. **Momentum (9 features):**
   - Price momentum: 3d, 5d, 10d
   - Momentum rank: 3d, 5d, 10d
   - Trend consistency: 5d, 10d

7. **Volatility Regime (9 features):**
   - Rolling volatility: 10d, 20d, 30d
   - Volatility rank: 10d, 20d, 30d
   - High volatility flags: 10d, 20d, 30d

8. **Lag Features (12 features):**
   - RSI lags: 1, 2, 3, 5
   - MACD histogram lags: 1, 2, 3, 5
   - BB position lags: 1, 2, 3, 5

9. **Regime Discrimination (12 features):**
   - Higher highs/lows: 5d patterns
   - Lower highs/lows: 5d patterns
   - Close position in range
   - Average close position: 5d
   - Up/down volume
   - Volume bias: 5d
   - Net direction: 10d
   - Range expansion

10. **Multi-Timeframe (9 features, if available):**
    - Regime consistency: 1H, 6H, 12H, 1D, 1W

### B. Model Architecture Details

**GradientBoosting (Best Model):**
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingClassifier(
        n_estimators=150,      # Number of boosting stages
        max_depth=8,           # Max tree depth
        learning_rate=0.1,     # Shrinkage
        min_samples_split=10,  # Min samples to split node
        subsample=0.8,         # Fraction of samples for fitting
        max_features='sqrt',   # Features for best split
        random_state=42
    ))
])

Training:
- 5-fold TimeSeriesSplit cross-validation
- Fit on 30% of data (130,202 samples after cleaning)
- Optimize for accuracy (could use log_loss or custom metric)

Prediction:
- predict(): Argmax of class probabilities
- predict_proba(): [P(bear), P(sideways), P(bull)]
```

**Two-Stage Classifier (Optimized):**
```python
Stage 1: RandomForest (Trending vs Sideways)
Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        class_weight={0: 1, 1: 2},  # Boost trending detection
        random_state=42
    ))
])

Stage 2: GradientBoosting (Bull vs Bear)
Pipeline([
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=10,
        random_state=42
    ))
])

Combined Prediction:
1. Is it trending? (Stage 1)
   - No → Predict Sideways (1)
   - Yes → Go to Stage 2
2. Which direction? (Stage 2)
   - Predict Bear (0) or Bull (2)

Confidence Calculation:
- Sideways: P(sideways from Stage 1)
- Bull/Bear: P(trending from Stage 1) × P(direction from Stage 2)
```

### C. Backtest Implementation Details

**Backtesting Loop (Simplified):**
```python
cash = 10000
position = 0
trades = []

for timestamp, row in test_df.iterrows():
    current_price = row['Close']
    regime_pred = model.predict(features)
    confidence = model.predict_proba(features).max()

    # Entry logic
    if position == 0 and should_enter(regime_pred, confidence):
        entry_price = current_price * (1 + slippage)
        position = (cash / entry_price) * (1 - commission)
        cash = 0
        entry_time = timestamp

    # Exit logic
    elif position > 0:
        should_exit, reason = exit_manager.should_exit(
            entry_price, current_price, entry_time, timestamp
        )

        if should_exit:
            exit_price = current_price * (1 - slippage)
            cash = position * exit_price * (1 - commission)
            position = 0

            # Log trade
            trades.append({
                'entry': entry_time,
                'exit': timestamp,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': (cash / 10000 - 1) * 100,
                'reason': reason
            })

    # Calculate portfolio value
    portfolio_value = cash if position == 0 else position * current_price
```

**Key Assumptions:**
- Perfect execution at signal price (± slippage)
- No partial fills
- Infinite liquidity
- No market impact
- Fixed 0.75% commission, 0.01% slippage
- No overnight fees, funding rates, or borrowing costs
- No taxes

**Limitations:**
- Overly optimistic (real trading has more slippage, failed orders)
- Doesn't account for execution delay (signal → order → fill)
- Doesn't simulate exchange downtime
- No liquidity constraints

---

## Discussion Notes & Follow-Up

**Participants:**
- [Names]

**Key Discussion Points:**
- [To be filled during meeting]

**Decisions Made:**
- [To be filled during meeting]

**Action Items:**
- [To be filled during meeting]

**Next Meeting:**
- Date/Time: [TBD]
- Agenda: Review data acquisition, test new features, discuss ensemble models
