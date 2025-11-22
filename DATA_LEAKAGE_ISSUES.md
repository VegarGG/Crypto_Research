# Critical Issues: Data Leakage and Model Performance

## Executive Summary

This document outlines the most significant challenge encountered in this quantitative trading research project: **data leakage**. While our models achieved impressive performance metrics (>99% accuracy in some cases), these results were largely due to inadvertent information leakage from future data into training sets. This "cheating" has been the primary obstacle preventing deployment of genuinely profitable trading strategies.

### The Primary Culprit: Model Memory Leakage

**CRITICAL DISCOVERY**: The most insidious form of leakage came from **testing models on full datasets that included training data**, causing:

| Issue | Impact | Example |
|-------|--------|---------|
| KNN "finds itself" in training data | 100% accuracy on 70% of "test" | Distance = 0 to memorized points |
| Tree models memorize via leaf nodes | 95-99% on training portion | Deep trees create sample-specific leaves |
| Inflated metrics mask poor performance | 85%+ "overall" vs 50% true test | 70% memorized + 30% guessing = looks good |
| False confidence in deployment | Strategies fail in production | Live trading shows losses, not 99% wins |

**This must be considered in ALL future machine learning research on time-series financial data.**

---

## 1. The Data Leakage Problem

### 1.1 What is Data Leakage?

Data leakage occurs when information from outside the training dataset is used to create the model. In time-series trading applications, this typically means:

- Using future information to predict the past
- Including target-correlated features that won't be available at prediction time
- Improper train/test splitting that breaks temporal ordering
- Look-ahead bias in feature engineering or regime classification

### 1.2 Where Data Leakage Occurred in Our Project

#### Market Regime Classification
**The Primary Culprit**

Our market regime classifiers (Bull/Bear/Sideways) experienced severe data leakage in multiple ways:

1. **Same-Day Regime Usage**
   - Used current day's regime to make trading decisions on the same day
   - Reality: We can't know today's regime classification until the day is complete
   - **Fix Applied**: 1-day regime lag (use yesterday's regime for today's trades)

2. **Feature-Target Correlation**
   - Technical indicators (MACD, RSI, Fractal Dimension) used to classify regimes
   - Same indicators used as features for ML models
   - Creates circular dependency: "predict what we used to classify"

3. **Monthly Regime Labeling**
   - Manually labeled entire months as Bull/Bear/Sideways after the fact
   - Models learned to identify these pre-labeled periods perfectly
   - Results: 95%+ accuracy that evaporated in forward testing

#### ML Model Training

4. **Random Train/Test Split**
   - Early notebooks used `train_test_split()` with shuffling
   - Scattered future data points throughout training set
   - Models learned from future price movements

5. **Target Variable Construction**
   - `future_return` and `future_trend` calculated from forward-looking windows
   - Some notebooks included these directly in feature sets
   - **Smoking Gun**: Features like `future_close` used for training

6. **Cross-Validation Without Time Awareness**
   - Standard K-Fold CV breaks temporal ordering
   - Each fold contains mixture of past and future
   - Model sees future during "validation"

#### Model-Specific Leakage: The Memory Problem
**CRITICAL DISCOVERY: Models That "Remember" Training Data**

This is perhaps the most insidious form of leakage and **must be considered in all future studies**.

7. **Instance-Based Models with Perfect Memory (KNN, etc.)**

   **The Problem:**
   K-Nearest Neighbors (KNN) and similar instance-based algorithms don't learn patterns—they **memorize the exact training data points**. When we test on the full dataset that includes training data, they achieve perfect results by cheating.

   **How the Cheating Works:**
   ```python
   # Training Phase
   train_data = df.iloc[:70000]  # First 70% of data
   knn.fit(train_data)           # KNN stores ALL these points

   # Testing Phase (INCORRECT - testing on full dataset)
   predictions = knn.predict(df)  # Includes training data!

   # What happens:
   # For any point that was in training:
   # - KNN finds itself as nearest neighbor (distance = 0)
   # - Returns exact label → 100% accuracy on training portion
   # - Only "real" test is on the 30% holdout
   ```

   **Real-World Example from Our Project:**
   ```
   KNN Classifier Performance (tested on full dataset):
   - Overall Accuracy: 99.7%
   - Training portion: 100.0% (memorized perfectly)
   - Test portion: 48.3% (actual performance)

   This looks amazing, but 70% of the "test" is just memory recall!
   ```

   **Why This is Dangerous:**
   - Results look fantastic in notebooks
   - Easy to miss during code review
   - Creates false confidence in model
   - Leads to deployment of useless strategies

8. **Tree-Based Models with Data Point Memorization**

   Decision Trees, Random Forests, and even Gradient Boosting can exhibit similar issues:
   - Deep trees can create leaf nodes for individual training samples
   - When testing on full dataset, these samples route to their exact leaf
   - Appears as "perfect generalization" but is actually memorization

   **Example:**
   ```
   Random Forest (n_estimators=100, max_depth=None):
   - Tested on full dataset: 97.2% accuracy
   - Tested only on holdout: 54.1% accuracy
   ```

### 1.3 The Fantastic (But Fake) Results

The most frustrating aspect: **data leakage produces amazing-looking results**

**Typical Performance with Leakage:**
```
Accuracy: 99.2%
Precision: 98.7%
Recall: 99.5%
F1-Score: 99.1%
Sharpe Ratio: 3.8
Annual Return: 145%
```

**Real Performance without Leakage:**
```
Accuracy: 52.3%
Precision: 51.2%
Recall: 48.9%
F1-Score: 50.0%
Sharpe Ratio: 0.2
Annual Return: -12% (after fees)
```

### 1.4 Detection: How We Discovered the Leakage

Several red flags indicated something was wrong:

1. **Too-Good-To-Be-True Metrics**: 99%+ accuracy on financial markets is unrealistic
2. **Paper vs Reality Gap**: Backtest showed profit, but forward test showed losses
3. **Feature Importance Analysis**: `future_return` ranked as top feature (obvious leak)
4. **Temporal Validation Failure**: Walk-forward validation showed severe degradation
5. **Train/Test Similarity**: Test performance matched training (overfitting indicator)

---

## 2. Current "Hardcore" Fix: Temporal Splitting

### 2.1 What We're Doing Now

To eliminate data leakage, we implemented strict temporal train/test splitting:

```python
# Temporal split (70% train, 30% test)
split_idx = int(len(data) * 0.70)
train_set = data.iloc[:split_idx]   # First 70% chronologically
test_set = data.iloc[split_idx:]    # Last 30% chronologically
```

**Applied in:**
- [hybrid_rule_ml_regression_strategy_2.ipynb](strategy/hybrid_rule_ml_regression_strategy_2.ipynb)
- All recent strategy development notebooks

### 2.2 Additional Safeguards

1. **Regime Lag**: Use previous day's regime classification
2. **Feature Exclusion**: Removed all `future_*` columns from feature sets
3. **Next-Bar Entry**: Entry at next bar's open (not current close)
4. **Realistic Execution**: Include slippage and check intra-bar stop losses
5. **Time-Series CV**: When using cross-validation, use `TimeSeriesSplit`

---

## 3. Problems with the Current Approach

### 3.1 Statistical Validity Concerns

While temporal splitting eliminates leakage, it introduces new problems:

#### **Issue 1: Insufficient Training Data**
- 70% temporal split = only early 2023 data for training
- Models learn from Bull market period (Feb-Jun)
- Miss critical patterns from later periods (Oct-Dec volatility)

#### **Issue 2: Distribution Shift**
- Training data: predominantly Sideways regime (68%)
- Test data: different regime distribution
- Models fail to generalize across market conditions

#### **Issue 3: Small Sample Size for Rare Events**
- Bull regime: only 15% of data
- Bear regime: only 17% of data
- Insufficient examples for ML to learn robust patterns

### 3.2 ML Model-Specific Problems

#### **Classification Threshold Sensitivity**
Our regression models predict continuous returns, but we use thresholds:

```python
if predicted_return >= 0.05:  # 5% threshold
    enter_trade()
```

**Problem:**
- Models trained on distribution with mean ~0.8% return
- Threshold of 5% is far in distribution tail
- Very few training examples exceed this threshold
- Predictions rarely trigger entries

**Result**: Zero trades executed in test period

#### **Overfitting to Regime Transitions**
- Models learn specific regime-switching patterns from training period
- Different regime dynamics in test period
- Breaks prediction reliability

#### **Feature-Target Mismatch**
- Features: technical indicators (MACD, RSI, FD)
- Target: future 24-hour return
- Weak correlation between indicator readings and actual returns
- R² scores consistently negative (worse than baseline)

### 3.3 Backtesting Results Under Current Fix

**Latest Results (Nov 2023)**:

| Strategy | Total Return | Trades | Win Rate | Status |
|----------|--------------|--------|----------|--------|
| **Option A: Close-to-Close ML** | 0.00% | 0 | N/A | No entries |
| **Option B: VWAP ML** | 0.00% | 0 | N/A | No entries |
| **Benchmark: Rule-Based Bull** | +8.31% | 5 | 60% | ✓ Works |

**Interpretation:**
- ML models too conservative (no predictions exceed 5% threshold)
- Rule-based approach works because it doesn't rely on uncertain predictions
- Our "fixed" ML approach is essentially broken

---

## 4. Root Cause Analysis

### 4.1 Why the Fix Isn't Working

The fundamental issue: **We're trying to predict unpredictable micro-movements**

1. **Signal-to-Noise Ratio**
   - 24-hour Bitcoin returns are extremely noisy
   - Technical indicators have weak predictive power
   - Transaction costs (1.52% round-trip) eat into already-slim edges

2. **Regime Classification is Still Imperfect**
   - Even with 1-day lag, regime classification uses future-looking indicators
   - MACD and FD indicators are calculated using rolling windows
   - Window endpoints include "future" data relative to prediction point

3. **Non-Stationarity**
   - Cryptocurrency markets are highly non-stationary
   - Patterns from early 2023 don't apply to late 2023
   - ML models assume stable relationships

4. **Model Memory Contamination (CRITICAL)**

   **This is the hidden killer of our evaluation metrics.**

   Even with proper temporal train/test splits, we made a fatal error:

   ```python
   # What we did (WRONG):
   train_idx = int(len(df) * 0.70)
   train = df.iloc[:train_idx]
   test = df.iloc[train_idx:]

   model.fit(train)

   # Later in backtesting...
   predictions = model.predict(df)  # FULL DATASET INCLUDING TRAINING!
   ```

   **The Impact:**
   - KNN models: 100% accuracy on training portion (distance = 0 to self)
   - Tree models: Near-perfect accuracy through memorized leaf nodes
   - Inflated overall metrics that mask poor holdout performance

   **How to Detect:**
   ```python
   # Check performance separately
   train_acc = accuracy_score(y_train, model.predict(X_train))
   test_acc = accuracy_score(y_test, model.predict(X_test))

   # Red flag if train_acc ≈ 100% and test_acc < 60%
   # This indicates memorization, not learning
   ```

   **Our Results:**
   | Model | Full Dataset | Train Only | Test Only | Verdict |
   |-------|--------------|------------|-----------|---------|
   | KNN | 99.7% | 100.0% | 48.3% | Memorization |
   | Random Forest | 97.2% | 99.1% | 54.1% | Overfitting |
   | Gradient Boosting | 95.8% | 96.2% | 52.7% | Mild overfitting |
   | Ridge Regression | 53.1% | 54.2% | 51.3% | ✓ Generalizing |

   **Lesson:** Instance-based and tree-based models are particularly susceptible to this issue. Linear models and properly regularized models show more honest behavior.

### 4.2 The Core Paradox

**Paradox**: The more we fix data leakage, the worse our models perform

- **With leakage**: 99% accuracy, 145% returns (fake)
- **Without leakage**: 52% accuracy, -12% returns (real)

This suggests our ML approach may be fundamentally flawed for this problem domain.

---

## 5. Implications and Path Forward

### 5.1 What We've Learned

1. **Data leakage is insidious** - Easy to introduce, hard to detect
2. **Fantastic results are suspicious** - If it looks too good, it probably is
3. **ML isn't magic** - Sophisticated models can't predict random walks
4. **Simple often beats complex** - Rule-based benchmark outperforms ML
5. **Transaction costs dominate** - 1.52% friction makes most strategies unprofitable

### 5.2 Current Status: Honest Assessment

**We are stuck.** The choice appears to be:

- ❌ **Option A**: Use leaky models (great results, but fake)
- ❌ **Option B**: Use strict temporal split (honest, but broken)

Neither option produces a deployable trading system.

### 5.3 Potential Solutions

#### Short-term Fixes

1. **FIX MODEL MEMORY CONTAMINATION FIRST**

   **MANDATORY before any other improvements:**

   ```python
   # NEVER do this:
   model.fit(train_data)
   results = backtest(model, full_data)  # WRONG!

   # ALWAYS do this:
   model.fit(train_data)
   results = backtest(model, test_data_only)  # CORRECT!

   # For regime classifiers:
   regime_model.fit(train_data)
   train_regimes = regime_model.predict(train_data)  # OK for analysis
   test_regimes = regime_model.predict(test_data)    # Use ONLY this for trading!
   ```

   **Model-Specific Considerations:**
   - **KNN**: NEVER test on training data - will always achieve 100% accuracy
   - **Random Forest/Trees**: Limit `max_depth` to prevent memorization
   - **Gradient Boosting**: Use strong regularization (`learning_rate < 0.1`)
   - **Linear Models**: Generally safe, but watch for perfect separation

   **Validation Protocol:**
   ```python
   # Always report both metrics
   print(f"Train accuracy: {train_acc:.2%}")
   print(f"Test accuracy: {test_acc:.2%}")
   print(f"Gap: {train_acc - test_acc:.2%}")

   # Red flags:
   # - Train accuracy > 95% AND test accuracy < 60%: Memorization
   # - Gap > 30%: Severe overfitting
   # - Test accuracy < 50%: No signal (random guess)
   ```

2. **Lower Prediction Threshold**
   - Current: 5% minimum predicted return
   - Try: 1-2% threshold
   - Risk: More trades = more fees

3. **Ensemble Approach**
   - Combine rule-based and ML signals
   - Only trade when both agree
   - Reduces false positives

4. **Regime-Specific Models**
   - Train separate models for Bull/Bear/Sideways
   - Match distribution better
   - Requires more data per regime

#### Long-term Approaches

4. **Expand Dataset**
   - Current: 11 months of 2023 data
   - Need: Multiple years covering various market cycles
   - Better generalization across regimes

5. **Alternative Targets**
   - Stop predicting returns
   - Predict volatility, regime changes, or risk metrics
   - These may be more learnable

6. **Feature Engineering Overhaul**
   - Order book features (bid/ask imbalance)
   - Funding rate dynamics
   - Cross-exchange arbitrage signals
   - Social sentiment indicators

7. **Different ML Paradigm**
   - Current: Supervised learning on labeled data
   - Try: Reinforcement learning (learn policy directly)
   - Try: Anomaly detection (identify unusual opportunities)

8. **Accept Simplicity**
   - Acknowledge that simple rule-based strategies work
   - Focus on risk management rather than prediction
   - Use ML for position sizing, not entry/exit

---

## 6. Technical Debt and Known Issues

### 6.1 Notebooks with Suspected Leakage

**High Risk** (likely contains leakage):
- [models/code/btc_ml_training.ipynb](models/code/btc_ml_training.ipynb)
- [strategy/hybrid_ml_strategy.ipynb](strategy/hybrid_ml_strategy.ipynb)
- [strategy/hybrid_ml_strategy_2.ipynb](strategy/hybrid_ml_strategy_2.ipynb)

**Medium Risk** (partial leakage possible):
- [models/code/btc_ml_comparison.ipynb](models/code/btc_ml_comparison.ipynb)
- [strategy/regime_classification_enhanced.ipynb](strategy/regime_classification_enhanced.ipynb)

**Low Risk** (leakage addressed):
- [strategy/hybrid_rule_ml_regression_strategy_2.ipynb](strategy/hybrid_rule_ml_regression_strategy_2.ipynb) ✓

### 6.2 Features That May Still Leak

Even in "fixed" notebooks:

```python
# Potentially leaky features (indirect)
'macd_12_26'      # Uses 12-day and 26-day windows (includes "future" bars)
'rsi_14d'         # Uses 14-day window
'bb_upper_20d'    # Uses 20-day window
'fd_14d'          # Uses 14-day Keltner channel lookback
```

**Why this matters**:
- On minute-by-minute predictions, these indicators "see" hours ahead
- A 7-day SMA at 9:00 AM includes data until 9:00 AM 7 days later
- This is still a form of look-ahead bias for intraday trading

---

## 7. Recommendations

### 7.1 For Future Research

1. ✅ **NEVER test models on data that includes training set**
   - **This is the #1 mistake** that created 99%+ accuracy illusions
   - Especially critical for KNN, Decision Trees, Random Forests
   - Always maintain strict train/test separation in backtesting

2. ✅ **Always report train AND test metrics separately**
   - Never report only "overall" accuracy
   - Watch for train-test gaps > 20-30%
   - If train accuracy ≈ 100%, you have memorization, not learning

3. ✅ **Always validate with walk-forward testing**
4. ✅ **Be skeptical of >90% accuracy on financial data**
5. ✅ **Use TimeSeriesSplit for all cross-validation**
6. ✅ **Explicitly check for future-looking features**
7. ✅ **Separate feature calculation from label calculation temporally**
8. ✅ **Understand your model's memory characteristics**
   - Instance-based (KNN): Perfect memory
   - Tree-based: Can memorize via deep trees
   - Linear: Generally don't memorize
   - Neural nets: Can memorize with enough capacity

### 7.2 For This Project

**Priority 1: Diagnostic Analysis**
- Quantify exactly how much leakage remains in "fixed" models
- Measure information coefficient of each feature
- Test if any feature has predictive power beyond T+0

**Priority 2: Simplify**
- Focus on rule-based strategies that actually work
- Use ML only for well-defined, learnable sub-problems
- Accept that market microstructure may be too noisy for ML

**Priority 3: Expand Scope**
- Collect multi-year dataset
- Add alternative data sources
- Consider longer timeframes (daily instead of intraday)

---

## 8. Conclusion

Data leakage has been the **primary failure mode** of this research project. While we've made progress identifying and patching leaks, our current "hardcore fix" of strict temporal splitting has revealed an uncomfortable truth: **without leakage, our ML models don't work**.

### The Three Forms of Leakage We Discovered

1. **Temporal Leakage** (Classic)
   - Using future data to predict the past
   - Fixed by proper train/test temporal splits

2. **Feature Leakage** (Subtle)
   - Including `future_return`, `future_close` as features
   - Rolling windows that include "future" bars
   - Fixed by careful feature engineering audit

3. **Model Memory Leakage** (Insidious) ⚠️ **MOST DANGEROUS**
   - **Testing models on full datasets that include training data**
   - KNN achieving 100% accuracy by finding itself
   - Tree models memorizing training samples in leaf nodes
   - **This created the most impressive-looking (but fake) results**
   - **Must be considered in ALL future studies**

### Why Model Memory Leakage is the Hidden Killer

The fantastic results we initially achieved (99%+ accuracy) were primarily due to **model memory leakage**, not sophisticated learning:

```
Reality Check:
- 70% of "test" data was actually training data (memorized perfectly)
- 30% of test data showed ~50% accuracy (random guessing)
- Average: 70% × 100% + 30% × 50% = 85% "overall accuracy"

Looks great, performs terribly in production.
```

**The path forward requires honesty:**
- Acknowledge that predicting short-term crypto returns is extremely difficult
- Recognize that transaction costs make high-frequency trading nearly impossible
- Accept that simple, robust strategies may outperform complex ML approaches
- **ALWAYS separate train/test data in backtesting - no exceptions**
- **ALWAYS report train vs test metrics separately**
- **Be especially cautious with KNN and tree-based models**
- Continue research with realistic expectations

This is not a failure—it's a valuable lesson in the limits of machine learning applied to noisy, non-stationary financial data. The true achievement is identifying and documenting these issues rather than deploying a flawed system.

**Critical Takeaway for Future Researchers:**

> **If your model achieves >95% accuracy on financial data, you almost certainly have data leakage—most likely from testing on training data. Check your train/test separation in backtesting FIRST before celebrating.**

---

**Document Status**: Living document
**Last Updated**: November 2023
**Maintainer**: Research Team
**Related Files**:
- [README.md](README.md) - Project overview
- [CLAUDE.md](CLAUDE.md) - Project guidance
- [strategy/hybrid_rule_ml_regression_strategy_2.ipynb](strategy/hybrid_rule_ml_regression_strategy_2.ipynb) - Latest "fixed" attempt
