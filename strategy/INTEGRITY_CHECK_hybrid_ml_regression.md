# Integrity Check: Hybrid ML Regression Strategy
**Date**: 2025-10-19
**Purpose**: Verify no data leakage, look-ahead bias, or unrealistic assumptions before production deployment

---

## EXECUTIVE SUMMARY

### Critical Issues Found:
1. **SEVERE DATA LEAKAGE** in regression target calculation
2. **Look-ahead bias** in backtesting execution
3. **Unrealistic execution assumptions**
4. **Distribution shift** between train/test not addressed
5. **Production deployment blockers** identified

### Recommendation:
**DO NOT DEPLOY TO PRODUCTION** without fixing critical issues below.

---

## 1. DATA LEAKAGE ANALYSIS

### Issue 1.1: CRITICAL - Future Data in Target Calculation

**Location**: Cell 7 - `create_regression_target()`

```python
for i in range(len(df)):
    future_window = df.iloc[i+1:i+1+lookforward_bars]  # LOOKS 24 HOURS AHEAD
    max_gain = (future_window['High'].max() - entry_price) / entry_price
    max_loss = (future_window['Low'].min() - entry_price) / entry_price
```

**Problem**:
- The model is trained to predict the **BEST POSSIBLE RETURN** over next 24 hours
- This assumes perfect exit timing (selling exactly at the high)
- **IN PRODUCTION, YOU CANNOT KNOW THE FUTURE HIGH/LOW**
- This creates massive overfitting - model learns patterns of "when does price go up" but cannot execute the perfect exit

**Impact**:
- Backtest results are UNREALISTICALLY OPTIMISTIC
- Model trained on unachievable targets
- Real-world performance will be significantly worse

**Evidence**:
```
Regression Target Distribution (Bull Regime Only):
  Mean: 0.0274 (2.74%)   <-- Average "best achievable" return
  Max: 0.1648 (16.48%)   <-- Perfect 16% exit timing
```

**Fix Required**:
```python
# WRONG - Uses future perfect information
gross_return = min(max_gain, 0.20)

# CORRECT - Use realistic exit (e.g., close-to-close return)
future_close = df.iloc[i + lookforward_bars]['Close']
gross_return = (future_close - entry_price) / entry_price
```

---

### Issue 1.2: CRITICAL - Regime Classification Uses Future Data

**Location**: Cell 5 - `classify_momentum_regimes_daily()`

**Problem**:
The regime is calculated on daily bars and then **mapped back to ALL 1-minute bars of that day**:

```python
df_result['period'] = df_result.index.floor('D')
regime_map = dict(zip(df_daily.index, df_daily['regime']))
df_result['regime'] = df_result['period'].map(regime_map)
```

**Impact**:
- At 00:00 on 2023-10-23, you KNOW the regime for the ENTIRE day
- But regime is calculated using the day's close, volume, etc.
- **You cannot know the regime at market open**

**Example**:
```
2023-10-23 00:00:00  -->  Regime = Bull (2.0)
But Bull regime determined by:
  - MACD at day's close
  - Price momentum using day's close
  - Volatility using full day's data
```

**Fix Required**:
- Use PREVIOUS day's regime, or
- Calculate regime based on data available AT THE TIME (e.g., only use prior bars)

---

## 2. TRAIN/TEST SPLIT INTEGRITY

### Issue 2.1: PASS - Temporal Split Correct

**Location**: Cell 9

```python
split_idx = int(len(train_data_clean) * CONFIG['train_test_split'])
train_set = train_data_clean.iloc[:split_idx]  # First 70%
test_set = train_data_clean.iloc[split_idx:]   # Last 30%
```

**Status**: CORRECT
- Train: 2023-03-14 to 2023-10-02
- Test: 2023-10-02 to 2023-12-06
- No overlap, chronological order preserved

---

### Issue 2.2: WARNING - Distribution Shift Not Addressed

```
Train mean target: 0.0243 (2.43%)
Test mean target:  0.0345 (3.45%)  <-- 42% higher!
```

**Problem**:
- Test period has significantly higher returns
- Model trained on lower-return period
- Predicts conservatively (1.41% mean) when actual is 3.45%

**Impact**:
- Model appears to "work" but only because test period was unusually bullish
- Reverse market conditions = model fails

**Fix Required**:
- Use walk-forward validation
- Test on multiple market regimes
- Add regime-specific models

---

## 3. BACKTESTING EXECUTION REALISM

### Issue 3.1: CRITICAL - Instant Execution Assumption

**Location**: Cell 16 - `backtest_hybrid_regression()`

```python
for i in range(len(df)):
    row = df.iloc[i]
    price = row['Close']  # <-- Assumes you can trade at bar close

    if predicted_return >= min_return:
        position = {
            'entry_price': price,  # <-- Entry at close price
        }
```

**Problem**:
- **You get the prediction AFTER the bar closes**
- But code enters trade AT THE CLOSE PRICE
- In reality, you'd enter at NEXT bar's open (with slippage)

**Impact**:
- **Look-ahead bias**: Uses information not available at trade time
- Trades executed at better prices than possible in reality

**Fix Required**:
```python
# CORRECT - Enter at NEXT bar
if predicted_return >= min_return:
    if i + 1 < len(df):
        next_bar_price = df.iloc[i + 1]['Open']  # Next bar open
        position = {
            'entry_price': next_bar_price * (1 + slippage),  # Add slippage
        }
```

---

### Issue 3.2: WARNING - No Slippage on Entries

**Current**:
```python
entry_price = price  # Direct price, no slippage
```

**Problem**:
- 1-minute bars on BTC can have 0.1-0.5% spread
- Market orders have slippage
- Large trades move the market

**Fix Required**:
```python
entry_price = price * (1 + entry_slippage)  # Add realistic slippage
```

---

### Issue 3.3: WARNING - Exit Logic Unrealistic

**Location**: Cell 15 - `FeeAwareExitManager.should_exit()`

```python
def should_exit(self, entry_price, current_price, ...):
    # Rule 1: Stop loss
    if net_pnl <= -stop_loss_pct:
        return True

    # Rule 2: Regime change
    if current_regime != entry_regime:
        return True
```

**Problem with Stop Loss**:
- Checks stop loss at **bar close** only
- In reality, stop loss can be hit INTRA-BAR
- 1-minute bars can have large wicks

**Example**:
```
Entry: $30,000
Stop: $28,800 (-4%)
Bar: Open=$29,500, High=$29,600, Low=$28,500, Close=$29,400

Code result: Stop NOT hit (close at $29,400)
Reality: Stop WAS hit (low at $28,500)
```

**Fix Required**:
```python
# Check intra-bar movement
if max_loss_intrabar <= -stop_loss_pct:
    exit_price = entry_price * (1 - stop_loss_pct)
    return True, "Stop loss hit"
```

---

## 4. FEATURE INTEGRITY

### Issue 4.1: PASS - No Future Features Used

**Verified**:
```python
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'regime', 'ml_target',
                'future_close', 'future_return', 'future_trend']  # Correctly excluded
```

**Status**: CORRECT
- All `future_*` columns excluded
- Only lagging indicators used (SMA, EMA, MACD, RSI, BB, ATR, FD)

---

### Issue 4.2: WARNING - Indicator Calculation Not Shown

**Problem**:
- Features like `ema_7d`, `macd_12_26` are loaded from CSV
- No visibility into how they were calculated
- **Risk**: If indicators use future data, model is compromised

**Recommendation**:
- Audit feature calculation in data preparation pipeline
- Ensure all indicators are strictly backward-looking

---

## 5. MODEL PERFORMANCE RED FLAGS

### Issue 5.1: WARNING - Negative R² on Test Set

```
Test R²: -0.3548
```

**Meaning**:
- Model performs WORSE than predicting the mean
- Essentially useless for prediction
- Only "works" in backtest due to 5% threshold filter

**Why It Still Made Money**:
```
Predictions > 5%: 3 (0.3%)  <-- Only 3 predictions triggered trades
```

The model:
1. Predicts conservatively (mean 1.41%)
2. Only 3 times predicted >5%
3. Those 3 times happened to be correct

**This is LUCK, not skill**

---

### Issue 5.2: CRITICAL - Cherry-Picking Trades

**Current Logic**:
```python
if predicted_return >= min_return:  # 5% threshold
    position = {...}
```

**Problem**:
- Model makes 6,893 predictions
- Only 3 exceed 5% threshold
- **All 3 were winners (100% win rate)**

**Is This Realistic?**
- Model has negative R²
- Predicts mean 1.41% vs actual 3.45%
- Yet the 3 predictions >5% were ALL correct?

**Likely Explanation**:
- Data leakage from future perfect exit timing
- These 3 periods had obvious momentum
- Model "got lucky" on a tiny sample

**Production Risk**:
- Next 3 predictions >5% could be 0% win rate
- Sample size too small for statistical significance

---

## 6. PRODUCTION DEPLOYMENT ISSUES

### Issue 6.1: CRITICAL - Real-Time Regime Detection

**Current**: Regime known for entire day at 00:00
**Reality**: You need to determine regime in real-time

**Production Requirements**:
```python
# At 2023-10-23 08:32:00 (when model wants to enter)
# You need regime, but you only have:
#   - Prior day's close: 2023-10-22 23:59:00
#   - Current intraday data up to 08:32:00

# CANNOT use today's close (not available yet)
```

**Fix Required**:
- Use previous day's regime, OR
- Use intraday regime calculation (less stable)

---

### Issue 6.2: CRITICAL - Model Predictions Take Time

**Current**: Prediction available instantly
**Reality**:
- Fetch data from exchange (100-500ms)
- Calculate 15 features (50-200ms)
- Run ExtraTreesRegressor (10-100ms)
- Total latency: 200-800ms

**Impact**:
- By the time you get prediction, price has moved
- 1-minute strategy needs <1 second execution
- Current architecture may be too slow

**Fix Required**:
- Pre-calculate features
- Keep model in memory
- Optimize prediction pipeline

---

### Issue 6.3: WARNING - PyCaret Dependency in Production

**Current**: Uses PyCaret's `predict_model()`

**Problem**:
```python
pred_df = predict_model(ml_model, data=features_df)
```

**Production Issues**:
- PyCaret adds overhead (normalization, validation, logging)
- Slower than raw sklearn
- Harder to version control
- Dependency on PyCaret version

**Fix Required**:
- Extract raw sklearn model
- Use direct `.predict()` calls
- Remove PyCaret dependency for production

---

### Issue 6.4: CRITICAL - No Handling of Missing Data

**Current**:
```python
if not features_df.isnull().any().any():
    # Make prediction
```

**Production Problem**:
- What if exchange API fails?
- What if feature calculation errors?
- **Code silently skips prediction - you miss trades**

**Fix Required**:
```python
# Add fallback logic
if features_df.isnull().any().any():
    logger.warning(f"Missing features at {timestamp}")
    # Use previous regime, or
    # Skip this bar, or
    # Use cached features
```

---

## 7. SUSPICIOUS RESULTS INVESTIGATION

### Question: Why Did Hybrid Beat Rule-Based?

**Hybrid**: 3 trades, 100% win rate, +28.4%
**Rule-Based**: 5 trades, 60% win rate, +22.6%

**Comparison**:
```
Trade #1 (Hybrid):  Entry 10/19 08:32 → Exit 10/21, +3.07%
Trade #1 (Rule):    Entry 10/17 00:00 → Exit 10/21, +2.44%

Trade #2 (Both):    Entry 10/23 00:00 → Exit 10/26, +13.52%  <-- SAME TRADE

Trade #3 (Hybrid):  Entry 12/03 03:43 → Exit 12/06, +9.75%
Trade #3 (Rule):    Entry 12/03 00:00 → Exit 12/06, +9.39%
```

**Observation**:
- Hybrid avoided 2 losing trades (Rule lost -3.39% and -0.25%)
- Hybrid caught same 3 winning periods as Rule-Based
- **Difference is TRADE AVOIDANCE, not better exits**

**Is This Skill or Luck?**
- Avoided Oct 2 trade (entered 10/19 instead) - Rule entered too early
- Avoided Nov 10 trade - Model didn't predict >5% (correct!)
- Avoided entering on regime change - better timing

**Verdict**: Possibly legitimate, but sample size too small (only 2 avoided trades)

---

## 8. RECOMMENDED FIXES (PRIORITY ORDER)

### P0 - CRITICAL (Must Fix Before Production)

1. **Fix Regression Target**
   ```python
   # Replace "best achievable" with realistic close-to-close
   future_close = df.iloc[i + lookforward_bars]['Close']
   gross_return = (future_close - entry_price) / entry_price
   ```

2. **Fix Regime Look-Ahead**
   ```python
   # Use previous day's regime
   df_result['regime'] = df_result['regime'].shift(1440)  # Shift by 1 day in minutes
   ```

3. **Fix Entry Execution**
   ```python
   # Enter at NEXT bar open with slippage
   next_bar = df.iloc[i + 1]
   entry_price = next_bar['Open'] * (1 + slippage)
   ```

4. **Fix Stop Loss Check**
   ```python
   # Check intra-bar movement
   max_drawdown = (future_window['Low'].min() - entry_price) / entry_price
   if max_drawdown <= -stop_loss_pct:
       exit_price = entry_price * (1 - stop_loss_pct - slippage)
   ```

---

### P1 - HIGH (Should Fix)

5. **Add Slippage to All Trades**
   ```python
   entry_slippage = 0.0005  # 0.05%
   exit_slippage = 0.0005
   ```

6. **Use Walk-Forward Validation**
   - Test on rolling windows
   - Retrain monthly
   - Validate on multiple market conditions

7. **Add Realistic Latency**
   ```python
   # Prediction available at bar close + latency
   execution_bar = i + 1 if latency < 60 else i + 2
   ```

---

### P2 - MEDIUM (Nice to Have)

8. **Add Position Sizing**
   ```python
   # Don't use full capital on each trade
   position_size = capital * kelly_fraction
   ```

9. **Add Maximum Holding Period**
   ```python
   # Don't hold forever if regime doesn't change
   if bars_in_trade > max_hold_bars:
       exit_position()
   ```

10. **Add Market Condition Filters**
    ```python
    # Don't trade during low liquidity
    if row['Volume'] < min_volume:
        skip_trade()
    ```

---

## 9. PRODUCTION READINESS CHECKLIST

### Data Pipeline
- [ ] Real-time data feed tested
- [ ] Feature calculation verified lag-free
- [ ] Backup data sources configured
- [ ] Data quality checks implemented

### Model Deployment
- [ ] Model serialization/deserialization tested
- [ ] Version control for models
- [ ] Rollback procedure documented
- [ ] Performance monitoring setup

### Execution
- [ ] Order execution tested on testnet
- [ ] Slippage measurement implemented
- [ ] Position management tested
- [ ] Risk limits enforced

### Monitoring
- [ ] Real-time P&L tracking
- [ ] Prediction accuracy logging
- [ ] Anomaly detection alerts
- [ ] Performance degradation detection

### Safety
- [ ] Kill switch implemented
- [ ] Maximum drawdown protection
- [ ] API rate limiting handled
- [ ] Error recovery procedures

---

## 10. FINAL VERDICT

### Can This Strategy Go Live?

**NO - Not in current state**

**Critical Blockers**:
1. Data leakage in target (perfect exit timing)
2. Regime look-ahead bias (using future data)
3. Entry execution at unavailable prices
4. Stop loss logic ignores intra-bar movement

**After Fixes**:
1. Retrain with realistic close-to-close targets
2. Fix regime lag to use previous day
3. Fix entry/exit to next bar with slippage
4. Rerun backtest - expect MUCH WORSE results

**Expected Impact**:
- Returns will drop 30-50%
- Win rate will decrease
- May underperform rule-based strategy
- Need larger sample size for validation

**Recommendation**:
1. Fix all P0 issues
2. Retrain and re-backtest
3. If still profitable, paper trade for 30 days
4. Monitor actual vs expected performance
5. Only then consider live deployment with small capital

---

## APPENDIX: Code Audit Trail

### Verified Clean:
- ✅ Train/test split temporal (no leakage)
- ✅ Feature selection (no future features)
- ✅ Fee calculation (realistic 1.52% round-trip)
- ✅ Commission and slippage in config

### Needs Fixing:
- ❌ Regression target uses future perfect information
- ❌ Regime classification uses intraday future data
- ❌ Entry execution at bar close (should be next bar open)
- ❌ Stop loss doesn't check intra-bar
- ❌ No entry/exit slippage applied
- ⚠️ Model has negative R² (not actually predictive)
- ⚠️ Only 3 trades triggered (sample size too small)
- ⚠️ Test period unusually bullish (distribution shift)
