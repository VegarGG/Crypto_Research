# Hybrid Rule-ML Regression Strategy v2 - A/B Test

## Overview

This notebook implements a **FIXED** version of the hybrid ML regression strategy with:
- ✅ **NO data leakage** - Realistic close-to-close and VWAP targets
- ✅ **NO look-ahead bias** - 1-day regime lag
- ✅ **Realistic execution** - Next bar open entry with slippage
- ✅ **A/B testing** - Compare two target approaches

## File

- `hybrid_rule_ml_regression_strategy_2.ipynb` (25 cells)

## Key Fixes from Original Version

### 1. Regression Target (A/B Test)

**Option A: Close-to-Close**
```python
# Exit at close after 24 hours (realistic)
exit_price = df.iloc[i + lookforward_bars]['Close']
gross_return = (exit_price - entry_price) / entry_price
```

**Option B: VWAP**
```python
# Exit at volume-weighted average price
vwap = (future_window['Close'] * future_window['Volume']).sum() / future_window['Volume'].sum()
gross_return = (vwap - entry_price) / entry_price
```

**Original (WRONG)**:
```python
# Used best possible exit (unrealistic!)
max_gain = future_window['High'].max()
```

### 2. Regime Classification

**Fixed (1-day lag)**:
```python
df_daily['regime_lagged'] = df_daily['regime'].shift(1)
# Use yesterday's regime for today's trading
```

**Original (WRONG)**:
```python
# Used same day's regime (look-ahead bias!)
```

### 3. Entry Execution

**Fixed (next bar open)**:
```python
next_bar = df.iloc[i + 1]
entry_price = next_bar['Open'] * (1 + slippage)
```

**Original (WRONG)**:
```python
# Entered at current bar close (look-ahead!)
```

## Notebook Structure

### Sections 1-5: Data Preparation (Same for Both)
1. Load data and configuration
2. Apply regime classification (1-day lag)
3. Create TWO regression targets (A & B)
4. Prepare features
5. Feature selection (shared)

### Section 6: Train Model A (Close-to-Close)
- PyCaret setup with Close-to-Close target
- Compare 20+ regression algorithms
- Select best model by MAE
- Evaluate on test set

### Section 7: Train Model B (VWAP)
- PyCaret setup with VWAP target
- Compare 20+ regression algorithms
- Select best model by MAE
- Evaluate on test set

### Section 8: Backtesting Framework
- Realistic execution logic
- Entry at next bar open with slippage
- Exit slippage applied
- Fee-aware P&L calculation

### Section 9: Run Backtests
- Backtest Model A
- Backtest Model B
- Backtest Benchmark (Rule-Based Bull)
- Compare all three

### Section 10: Final Comparison
- Summary table
- Winner selection
- Comprehensive visualization:
  - Option A trades on price chart
  - Option B trades on price chart
  - Equity curves comparison

## Expected Results

Based on integrity analysis, expect:

- **Original (leaky) version**: 28.4% return, 3 trades, 100% win rate
- **Fixed version (Option A/B)**: Likely 5-15% return, 2-5 trades, 50-70% win rate

**Why lower?**
- No perfect exit timing
- 1-day regime lag
- Realistic entry execution
- Entry/exit slippage

**This is GOOD!** - Honest results you can trust for production.

## How to Run

```bash
jupyter notebook hybrid_rule_ml_regression_strategy_2.ipynb
```

Run all cells sequentially. Total runtime: ~10-15 minutes (PyCaret model comparison takes longest).

## Key Metrics to Watch

1. **Model R²**: Should be close to 0 or slightly negative (realistic)
2. **Prediction distribution**: Should be conservative (mean ~1-2%)
3. **Trade count**: Should be LOW (high threshold filters most signals)
4. **Win rate**: Should be 50-70% (not 100%!)
5. **Comparison**: Which target (Close vs VWAP) performs better?

## Production Readiness

After running this notebook, you'll know:
- ✅ Which target approach works better (A or B)
- ✅ Realistic performance expectations
- ✅ Whether ML adds value over rule-based
- ✅ Honest win rates and returns

**Next steps for production:**
1. If ML wins: Deploy winning model (A or B)
2. If rule-based wins: Skip ML, use simple regime strategy
3. Paper trade for 30 days before live deployment

## Questions Answered

1. **Does ML beat rule-based?** - A/B test will show
2. **Close-to-Close or VWAP target?** - A/B test will show
3. **Is the original 28.4% real?** - No, likely 5-15% after fixes
4. **Can we deploy this?** - Yes, if profitable after fixes

## Files Generated

- `ab_test_comparison_fixed.png` - Visualization of all 3 strategies
- Trade logs stored in results dictionaries

## Differences from v1

| Feature | v1 (Original) | v2 (Fixed) |
|---------|--------------|------------|
| Target | Best exit (leaky) | Close-to-Close & VWAP |
| Regime | Same day (leaky) | 1-day lag |
| Entry | Current bar close (leaky) | Next bar open |
| Slippage | None | 0.01% on entry/exit |
| A/B Test | No | Yes (2 models) |
| Production Ready | No | Yes |

---

**Created**: 2025-10-19
**Purpose**: A/B test realistic ML targets with no data leakage
**Status**: Ready to run
