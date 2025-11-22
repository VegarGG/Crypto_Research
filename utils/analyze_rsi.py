import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def analyze_rsi_stability(filepath):
    print(f"Analyzing RSI stability for: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Identify timestamp column
    ts_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if not ts_cols:
        print("Error: Could not identify timestamp column")
        return
    ts_col = ts_cols[0]
    
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).set_index(ts_col)
    
    # Calculate RSI manually if not present
    if 'rsi' not in [c.lower() for c in df.columns]:
        print("Calculating RSI (14) manually...")
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    else:
        rsi_col = [c for c in df.columns if 'rsi' in c.lower()][0]
        print(f"Using existing RSI column: {rsi_col}")
        df['rsi'] = df[rsi_col]
        
    # Drop NaNs
    df = df.dropna(subset=['rsi'])
    
    # Monthly Analysis
    print("\nMonthly RSI Statistics:")
    monthly_stats = df['rsi'].resample('M').agg(['mean', 'std', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    monthly_stats.columns = ['Mean', 'Std', 'Min', 'Max', '25%', '75%']
    print(monthly_stats)
    
    # Check for drift
    print("\nRSI Distribution Drift Check:")
    first_month = df['rsi'].loc[df.index.month == df.index.month.min()]
    last_month = df['rsi'].loc[df.index.month == df.index.month.max()]
    
    print(f"First Month Mean: {first_month.mean():.2f}, Std: {first_month.std():.2f}")
    print(f"Last Month Mean: {last_month.mean():.2f}, Std: {last_month.std():.2f}")
    
    # Rolling Statistics
    df['rsi_rolling_mean'] = df['rsi'].rolling(window=1440*7).mean() # 7-day rolling mean
    
    print("\nRolling 7-Day Mean RSI Range:")
    print(f"Min: {df['rsi_rolling_mean'].min():.2f}")
    print(f"Max: {df['rsi_rolling_mean'].max():.2f}")
    
    # Regime Stability Check
    # Bullish (40-80), Bearish (20-60) - Overlap 40-60
    # Let's see how much time is spent in these ranges per month
    
    def get_regime_breakdown(x):
        bullish = ((x >= 40) & (x <= 80)).mean()
        bearish = ((x >= 20) & (x <= 60)).mean()
        extreme_overbought = (x > 80).mean()
        extreme_oversold = (x < 20).mean()
        return pd.Series({
            'Bullish_Range (40-80)': bullish, 
            'Bearish_Range (20-60)': bearish,
            'Overbought (>80)': extreme_overbought,
            'Oversold (<20)': extreme_oversold
        })

    print("\nRegime Occupancy by Month:")
    regime_stats = df['rsi'].resample('M').apply(get_regime_breakdown)
    print(regime_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze RSI Stability")
    parser.add_argument("filepath", help="Path to CSV file")
    args = parser.parse_args()
    
    analyze_rsi_stability(args.filepath)
