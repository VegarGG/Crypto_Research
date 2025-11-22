import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def check_data_integrity(filepath):
    print(f"Checking integrity of: {filepath}")
    
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
    print(f"Using timestamp column: {ts_col}")
    
    # Parse timestamps
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col)
    
    # 1. Check for duplicates
    duplicates = df[ts_col].duplicated().sum()
    print(f"Duplicate timestamps: {duplicates}")
    
    # 2. Check for gaps
    df['time_diff'] = df[ts_col].diff()
    # Assuming 1-minute data
    expected_diff = pd.Timedelta(minutes=1)
    gaps = df[df['time_diff'] > expected_diff]
    
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df[ts_col].min()} to {df[ts_col].max()}")
    
    if not gaps.empty:
        print(f"Found {len(gaps)} gaps > 1 minute")
        print("Top 5 gaps:")
        print(gaps[['time_diff']].sort_values('time_diff', ascending=False).head())
        
        total_missing_time = gaps['time_diff'].sum() - (len(gaps) * expected_diff)
        print(f"Total missing time: {total_missing_time}")
        
        # Estimate missing bars
        missing_bars = total_missing_time / expected_diff
        print(f"Estimated missing bars: {missing_bars:.0f} ({missing_bars/len(df)*100:.2f}%)")
    else:
        print("No gaps found.")

    # 3. Check for NaNs
    print("\nMissing Values:")
    print(df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().sum())
    
    # 4. Check for Outliers/Invalid Data
    print("\nInvalid Data Checks:")
    print(f"Negative Prices: {(df[['Open', 'High', 'Low', 'Close']] < 0).sum().sum()}")
    print(f"Zero Prices: {(df[['Open', 'High', 'Low', 'Close']] == 0).sum().sum()}")
    print(f"Negative Volume: {(df['Volume'] < 0).sum()}")
    
    # 5. High/Low Logic
    invalid_hl = df[df['Low'] > df['High']]
    print(f"Low > High: {len(invalid_hl)}")
    
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check OHLCV data integrity")
    parser.add_argument("filepath", help="Path to CSV file")
    args = parser.parse_args()
    
    check_data_integrity(args.filepath)
