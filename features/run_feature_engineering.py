import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.trend_indicator_pipeline_pkg import (
    TrendIndicatorPipeline,
    MomentumIndicatorPipeline,
    VolatilityIndicatorPipeline,
    FractalDimensionPipeline,
    TimeSeriesFeaturesPipeline
)

def run_feature_engineering():
    print("Starting Feature Engineering Pipeline...")
    
    # Load data
    data_path = project_root / 'data' / 'BTCUSDT_2021_2023_1m.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Filter to save time during dev/test if needed, but we want full run
    # df = df.iloc[-100000:] 
    
    print(f"Data shape: {df.shape}")
    
    # 1. Trend Indicators
    print("\n--- Running Trend Indicators ---")
    trend_pipe = TrendIndicatorPipeline()
    df = trend_pipe.run(df, symbol="BTCUSDT", sma_windows=[7, 30, 90], ema_spans=[7, 21], adx_windows=[14])
    
    # 2. Momentum Indicators (includes Dynamic RSI)
    print("\n--- Running Momentum Indicators ---")
    mom_pipe = MomentumIndicatorPipeline()
    df = mom_pipe.run(df, symbol="BTCUSDT", rsi_windows=[14, 30], stoch_windows=[14], macd_params=(12, 26, 9))
    
    # 3. Volatility Indicators
    print("\n--- Running Volatility Indicators ---")
    vol_pipe = VolatilityIndicatorPipeline()
    df = vol_pipe.run(df, symbol="BTCUSDT", bb_days_list=[20], atr_days_list=[14])
    
    # 4. Time-Series Features
    print("\n--- Running Time-Series Features ---")
    ts_pipe = TimeSeriesFeaturesPipeline()
    df = ts_pipe.run(df, symbol="BTCUSDT", trend_days=[1, 7, 30], vol_days=[7, 30])
    
    # 5. Fractal Dimension (Computationally Expensive)
    print("\n--- Running Fractal Dimension ---")
    fd_pipe = FractalDimensionPipeline(max_workers=4)
    # Use parallel processing for speed
    df = fd_pipe.run(df, symbol="BTCUSDT", days_list=[7], use_parallel=True)
    
    # Save final dataset
    output_file = project_root / 'data' / 'BTCUSDT_2021_2023_1m_features.csv'
    print(f"\nSaving feature-rich dataset to {output_file}...")
    df.to_csv(output_file)
    print("Feature Engineering Complete!")

if __name__ == "__main__":
    run_feature_engineering()
