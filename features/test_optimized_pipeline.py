#!/usr/bin/env python3

import pandas as pd
import numpy as np
from trend_indicator_pipeline_pkg import FractalDimensionPipeline

def create_test_data(n_points=10000):
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_points, freq='1min')
    
    # Create realistic OHLCV-like data
    base_price = 50000
    noise = np.random.randn(n_points) * 100
    trend = np.linspace(0, 5000, n_points)
    prices = base_price + trend + np.cumsum(noise * 0.1)
    
    # Ensure proper OHLC relationships
    high_noise = np.random.rand(n_points) * 50
    low_noise = np.random.rand(n_points) * 50
    
    df = pd.DataFrame({
        'Close': prices,
        'High': prices + high_noise,
        'Low': prices - low_noise,
        'Volume': np.random.randint(1000, 10000, n_points)
    }, index=dates)
    
    # Ensure High >= Low and Close is between them
    df['High'] = np.maximum(df['High'], df['Close'])
    df['Low'] = np.minimum(df['Low'], df['Close'])
    
    return df

def test_optimized_pipeline():
    print("Testing Optimized Fractal Dimension Pipeline")
    print("=" * 50)
    
    # Create test data
    print("Creating test data...")
    df_small = create_test_data(1000)  # Small dataset
    df_large = create_test_data(20000)  # Large dataset for parallel processing
    
    # Initialize pipeline
    pipeline = FractalDimensionPipeline(max_workers=4)
    
    # Test 1: Small dataset (sequential processing)
    print("\n1. Testing small dataset (sequential processing expected):")
    try:
        result_small = pipeline.run(df_small, symbol="TEST_SMALL", days_list=[7], show_performance=True)
        print("[OK] Small dataset test passed")
    except Exception as e:
        print(f"[ERROR] Small dataset test failed: {e}")
        return False
    
    # Test 2: Large dataset (parallel processing)
    print("\n2. Testing large dataset (parallel processing expected):")
    try:
        result_large = pipeline.run(df_large, symbol="TEST_LARGE", days_list=[7], 
                                  use_parallel=True, show_performance=True)
        print("[OK] Large dataset test passed")
    except Exception as e:
        print(f"[ERROR] Large dataset test failed: {e}")
        return False
    
    # Test 3: Performance benchmark
    print("\n3. Running performance benchmark:")
    try:
        pipeline.benchmark_performance(df_large, days=7, iterations=2)
        print("[OK] Performance benchmark completed")
    except Exception as e:
        print(f"[ERROR] Performance benchmark failed: {e}")
        return False
    
    # Test 4: Validation test
    print("\n4. Validating results consistency:")
    try:
        is_valid = pipeline.validate_results(df_large, days=7)
        if is_valid:
            print("[OK] Results validation passed")
        else:
            print("[ERROR] Results validation failed")
            return False
    except Exception as e:
        print(f"[ERROR] Results validation failed: {e}")
        return False
    
    # Test 5: Multiple timeframes
    print("\n5. Testing multiple timeframes:")
    try:
        result_multi = pipeline.run(df_large, symbol="TEST_MULTI", days_list=[3, 7, 14], 
                                  use_parallel=True, show_performance=True)
        print("[OK] Multiple timeframes test passed")
    except Exception as e:
        print(f"[ERROR] Multiple timeframes test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("[SUCCESS] All tests passed! The optimized pipeline is working correctly.")
    
    # Display some sample results
    print(f"\nSample FD results for large dataset:")
    for col in result_large.columns:
        if col.startswith('fd_'):
            non_nan_values = result_large[col].dropna()
            if len(non_nan_values) > 0:
                print(f"{col}: mean={non_nan_values.mean():.4f}, "
                      f"std={non_nan_values.std():.4f}, "
                      f"min={non_nan_values.min():.4f}, "
                      f"max={non_nan_values.max():.4f}")
    
    return True

if __name__ == "__main__":
    test_optimized_pipeline()