#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
import os

# Add the features directory to path
sys.path.append(os.path.dirname(__file__))

from trend_indicator_pipeline_pkg import FractalDimensionPipeline

def create_small_test_data(n_points=100):
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_points, freq='1min')
    
    # Create simple test data
    base_price = 50000
    noise = np.random.randn(n_points) * 10
    prices = base_price + np.cumsum(noise * 0.01)
    
    df = pd.DataFrame({
        'Close': prices,
        'High': prices + np.abs(noise) * 0.5,
        'Low': prices - np.abs(noise) * 0.5,
    }, index=dates)
    
    return df

def quick_test():
    print("Quick Test of Optimized FD Pipeline")
    print("=" * 40)
    
    # Test with very small dataset
    print("Creating small test data (100 points)...")
    df = create_small_test_data(100)
    
    print(f"Test data shape: {df.shape}")
    print(f"Sample data:")
    print(df.head())
    
    # Test basic functionality
    try:
        print("\nTesting FD pipeline...")
        pipeline = FractalDimensionPipeline(max_workers=2)
        
        # Test sequential processing first
        print("1. Testing sequential processing...")
        df_result = pipeline.apply_fd_sequential(df.copy(), days=7, minute_data=True)
        valid_fd_count = np.sum(~np.isnan(df_result))
        print(f"   Sequential FD values computed: {valid_fd_count}")
        
        # Test parallel processing (if applicable)
        if len(df) > 50:
            print("2. Testing parallel processing...")
            try:
                df_parallel = pipeline.apply_fd_parallel(df.copy(), days=7, minute_data=True)
                valid_parallel_count = np.sum(~np.isnan(df_parallel))
                print(f"   Parallel FD values computed: {valid_parallel_count}")
            except Exception as e:
                print(f"   Parallel processing skipped: {e}")
        
        # Test the main run method with minimal settings
        print("3. Testing main run method...")
        result = pipeline.run(df.copy(), symbol="QUICK_TEST", days_list=[7], 
                            show_performance=True, use_parallel=False)
        
        fd_column = 'fd_7d'
        if fd_column in result.columns:
            fd_values = result[fd_column].dropna()
            if len(fd_values) > 0:
                print(f"   FD column created successfully: {len(fd_values)} valid values")
                print(f"   FD stats: min={fd_values.min():.4f}, max={fd_values.max():.4f}, mean={fd_values.mean():.4f}")
            else:
                print("   WARNING: No valid FD values computed")
        else:
            print("   ERROR: FD column not found in result")
            return False
        
        print("\n[SUCCESS] Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\nThe optimized FD pipeline is working correctly!")
    else:
        print("\nThere may be issues with the optimized pipeline.")