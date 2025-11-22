import os
import pandas as pd
import urllib.request
import zipfile
import io
from datetime import datetime
import argparse
from pathlib import Path

def download_binance_monthly_data(symbol, year, output_dir):
    base_url = "https://data.binance.vision/data/spot/monthly/klines"
    interval = "1m"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    
    print(f"Downloading data for {symbol} {year}...")
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        filename = f"{symbol}-{interval}-{year}-{month_str}.zip"
        url = f"{base_url}/{symbol}/{interval}/{filename}"
        
        print(f"  Fetching {year}-{month_str}...", end="", flush=True)
        
        try:
            with urllib.request.urlopen(url) as response:
                zip_content = response.read()
                
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                # The zip usually contains one csv file with the same name as the zip but .csv extension
                csv_filename = filename.replace(".zip", ".csv")
                with zf.open(csv_filename) as csv_file:
                    # Binance data has no header
                    # Columns: Open time, Open, High, Low, Close, Volume, Close time, ...
                    df = pd.read_csv(csv_file, header=None)
                    
                    # Rename columns
                    df.columns = [
                        "timestamp", "Open", "High", "Low", "Close", "Volume", 
                        "close_time", "quote_asset_volume", "trades", 
                        "taker_buy_base", "taker_buy_quote", "ignore"
                    ]
                    
                    # Keep only OHLCV and timestamp
                    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
                    
                    # Convert timestamp (ms to datetime)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    
                    dfs.append(df)
            print(" Done.")
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f" Not found (might be future date or not available).")
            else:
                print(f" Error: {e}")
        except Exception as e:
            print(f" Error: {e}")

    if not dfs:
        print("No data downloaded.")
        return

    print("Merging data...")
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values("timestamp")
    
    output_file = output_dir / f"{symbol}_{year}_1m.csv"
    full_df.to_csv(output_file, index=False)
    print(f"Saved {len(full_df)} rows to {output_file}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance historical data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--years", nargs="+", default=["2021", "2022"], help="Years to download")
    parser.add_argument("--output", default="data", help="Output directory")
    
    args = parser.parse_args()
    
    for year in args.years:
        download_binance_monthly_data(args.symbol, year, args.output)
