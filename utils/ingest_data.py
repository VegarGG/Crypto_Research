import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path to import config
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config, ARCTIC_URI

def ingest_data():
    print("Starting data ingestion...")
    
    data_dir = project_root / 'data'
    
    # Load datasets
    print("Loading 2021 data...")
    df_2021 = pd.read_csv(data_dir / 'BTCUSDT_2021_1m.csv')
    
    print("Loading 2022 data...")
    df_2022 = pd.read_csv(data_dir / 'BTCUSDT_2022_1m.csv')
    
    print("Loading 2023 data...")
    # 2023 data has an index column and Capitalized Timestamp
    df_2023 = pd.read_csv(data_dir / 'BTCUSD_1m_2023.csv')
    if 'Timestamp' in df_2023.columns:
        df_2023 = df_2023.rename(columns={'Timestamp': 'timestamp'})
    
    # Drop unnamed index column if exists
    df_2023 = df_2023[[col for col in df_2023.columns if 'Unnamed' not in col]]
    
    # Ensure consistent columns
    required_cols = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    for df, name in [(df_2021, '2021'), (df_2022, '2022'), (df_2023, '2023')]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Error: {name} data missing columns: {missing}")
            return
        # Keep only required columns
        df = df[required_cols]

    # Concatenate
    print("Merging datasets...")
    full_df = pd.concat([df_2021, df_2022, df_2023], ignore_index=True)
    
    # Convert timestamp
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    
    # Sort and drop duplicates
    full_df = full_df.sort_values('timestamp')
    full_df = full_df.drop_duplicates(subset='timestamp', keep='last')
    
    # Set index
    full_df = full_df.set_index('timestamp')
    
    print(f"Total rows: {len(full_df)}")
    print(f"Date range: {full_df.index.min()} to {full_df.index.max()}")
    
    # Save to CSV
    output_file = data_dir / 'BTCUSDT_2021_2023_1m.csv'
    print(f"Saving merged data to {output_file}...")
    full_df.to_csv(output_file)
    
    # Try ArcticDB if available
    try:
        from arcticdb import Arctic
        print(f"Connecting to ArcticDB at {ARCTIC_URI}...")
        arctic = Arctic(ARCTIC_URI)
        
        lib_name = 'market_data'
        if lib_name not in arctic.list_libraries():
            arctic.create_library(lib_name)
            print(f"Created library '{lib_name}'")
        
        library = arctic[lib_name]
        
        # Write to ArcticDB
        symbol = 'BTCUSDT'
        print(f"Writing {symbol} to ArcticDB...")
        library.write(symbol, full_df)
        print("ArcticDB write complete.")
        
    except ImportError:
        print("ArcticDB not available. Skipping ArcticDB ingestion.")
    except Exception as e:
        print(f"ArcticDB error: {e}")

    print("Ingestion/Merge complete.")

if __name__ == "__main__":
    ingest_data()
