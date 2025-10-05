
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

class PreClassifiedStrategyLoader:
    """Load and use saved pre-classified hybrid strategy pipeline"""

    def __init__(self, pipeline_dir='../pipeline'):
        self.pipeline_dir = pipeline_dir
        self.config = None

    def load_pipeline(self):
        """Load the complete strategy pipeline"""
        # Load configuration
        config_path = os.path.join(self.pipeline_dir, 'preclassified_strategy_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Loaded pre-classified strategy configuration")
            print(f"Strategy type: {self.config.get('strategy_type', 'Unknown')}")
            print(f"Regime source: {self.config.get('regime_source', 'Unknown')}")
        else:
            print(f"Configuration file not found: {config_path}")

        return self

    def load_regime_data(self, data_path: str = None) -> pd.DataFrame:
        """Load pre-classified regime data"""
        if data_path is None:
            data_path = self.config.get('regime_source', 'BTCUSD_2023_1min_enhanced_regimes_1D_momentum.csv')
            data_path = f"../data/{data_path}"

        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            print(f"Loaded regime data: {df.shape}")
            return df
        else:
            raise FileNotFoundError(f"Regime data not found: {data_path}")

    def get_strategy_config(self) -> Dict:
        """Get strategy configuration"""
        return self.config

# Usage example:
# loader = PreClassifiedStrategyLoader().load_pipeline()
# regime_data = loader.load_regime_data()
